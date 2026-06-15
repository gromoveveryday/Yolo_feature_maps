import torch
import numpy as np
from ultralytics import YOLO
import argparse
from pathlib import Path

# Скрипт для извлечения информации из детектирующей головы YOLO (модуль Detect)
# Печатает архитектуру, параметры слоёв (веса, bias), anchors, strides
# Также может выполнить forward и сохранить выходы головы

# Из модели YOLO извлекает модуль Detect (детектирующая голова).
# В ultralytics это последний модуль model.model (torch_model) и обычно имеет тип 'Detect'
def get_detect_module(model_yolo):
    torch_model = model_yolo.model
    # Модель может быть wrapped, но model.model - это Sequential
    # Последний элемент - Detect
    detect_module = None
    for name, module in torch_model.named_modules():
        if module.__class__.__name__ == 'Detect':
            detect_module = module
            print(f"Найден модуль Detect: {name}")
            break
    if detect_module is None:
        raise RuntimeError("Не удалось найти модуль Detect в модели. Проверьте версию ultralytics.")
    return detect_module

# Выводит статистику тензора: форма, min, max, mean, std
def print_tensor_stats(name, tensor):
    if tensor is None:
        print(f"{name}: None")
        return
    if tensor.is_floating_point():
        # Для float-тензоров выводим полную статистику
        print(f"{name}: shape={tuple(tensor.shape)}, "
              f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, "
              f"mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}")
    else:
        # Для целочисленных тензоров выводим только min, max и dtype
        print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
              f"min={tensor.min().item()}, max={tensor.max().item()}")

# Извлекает и выводит все параметры (веса и bias) из слоёв детектирующей головы.
#    В типичной голове YOLOv8:
#        - cv2 (bbox): несколько свёрточных слоёв, последний слой имеет bias для координат
#        - cv3 (cls): несколько свёрточных слоёв, последний слой имеет bias для классов
#        - dfl (Distribution Focal Loss) - обычно без обучаемых параметров, но может содержать свертку
#    Также в detect_module есть атрибуты: anchors, stride, nc (число классов), no (число выходов) и т.д.

def extract_head_parameters(detect_module):
    print("\n=== Детектирующая голова: параметры ===")
    # Общие атрибуты
    print(f"nc (число классов): {detect_module.nc}")
    print(f"no (число выходов на якорь): {detect_module.no}")
    print(f"reg_max (для DFL): {detect_module.reg_max}")
    print(f"stride (шаги для каждого уровня пирамиды): {detect_module.stride}")
    if hasattr(detect_module, 'anchors'):
        # anchors может быть тензором
        anchors = detect_module.anchors
        if anchors is not None:
            print(f"anchors: shape={anchors.shape}")
            print(f"anchors (первые 5): {anchors.flatten()[:10]}")
    # Проход по всем подмодулям Detect
    for name, param in detect_module.named_parameters():
        print_tensor_stats(f"Параметр {name}", param.data)
    # Также можно посмотреть буферы (например, anchors, stride если они буферы)
    for name, buf in detect_module.named_buffers():
        print_tensor_stats(f"Буфер {name}", buf)

#  Регистрирует forward-hooks на все свёрточные слои внутри Detect,
#  чтобы получить выходы головы при прогоне изображения.
#  Возвращает список (имя, выход) после forward.
def register_head_hooks(detect_module):
    outputs = {}

    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                outputs[name] = output.detach().cpu()
            elif isinstance(output, (tuple, list)):
                outputs[name] = [o.detach().cpu() if isinstance(o, torch.Tensor) else o for o in output]
            else:
                outputs[name] = output
        return hook

    hooks = []
    for name, module in detect_module.named_modules():
        # Регистрируем на всех подмодулях Detect, кроме самого detect_module (чтобы избежать дублей)
        if module is not detect_module and not isinstance(module, torch.nn.Sequential):
            hook = module.register_forward_hook(hook_fn(f"{name}"))
            hooks.append(hook)
            print(f"Зарегистрирован хук на {name} ({module.__class__.__name__})")
    return hooks, outputs

# python main/extract_head_info.py --model models/yolov8n.pt --image images/sample.jpg --save_outputs
def main():
    parser = argparse.ArgumentParser(description="Извлечение информации из детектирующей головы YOLO")
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Путь к модели YOLO')
    parser.add_argument('--image', type=str, help='Опционально: извлечь выходы головы на этом изображении')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_outputs', action='store_true', help='Сохранить выходы головы в .npy')
    args = parser.parse_args()

    # Загрузка модели
    print(f"Загрузка модели {args.model} на {args.device}")
    model_yolo = YOLO(args.model)
    detect = get_detect_module(model_yolo)
    detect = detect.to(args.device)
    detect.eval()

    # Извлечение параметров (веса и пр.)
    extract_head_parameters(detect)

    # Если нужно получить выходы головы при forward
    if args.image:
        print(f"\n=== Forward головы на изображении {args.image} ===")
        # Подготавливаем входное изображение (используем подготовку из первого скрипта)
        from extract_features import preprocess_image  # переиспользуем функцию
        input_tensor = preprocess_image(args.image, model_yolo)
        input_tensor = input_tensor.to(args.device)

        # Регистрируем хуки на внутренних слоях Detect
        hooks, hook_outputs = register_head_hooks(detect)

        # Forward: нам нужно передать через Detect выходы backbone/neck.
        # Но напрямую detect принимает список тензоров (признаки с разных уровней).
        # Чтобы получить корректные входы для Detect, нужно выполнить полный forward модели
        # и перехватить вход detect. Проще: сделать forward всей модели, но тогда хуки на detect
        # сработают автоматически, если они зарегистрированы на самом detect.
        # Однако detect - это часть torch_model, и forward всей модели вызовет detect внутри.
        # Поэтому мы делаем forward всей модели, а хуки на detect_module соберут выходы.
        # ВНИМАНИЕ: detect_module уже в составе torch_model, и при вызове torch_model(input) хуки сработают.
        torch_model = model_yolo.model.to(args.device)
        torch_model.eval()

        # Хуки уже зарегистрированы на detect (и его подмодулях). Выполняем forward.
        print("Выполняется forward модели...")
        with torch.no_grad():
            _ = torch_model(input_tensor)

        # Выводим собранные выходы
        print(f"Получено выходов: {len(hook_outputs)}")
        for name, out in hook_outputs.items():
            if isinstance(out, torch.Tensor):
                print_tensor_stats(f"Выход {name}", out)
            elif isinstance(out, list):
                print(f"Выход {name}: список из {len(out)} элементов, первый элемент: {out[0].shape if isinstance(out[0], torch.Tensor) else type(out[0])}")
            else:
                print(f"Выход {name}: {type(out)}")

        # Сохраняем выходы, если нужно
        if args.save_outputs:
            save_dir = Path("outputs/head_outputs")
            save_dir.mkdir(parents=True, exist_ok=True)
            for name, out in hook_outputs.items():
                if isinstance(out, torch.Tensor):
                    np.save(save_dir / f"{name.replace('.', '_')}.npy", out.numpy())
                elif isinstance(out, list):
                    for i, o in enumerate(out):
                        if isinstance(o, torch.Tensor):
                            np.save(save_dir / f"{name.replace('.', '_')}_{i}.npy", o.numpy())
            print(f"Выходы сохранены в {save_dir}")

        # Снимаем хуки
        for h in hooks:
            h.remove()
    else:
        print("\nНе указано изображение, выходы головы не вычислены. Добавьте --image, чтобы получить их.")

if __name__ == '__main__':
    main()