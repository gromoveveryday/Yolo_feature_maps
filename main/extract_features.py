import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
import cv2
from pathlib import Path
import argparse
from collections import OrderedDict

# Скрипт для извлечения всех промежуточных карт признаков (feature maps) из YOLO (ultralytics)
# Регистрирует хуки на все модули модели, прогоняет одно изображение и сохраняет выходы, также несколько визуализирует

# Возвращает множество полных имён модулей, которые являются Detect или его потомками
def find_detect_names(model):
    detect_names = set()
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'Detect':
            detect_names.add(name)
    return detect_names

# Вспомогательный класс для сбора выходов с помощью хуков
class FeatureHook:
    def __init__(self, module, name):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.name = name
        self.outputs = None

    def hook_fn(self, module, input, output):
        # Сохраняем копию выходного тензора (detach, чтобы не влиять на граф)
        if isinstance(output, torch.Tensor):
            self.outputs = output.detach().cpu()
        elif isinstance(output, (tuple, list)):
            # Если модуль возвращает несколько тензоров, берём первый (обычно признаковый)
            # или сохраняем список, но для простоты берём первый.
            if len(output) > 0 and isinstance(output[0], torch.Tensor):
                self.outputs = output[0].detach().cpu()
            else:
                self.outputs = None
        else:
            self.outputs = None

    def remove(self):
        self.hook.remove()

# Рекурсивно обходит все дочерние модули модели и регистрирует хуки. Возвращает список объектов FeatureHook
def register_all_hooks(model, verbose=True):
    detect_names = find_detect_names(model)
    hooks = []
    for name, module in model.named_modules():
        # Пропускаем контейнеры, которые просто группируют слои (их выход часто неинформативен)
        # Регистрируем все модули, у которых есть forward, но не Sequential/ModuleList/ModuleDict/Detect
        if isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList, torch.nn.ModuleDict)):
            continue
        
        if any(name.startswith(dn) for dn in detect_names):
            if verbose:
                print(f"Пропускаем модуль {name} (принадлежит Detect)")
            continue
        # У некоторых модулей (например, Dropout) выход может быть None или не тензор – пропускаем
        if hasattr(module, 'forward'):
            hook = FeatureHook(module, name)
            hooks.append(hook)
            if verbose:
                print(f"Зарегистрирован хук на {name} ({module.__class__.__name__})")
    return hooks

# Подготавливает изображение для модели YOLO. Возвращает тензор, готовый для forward (1,3,H,W)
def preprocess_image(image_path, model, target_size=640):
    # Загрузка изображения
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")

    # OpenCV загружает изображения в формате BGR, YOLO обучается на RGB.
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Изменение размера с сохранением пропорций и добавлением паддинга)
    letterbox = LetterBox(new_shape=(target_size, target_size), auto=False, stride=32)
    img_letterboxed = letterbox(image=img_rgb)  # Результат — numpy array (H, W, 3) в формате RGB
    # Изменяем порядок осей с (Height, Width, Channels) на (Channels, Height, Width)
    img_chw = img_letterboxed.transpose((2, 0, 1))
    # Нормализуем значения пикселей из диапазона [0, 255] в [0, 1]
    img_normalized = img_chw.astype(np.float32) / 255.0
    # Добавление batch dimension и преобразование в тензор PyTorch ---
    img_tensor = torch.from_numpy(img_normalized).unsqueeze(0)
    return img_tensor

# python main/extract_features.py --image images/sample.jpg --model yolo.pt --save_images
def main():
    parser = argparse.ArgumentParser(description="Извлечение промежуточных карт признаков YOLO")
    parser.add_argument('--model', type=str, help='Путь к модели YOLO или имя предобученной')
    parser.add_argument('--image', type=str, required=True, help='Путь к входному изображению')
    parser.add_argument('--output_dir', type=str, default='outputs/features', help='Папка для сохранения карт признаков')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_npy', action='store_true', default=True, help='Сохранять как .npy')
    parser.add_argument('--save_images', action='store_true', help='Визуализировать и сохранить первые каналы каждой карты')
    args = parser.parse_args()

    # Создаём выходную папку
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Загружаем модель YOLO (полная модель, включая голову)
    print(f"Загрузка модели {args.model} на устройство {args.device}")
    model_yolo = YOLO(args.model)
    # Получаем PyTorch модель (backbone + head)
    torch_model = model_yolo.model
    torch_model = torch_model.to(args.device)
    torch_model.eval()

    # Регистрируем хуки на всех модулях
    hooks = register_all_hooks(torch_model, verbose=True)
    print(f"Всего зарегистрировано хуков: {len(hooks)}")

    # Подготавливаем изображение
    input_tensor = preprocess_image(args.image, model_yolo)
    input_tensor = input_tensor.to(args.device)

    # Выполняем forward (детекция при этом тоже произойдёт, но нас интересуют промежуточные выходы)
    print("Прогон изображения через модель...")
    with torch.no_grad():
        _ = torch_model(input_tensor)

    # Собираем результаты хуков
    feature_maps = OrderedDict()
    for hook in hooks:
        if hook.outputs is not None:
            feature_maps[hook.name] = hook.outputs
            print(f"Сохранён выход для {hook.name}: форма {hook.outputs.shape}")
        else:
            print(f"Выход для {hook.name} отсутствует (None или не тензор)")

    # Сохраняем карты признаков
    if args.save_npy:
        npy_dir = out_path / 'npy'
        npy_dir.mkdir(exist_ok=True)
        for name, fm in feature_maps.items():
            # Сохраняем как .npy
            safe_name = name.replace('.', '_').replace('/', '_')
            np.save(npy_dir / f"{safe_name}.npy", fm.numpy())
        print(f"Сохранено {len(feature_maps)} карт признаков в {npy_dir}")

    # Опционально: визуализация и сохранение первых каналов карт признаков
    if args.save_images:
        import matplotlib.pyplot as plt
        vis_dir = out_path / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        for name, fm in feature_maps.items():
            # fm форма: [batch, channels, height, width] или [batch, channels, height, width]?
            if fm.dim() == 4:
                # Берем первые 16 каналов или меньше
                num_channels = min(16, fm.shape[1])
                fig, axes = plt.subplots(4, 4, figsize=(12, 12))
                axes = axes.flatten()
                for i in range(num_channels):
                    channel_img = fm[0, i, :, :].numpy()
                    axes[i].imshow(channel_img, cmap='viridis')
                    axes[i].axis('off')
                    axes[i].set_title(f'ch{i}')
                for j in range(num_channels, 16):
                    axes[j].axis('off')
                plt.suptitle(f"Feature map: {name}")
                plt.tight_layout()
                safe_name = name.replace('.', '_').replace('/', '_')
                plt.savefig(vis_dir / f"{safe_name}.png")
                plt.close()
            else:
                print(f"Пропускаем визуализацию для {name}: не 4D тензор ({fm.shape})")
        print(f"Визуализации сохранены в {vis_dir}")

    # Очистка хуков
    for hook in hooks:
        hook.remove()
    print("Готово!")

if __name__ == '__main__':
    main()