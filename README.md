# Инструмент для извлечения карт признаков из моделей YOLO (detection/segmentation) и их интерпретации

Локально разворачиваемое приложение для извлечения карт признаков из моделей семейства You Only Look Once (YOLO) и получения отчета по их интерпретации

## Требования

- Python версии 3.12.8
- Установленный git   

## Установка

1. Клонировать репозиторий:

```shell
git clone https://github.com/gromoveveryday/Yolo_feature_maps.git
```

```shell
cd Yolo_feature_maps
```

2. Создать и активировать виртуальное окружение:

```shell
python -m venv venv
```

```shell
venv\Scripts\activate
```

3. Установить/обновить зависимости:

```shell
pip install -r requirements.txt