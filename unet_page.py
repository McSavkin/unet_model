import os
from unet import UNet  # класс модели Unet
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import sys
sys.path.append('./models')  # Убедитесь, что Streamlit видит папку models

# Глобальная переменная для модели
model_unet = None


def load_model():
    global model_unet
    if model_unet is None:
        # Проверка на доступность GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_unet = UNet(n_class=1).to(device)
        model_unet.load_state_dict(torch.load(
            'models/model_weights.pth', map_location=device, weights_only=True))
        model_unet.eval()  # Устанавливаем модель в режим оценки
    return model_unet

# Функция для предсказания сегментации


def predict_image(_model, image, device):
    # Загружаем изображение
    image = Image.open(image).convert("RGB")

    # Предобработка изображения (изменение размера и нормализация)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),   # Размер изображения
        transforms.ToTensor(),           # Преобразуем в тензор
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])  # Нормализация
    ])

    # Применяем преобразования
    # Добавляем размерность для батча (1, C, H, W)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Получаем предсказание
    _model.eval()  # Переводим модель в режим оценки
    with torch.no_grad():  # Без расчета градиентов
        output = _model(image_tensor)  # Предсказание модели

    # Применение сигмоида (для бинарной сегментации)
    # Если выход модели был без активации, используем сигмоиду для нормализации
    output = torch.sigmoid(output)

    # Преобразуем предсказания в 0 или 1 (для бинарной сегментации)
    output = (output > 0.5).float()  # Используем порог 0.5 для бинаризации

    return output.squeeze(0)  # Убираем размерность батча


def unet_segmentation():
    # Загрузка модели один раз
    model_unet = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Streamlit интерфейс
    st.title("Применение модели Unet к задаче сегментации")
    st.markdown(
        "Upload one or more images or provide image URLs to get their segmentation masks.")

    # Выбор изображения (по ссылке или загрузка)
    image_choice = st.radio("Choose how to upload the image(s):",
                            ('Upload Image(s)', 'Provide Image URL(s)'))

    if image_choice == 'Upload Image(s)':
        uploaded_files = st.file_uploader("Choose images...", type=[
                                          "jpg", "png", "jpeg"], accept_multiple_files=True)

        if uploaded_files is not None:
            for uploaded_file in uploaded_files:
                # Получаем предсказание
                output_mask = predict_image(model_unet, uploaded_file, device)

                # Преобразуем маску в правильный формат для отображения
                # Преобразуем в диапазон [0, 255] для отображения
                output_mask = output_mask.cpu().numpy() * 255
                # Преобразуем в целочисленный тип для отображения
                output_mask = output_mask.astype(np.uint8)

                # Убираем лишнюю размерность, чтобы сделать маску двумерной
                # Получаем только размерности (256, 256)
                output_mask = output_mask[0]

                # Создание колонок для левого и правого изображения
                col1, col2 = st.columns(2)

                with col1:
                    st.image(uploaded_file, caption="Uploaded Image.",
                             use_container_width=True)

                with col2:
                    st.image(output_mask, caption="Predicted Mask",
                             use_container_width=True, clamp=True)

    elif image_choice == 'Provide Image URL(s)':
        image_urls = st.text_area(
            "Enter the image URLs (one per line):").splitlines()

        if image_urls:
            for image_url in image_urls:
                try:
                    # Загружаем изображение по URL
                    response = requests.get(image_url)
                    image = Image.open(
                        BytesIO(response.content)).convert("RGB")

                    # Получаем предсказание
                    output_mask = predict_image(model_unet, image, device)

                    # Преобразуем маску в правильный формат для отображения
                    # Преобразуем в диапазон [0, 255] для отображения
                    output_mask = output_mask.cpu().numpy() * 255
                    # Преобразуем в целочисленный тип для отображения
                    output_mask = output_mask.astype(np.uint8)

                    # Убираем лишнюю размерность, чтобы сделать маску двумерной
                    # Получаем только размерности (256, 256)
                    output_mask = output_mask[0]

                    # Создание колонок для левого и правого изображения
                    col1, col2 = st.columns(2)

                    with col1:
                        st.image(image, caption="Image from URL",
                                 use_container_width=True)

                    with col2:
                        st.image(output_mask, caption="Predicted Mask",
                                 use_container_width=True, clamp=True)

                except Exception as e:
                    st.error(f"Error loading image from URL: {e}")

    st.title('Метрики модели Unet для задаче сегментации')
    st.divider()

    st.subheader('Accuracy для train и valid')
    st.image('images/accuracy_train_unet.jpg')
    st.image('images/accuracy_val_unet.jpg')

    st.subheader('Dice для train и valid')
    st.image('images/dice_train_unet.jpg')
    st.image('images/dice_val_unet.jpg')

    st.subheader('IOU для train и valid')
    st.image('images/iou_train_unet.jpg')
    st.image('images/iou_val_unet.jpg')

    st.subheader('Loss для train и valid')
    st.image('images/loss_train_unet.jpg')
    st.image('images/loss_val_unet.jpg')

    st.divider()


# Вызов функции для запуска сегментации
unet_segmentation()
