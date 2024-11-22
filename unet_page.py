import os
# from models.unet import UNet  # класс модели Unet
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import sys
import torch.nn as nn
import torch.nn.functional as F
# sys.path.append('./models')  # Убедитесь, что Streamlit видит папку models


class UNet(nn.Module):
    def __init__(self, n_class):
        super(UNet, self).__init__()

        # Encoder
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        # Ожидается 1024 канала после конкатенации
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # Ожидается 512 канала после конкатенации
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # Ожидается 256 канала после конкатенации
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # Ожидается 128 канала после конкатенации
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

        # Сигмоида для бинарной сегментации
        self.sigmoid = nn.Sigmoid()  # Для бинарной сегментации

    def forward(self, x):
        # Encoder
        xe11 = F.relu(self.e11(x))
        xe12 = F.relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = F.relu(self.e21(xp1))
        xe22 = F.relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = F.relu(self.e31(xp2))
        xe32 = F.relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = F.relu(self.e41(xp3))
        xe42 = F.relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = F.relu(self.e51(xp4))
        xe52 = F.relu(self.e52(xe51))

        # Decoder
        xu1 = self.upconv1(xe52)
        # Concatenate corresponding encoder feature maps
        xu1 = torch.cat([xu1, xe42], dim=1)
        xd11 = F.relu(self.d11(xu1))
        xd12 = F.relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        # Concatenate corresponding encoder feature maps
        xu2 = torch.cat([xu2, xe32], dim=1)
        xd21 = F.relu(self.d21(xu2))
        xd22 = F.relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        # Concatenate corresponding encoder feature maps
        xu3 = torch.cat([xu3, xe22], dim=1)
        xd31 = F.relu(self.d31(xu3))
        xd32 = F.relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        # Concatenate corresponding encoder feature maps
        xu4 = torch.cat([xu4, xe12], dim=1)
        xd41 = F.relu(self.d41(xu4))
        xd42 = F.relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)
        out = self.sigmoid(out)  # Применяем сигмоиду для бинарной сегментации
        return out
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
