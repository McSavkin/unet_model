import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from unet import UNet  # класс модели Unet
from tensorboard import program
import os

def unet_segmentation():

    # Загрузка модели Unet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_unet = UNet(n_class=1).to(device)
    model_unet.load_state_dict(torch.load('model_weights.pth', map_location=device))
    model_unet.eval()  # Устанавливаем модель в режим оценки

    # Функция для получения предсказания
    def predict_image(model, image, device):
        # Загружаем изображение
        image = Image.open(image).convert("RGB")
    
        # Предобработка изображения (например, изменение размера и нормализация)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),   # Размер изображения
            transforms.ToTensor(),           # Преобразуем в тензор
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация
        ])
    
        # Применяем преобразования
        image_tensor = transform(image).unsqueeze(0).to(device)  # Добавляем размерность для батча (1, C, H, W)

        # Получаем предсказание
        model.eval()  # Переводим модель в режим оценки
        with torch.no_grad():  # Без расчета градиентов
            output = model(image_tensor)  # Предсказание модели
        
        # Применение сигмоида (для бинарной сегментации)
        output = torch.sigmoid(output)  # Если выход модели был без активации, используем сигмоиду для нормализации
    
        # Преобразуем предсказания в 0 или 1 (для бинарной сегментации)
        output = (output > 0.5).float()  # Используем порог 0.5 для бинаризации
    
        return output.squeeze(0)  # Убираем размерность батча


    # Streamlit интерфейс
    st.title("Применение модели Unet к задаче сегментации")
    st.markdown("Upload one or more images or provide image URLs to get their segmentation masks.")

    # Выбор изображения (по ссылке или загрузка)
    image_choice = st.radio("Choose how to upload the image(s):", ('Upload Image(s)', 'Provide Image URL(s)'))

    if image_choice == 'Upload Image(s)':
        uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

        if uploaded_files is not None:
            for uploaded_file in uploaded_files:
                # Сохранить изображение на диск
                image_path = 'temp_image.jpg'
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
                # Получаем предсказание
                output_mask = predict_image(model_unet, image_path, device)
                
                # Преобразуем маску в правильный формат для отображения
                output_mask = output_mask.cpu().numpy() * 255  # Преобразуем в диапазон [0, 255] для отображения
                output_mask = output_mask.astype(np.uint8)  # Преобразуем в целочисленный тип для отображения

                # Убираем лишнюю размерность, чтобы сделать маску двумерной
                output_mask = output_mask[0]  # Получаем только размерности (256, 256)

                # Создание колонок для левого и правого изображения
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(uploaded_file, caption="Uploaded Image.", use_container_width=True)
                
                with col2:
                    st.image(output_mask, caption="Predicted Mask", use_container_width=True, clamp=True)

    elif image_choice == 'Provide Image URL(s)':
        image_urls = st.text_area("Enter the image URLs (one per line):").splitlines()

        if image_urls:
            for image_url in image_urls:
                try:
                    # Загружаем изображение по URL
                    response = requests.get(image_url)
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                
                    # Сохраняем изображение для предсказания
                    image_path = 'temp_image_from_url.jpg'
                    image.save(image_path)
                
                    # Получаем предсказание
                    output_mask = predict_image(model_unet, image_path, device)
                
                    # Преобразуем маску в правильный формат для отображения
                    output_mask = output_mask.cpu().numpy() * 255  # Преобразуем в диапазон [0, 255] для отображения
                    output_mask = output_mask.astype(np.uint8)  # Преобразуем в целочисленный тип для отображения

                    # Убираем лишнюю размерность, чтобы сделать маску двумерной
                    output_mask = output_mask[0]  # Получаем только размерности (256, 256)

                    # Создание колонок для левого и правого изображения
                    col1, col2 = st.columns(2)
                
                    with col1:
                        st.image(image, caption="Image from URL", use_container_width=True)
                
                    with col2:
                        st.image(output_mask, caption="Predicted Mask", use_container_width=True, clamp=True)

                except Exception as e:
                    st.error(f"Error loading image from URL: {e}")
    # Функция для запуска TensorBoard
    def run_tensorboard(logdir):
    # Запуск TensorBoard в фоновом процессе
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', logdir])
        url = tb.launch()
        return url

    # Создание интерфейса Streamlit
    st.title("Интеграция TensorBoard с Streamlit")

    # Введите директорию с логами
    log_dir = "logs/"

    # Проверка на наличие папки
    if os.path.exists(log_dir):
        st.write("Запускаем TensorBoard...")

        # Получаем URL, на котором будет доступен TensorBoard
        tensorboard_url = run_tensorboard(log_dir)

        # Встраиваем TensorBoard в iframe
        st.write(f"Вы можете увидеть TensorBoard здесь:")
        st.components.v1.iframe(tensorboard_url, width=1500, height=1500)
    else:
        st.write("Папка с логами не найдена. Убедитесь, что указали правильный путь.")
                
# Вызов функции для запуска сегментации
unet_segmentation()
