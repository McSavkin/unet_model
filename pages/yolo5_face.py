import streamlit as st
from PIL import Image
import cv2
import numpy as np
import torch

def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt')  # Путь к вашей модели
    model.conf = 0.7  # Уровень уверенности для фильтрации предсказаний
    return model

model = load_model()

# Приём файлов от пользователя
uploaded_files = st.file_uploader(
    "Загрузите изображения",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# Проверяем, загружены ли файлы
if uploaded_files:
    images = []  # Список для сохранения изображений
    
    # Обработка каждого загруженного изображения
    for uploaded_file in uploaded_files:
        # Открываем изображение с помощью Pillow
        image = Image.open(uploaded_file)
        image_np = np.array(image)  # Преобразуем изображение в формат NumPy

        # Проверяем, является ли изображение цветным
        if len(image_np.shape) == 3:  # Цветное изображение
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_np

        # Детектируем лица на изображении
        results = model(image_bgr)
        detections = results.xyxy[0].cpu().numpy()  # Формат: [x_min, y_min, x_max, y_max, confidence, class]

        # Обработка каждого детектированного лица
        for det in detections:
            x_min, y_min, x_max, y_max = map(int, det[:4])
            conf = det[4]
            label = f"Face: {conf:.2f}"

            # Вырезаем область лица и размываем её
            face = image_bgr[y_min:y_max, x_min:x_max]
            blurred_face = cv2.GaussianBlur(face, (51, 51), 30)
            image_bgr[y_min:y_max, x_min:x_max] = blurred_face

            # Рисуем рамку
            color = (0, 255, 0)  # Цвет рамки (зеленый)
            thickness = 2  # Толщина рамки
            cv2.rectangle(image_bgr, (x_min, y_min), (x_max, y_max), color, thickness)

            # Добавляем текст с уверенностью
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_thickness = 2
            text_size = cv2.getTextSize(label, font, font_scale, text_thickness)[0]
            text_x, text_y = x_min, y_min - 10  # Координаты текста
            
            # Фон для текста
            cv2.rectangle(image_bgr, (text_x, text_y - text_size[1] - 5), 
                          (text_x + text_size[0], text_y + 5), (0, 255, 0), -1)
            cv2.putText(image_bgr, label, (text_x, text_y), font, font_scale, (0, 0, 0), text_thickness)

        # Конвертируем BGR обратно в RGB для отображения
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption=f"Результат для {uploaded_file.name}", use_container_width=True)
