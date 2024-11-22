import streamlit as st
from PIL import Image
import cv2
import numpy as np
import torch
import requests
from io import BytesIO
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Загружаем YOLOv5 модель
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt')
model.conf = 0.7  # Уровень уверенности для фильтрации предсказаний


st.title("Результаты дообучения модели YOLO5 по датасету с лицами людей")

st.write('Время обучения: 60 минут')
st.write('Количество эпох: 5')
st.write('Обучающая выборка: 13400 изображений')
st.write('Порог уверенности для фильтрации изображений: 0.7')

st.title('Графики обучения')
st.image('images/face_results.png')

st.title('Матрица неточностей')
st.image('images/face_confusion_matrix.png')

st.title('Precision-Recall Curve')
st.image('images/face_PR_curve.png')

col1, col2 = st.columns(2)

with col1:
    st.image('images/face_P_curve.png', use_container_width=True)

with col2:
    st.image('images/face_R_curve.png', use_container_width=True)

st.title('F1-Confidence Curve')
st.image('images/face_F1_curve.png')


st.title("Детекция лиц с помощью YOLO5, с последующей маскировкой детектированной области")

# Радио-кнопки для выбора способа ввода
input_method = st.radio(
    "Выберите способ ввода изображения:",
    ("Загрузить файл", "Ввести ссылку")
)

uploaded_files = []

if input_method == "Загрузить файл":

    uploaded_files = st.file_uploader(
        "Загрузите изображения",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    for images in uploaded_files:
        st.image(images, caption="Загруженное изображение",
                 use_container_width=True)

elif input_method == "Ввести ссылку":

    # Поле для ввода URL
    url = st.text_input("Введите прямую ссылку на изображение:")

    uploaded_files = []

    if url:
        try:
            # Загружаем изображение из URL
            response = requests.get(url)
            response.raise_for_status()  # Проверка успешного запроса
            img = Image.open(BytesIO(response.content))

            img_data = BytesIO(response.content)

            # Добавляем изображение в список для обработки
            uploaded_files.append(
                {"data": img_data, "name": url.split("/")[-1]})

            # Отображаем изображение в Streamlit
            st.image(img_data, caption="Загруженное изображение",
                     use_container_width=True)

        except requests.exceptions.RequestException as e:
            st.error(f"Ошибка загрузки изображения: {e}")
        except Exception as e:
            st.error(f"Произошла ошибка при обработке изображения: {e}")


# Проверяем, загружены ли файлы
if uploaded_files:

    # Обработка каждого загруженного изображения
    for uploaded_file in uploaded_files:

        if isinstance(uploaded_file, dict):
            file_data = uploaded_file["data"]  # Данные из словаря
            file_name = uploaded_file["name"]
        else:
            file_data = uploaded_file  # UploadedFile напрямую
            file_name = uploaded_file.name

        # Открываем изображение с помощью Pillow
        image = Image.open(file_data)
        image_np = np.array(image)  # Преобразуем изображение в формат NumPy

        # Проверяем, является ли изображение цветным
        if len(image_np.shape) == 3:  # Цветное изображение
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_np

        # Детектируем лица на изображении
        results = model(image_bgr)
        # Формат: [x_min, y_min, x_max, y_max, confidence, class]
        detections = results.xyxy[0].cpu().numpy()

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
            cv2.rectangle(image_bgr, (x_min, y_min),
                          (x_max, y_max), color, thickness)

            # Добавляем текст с уверенностью
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_thickness = 2
            text_size = cv2.getTextSize(
                label, font, font_scale, text_thickness)[0]
            text_x, text_y = x_min, y_min - 10  # Координаты текста

            # Фон для текста
            cv2.rectangle(image_bgr, (text_x, text_y - text_size[1] - 5),
                          (text_x + text_size[0], text_y + 5), (0, 255, 0), -1)
            cv2.putText(image_bgr, label, (text_x, text_y), font,
                        font_scale, (0, 0, 0), text_thickness)

        # Конвертируем BGR обратно в RGB для отображения
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption=f"Результат обработки модели",
                 use_container_width=True)
