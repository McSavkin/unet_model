import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import torchvision.transforms as T
from torchvision.io import read_image

st.title('Детекция ветряных мельниц')
st.divider()

st.subheader('Информация о модели')
st.write('Модель: Yolov5n')
st.write('Число эпох: 15')
st.write('Время обучения: 36 мин.')
st.write('mAP@0.5 = 0.819')
st.write('mAP@0.5:0.95 = 0.402')
st.subheader('Процесс обучения')
st.image('images/results_one.png')
st.subheader('Карта предсказаний')
st.image('images/confusion_matrix_one.png')
st.subheader('Precision-Recall curve')
st.image('images/PR_curve_one.png')
st.subheader('F1 curve')
st.image('images/F1_curve_one.png')
st.divider()

st.subheader('Детекция')
load_option = st.radio(
    'Выберите способ загрузки изображения', ('По ссылке', 'Файлом'))

model2 = torch.hub.load('ultralytics/yolov5', 'custom',
                        path='models/best_one_class.pt')  # Замените на ваш путь


def load_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        st.error(f"Не удалось загрузить изображение: {e}")
        return None


# def detect_show(image, model, proba=0.5):
#     model.conf = proba
#     img = T.ToPILImage()(read_image(image))
#     model.eval()
#     with torch.inference_mode():
#         results = model(img)
# # results.show()  # or .show(), .save(), .crop(), .pandas(), render(), etc
#     results.show()

def detect_show(image, model, proba=0.5):
    model.conf = proba
    img_tensor = T.ToTensor()(image).unsqueeze(
        0)  # Добавляем размерность для батча
    img_pil = image  # Это уже PIL.Image
    # Детекция
    model.eval()
    with torch.no_grad():
        results = model(img_pil)
    # Возвращаем результат в виде изображение с аннотациями
    # Получаем изображение с боксовыми аннотациями
    img_with_boxes = results.render()[0]
    img_with_boxes_pil = Image.fromarray(
        img_with_boxes)  # Конвертируем в PIL.Image
    return img_with_boxes_pil


if load_option == 'По ссылке':
    url = st.text_input("Введите ссылку")
    if url:
        img = load_image_from_url(url)
        col1, col2 = st.columns(2)
        proba = st.slider('Выберите вероятность', 0.0,
                          1.0, step=0.1, value=0.6)
        with col1:
            st.image(img, caption="Исходное изображение",
                     use_column_width=True)
        with col2:
            res_img = detect_show(img, model2, proba)
            st.image(res_img, caption="Детекция", use_column_width=True)
else:
    uploaded_files = st.file_uploader(
        'Загурзите изображение', accept_multiple_files=True)
    if uploaded_files:
        for i, uploaded_files in enumerate(uploaded_files):
            img = Image.open(uploaded_files)
            col1, col2 = st.columns(2)
            proba = st.slider(
                f'Выберите вероятность для изображения {i + 1}',
                0.0, 1.0, step=0.1, value=0.6,
                key=f'slider_{i}'
            )
            with col1:
                st.image(
                    img, caption=f'Изображение {i + 1}', use_column_width=True)
            with col2:
                result_img = detect_show(img, model2, proba)
                st.image(
                    result_img, caption=f'Изображение {i + 1} с детекцией', use_column_width=True)
