import streamlit as st

# Функция для генерации чертежа на основе введенных данных
def generate_drawing(data):
    # Здесь может быть ваша логика по генерации чертежа на основе данных
    # В этом примере просто выводим собранные данные
    return f"Чертеж для изделия с параметрами:\n{data}"

# Заголовок
st.title("Экспертная система для построения чертежа")

# Вопросы и поля для ввода ответов
description = st.text_area("Введите описание изделия:")
material = st.text_input("Материал изделия:")
dimensions = st.text_input("Габариты изделия:")
weight = st.text_input("Вес изделия:")

# Кнопка для генерации чертежа
if st.button("Сгенерировать чертеж"):
    # Собираем данные из введенных пользователем ответов
    data = {
        "Описание": description,
        "Материал": material,
        "Габариты": dimensions,
        "Вес": weight
    }

    # Проверяем, что все обязательные поля заполнены
    if description and material and dimensions and weight:
        drawing = generate_drawing(data)
        st.success(drawing)
    else:
        st.warning("Пожалуйста, заполните все поля.")
