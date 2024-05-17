# import csv
# import openpyxl
# from pymystem3 import Mystem
# from SVM_DIP import predict_detail_class  # Импортируем функцию predict_detail_class из модуля SVM_DIP

# def process_detail_text(text):
#     # Создаем новую книгу Excel
#     workbook = openpyxl.Workbook()
#     # Выбираем активный лист
#     sheet = workbook.active
    
#     mystem_analyzer = Mystem()
#     lemmatized_text = mystem_analyzer.lemmatize(text.lower())
#     cleaned_text = [token for token in lemmatized_text if token.isalpha()]

#     data_dict = {
#         "Двойной_ряд_шаровой_подшипник": {
#             "внутренний диаметр": "",
#             "внешний диаметр": "",
#             "обозначение модели": "",
#             "толщина": "",
#         },
#         "гайка": {
#             "обозначение резьба": "",
#             "ширина по плоскость": "",
#             "диаметр головка": "",
#             "толщина": "",
#             "диаметр отверстие": ""
#         },
#         "профиль": {
#             "тип": "",
#             "балки длина": "",
#             "балки радиус скругления": "",
#             "ширина полки": "",
#             "толщина полотна": "",
#             "высота балки": "",
#             "толщина фланца": ""
#         },
#         "радиальный подшипник": {
#             "обозначение модели": "",
#             "тип защиты подшипника": "",
#             "радиус скругления": "",
#             "толщина постфикс": "",
#             "внешний диаметр": "",
#             "внутренний диаметр": ""
#         },
#         "Батарея": {
#             "диаметр": "",
#             "тип код": "",
#             "высота": ""
#         }
#     }

#     # Предсказываем класс детали
#     predicted_class = predict_detail_class(text)
#     print("Предсказанный класс детали:", predicted_class)

#     # Определяем форму детали
#     data_dict[predicted_class]["форма"] = predicted_class

#     # Итерируемся по токенам и заполняем словарь
#     for i in range(len(cleaned_text)):
#         if cleaned_text[i] == "обозначение модели":
#             data_dict[predicted_class]["обозначение модели"] = cleaned_text[i+2]  # Получаем название детали
#         elif cleaned_text[i] == "внутренний" and cleaned_text[i+1] == "диаметр" and cleaned_text[i+2].isdigit():
#             data_dict[predicted_class]["внутренний диаметр"] = cleaned_text[i+2] + ' ' + cleaned_text[i+3]  # Получаем внутренний диаметр
#         elif cleaned_text[i] == "внешний" and cleaned_text[i+1] == "диаметр" and cleaned_text[i+2].isdigit():
#             data_dict[predicted_class]["внешний диаметр"] = cleaned_text[i+2] + ' ' + cleaned_text[i+3]  # Получаем внешний диаметр
#         elif cleaned_text[i] == "длина":
#             data_dict[predicted_class]["длина"] = cleaned_text[i+1] + ' ' + cleaned_text[i+2]  # Получаем длину детали
#         elif cleaned_text[i] == "ширина" and cleaned_text[i+1] == "по" and cleaned_text[i+2] == "плоскость":
#             data_dict[predicted_class]["ширина по плоскость"] = cleaned_text[i+3] + ' ' + cleaned_text[i+4]  # Получаем ширину по плоскости
#         elif cleaned_text[i] == "диаметр" and cleaned_text[i+1] == "головка" and cleaned_text[i+2].isdigit():
#             data_dict[predicted_class]["диаметр головка"] = cleaned_text[i+2] + ' ' + cleaned_text[i+3]  # Получаем диаметр головки
#         elif cleaned_text[i] == "толщина":
#             data_dict[predicted_class]["толщина"] = cleaned_text[i+1] + ' ' + cleaned_text[i+2]  # Получаем толщину
#         elif cleaned_text[i] == "диаметр" and cleaned_text[i+1] == "отверстие" and cleaned_text[i+2].isdigit():
#             data_dict[predicted_class]["диаметр отверстие"] = cleaned_text[i+2] + ' ' + cleaned_text[i+3]  # Получаем диаметр отверстия
#         elif cleaned_text[i] == "тип":
#             data_dict[predicted_class]["тип"] = cleaned_text[i+1]  # Получаем тип
#         elif cleaned_text[i] == "балки" and cleaned_text[i+1] == "длина" and cleaned_text[i+2].isdigit():
#             data_dict[predicted_class]["балки длина"] = cleaned_text[i+2] + ' ' + cleaned_text[i+3]  # Получаем длину балок
#         elif cleaned_text[i] == "балки" and cleaned_text[i+1] == "радиус" and cleaned_text[i+2] == "скругление":
#             data_dict[predicted_class]["балки радиус скругления"] = cleaned_text[i+3] + ' ' + cleaned_text[i+4]  # Получаем радиус скругления балок
#         elif cleaned_text[i] == "ширина" and cleaned_text[i+1] == "полки":
#             data_dict[predicted_class]["ширина полки"] = cleaned_text[i+2] + ' ' + cleaned_text[i+3]  # Получаем ширину полки
#         elif cleaned_text[i] == "толщина" and cleaned_text[i+1] == "полотно":
#             data_dict[predicted_class]["толщина полотна"] = cleaned_text[i+2] + ' ' + cleaned_text[i+3]  # Получаем толщину полотна
#         elif cleaned_text[i] == "высота" and cleaned_text[i+1] == "балки":
#             data_dict[predicted_class]["высота балки"] = cleaned_text[i+2] + ' ' + cleaned_text[i+3]  # Получаем высоту балки
#         elif cleaned_text[i] == "толщина" and cleaned_text[i+1] == "фланец":
#             data_dict[predicted_class]["толщина фланца"] = cleaned_text[i+2] + ' ' + cleaned_text[i+3]  # Получаем толщину фланца
#         elif cleaned_text[i] == "тип" and cleaned_text[i+1] == "защита" and cleaned_text[i+2] == "подшипник":
#             data_dict[predicted_class]["тип защиты подшипника"] = cleaned_text[i+3]  # Получаем тип защиты подшипника
#         elif cleaned_text[i] == "радиус" and cleaned_text[i+1] == "скругление" and cleaned_text[i+2].isdigit():
#             data_dict[predicted_class]["радиус скругления"] = cleaned_text[i+2] + ' ' + cleaned_text[i+3]  # Получаем радиус скругления
#         elif cleaned_text[i] == "толщина" and cleaned_text[i+1] == "постфикс":
#             data_dict[predicted_class]["толщина постфикс"] = cleaned_text[i+2] + ' ' + cleaned_text[i+3]  # Получаем толщину постфикса
#         elif cleaned_text[i] == "диаметр" and cleaned_text[i+1] == "батарея" and cleaned_text[i+2].isdigit():
#             data_dict[predicted_class]["диаметр"] = cleaned_text[i+2] + ' ' + cleaned_text[i+3]  # Получаем диаметр батареи
#         elif cleaned_text[i] == "тип" and cleaned_text[i+1] == "код":
#             data_dict[predicted_class]["тип код"] = cleaned_text[i+2]  # Получаем тип кода
#         elif cleaned_text[i] == "высота" and cleaned_text[i+1] == "батарея":
#             data_dict[predicted_class]["высота"] = cleaned_text[i+2]  # Получаем высоту


#     # Записываем данные в файл CSV
#     with open('technical_data.csv', mode='w', newline='', encoding='utf-8') as file:
#         writer = csv.DictWriter(file, fieldnames=data_dict[predicted_class].keys(), delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#         # Записываем заголовки столбцов
#         writer.writeheader()
#         # Записываем информацию о детали
#         writer.writerow(data_dict[predicted_class])

#     row = 1
#     for key, value in data_dict[predicted_class].items():
#         sheet.cell(row=row, column=1, value=key)
#         sheet.cell(row=row, column=2, value=value)
#         row += 1

#     workbook.save("technical_data.xlsx")
#     print("Данные записаны в файл 'technical_data.xlsx'.")

# text = "Деталь представляет из себя батарею с типом АА диаметром 45 мм и высотой 34 мм"
# process_detail_text(text)


import re

# Пример массива
array = ["45 мм", "45мм", "30мм", "20 мм", "10мм", "100 мм", "45", "мм"]


def extract_number(token):
    pattern = r"^\d+"  # Шаблон для поиска начала строки, за которым следует одно или более цифр
    match = re.match(pattern, token)
    if match:
        return int(match.group())  # Возвращаем числовое значение из токена
    else:
        return None  # Если токен не содержит числа, возвращаем None

# Пример использования:
tokens = ["45 мм", "45мм", "30мм", "20 мм", "10мм", "100 мм", "45", "мм"]
for token in tokens:
    number = extract_number(token)
    print(f"Вход: {token}, Выход: {number}")

