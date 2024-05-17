import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from pymystem3 import Mystem
from sklearn.feature_extraction.text import CountVectorizer

# Функция для предсказания класса с использованием модели SVM
def predict_detail_class_SVM(text):
    # Загрузка данных из файла CSV
    df = pd.read_csv('./datasetTemas/random.csv')

    # Преобразование текста в числовой формат с помощью TF-IDF векторизации
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['message'])

    # Обучение модели SVM
    svm_model = SVC(kernel='linear')
    svm_model.fit(X, df['label'])

    # Предобработка и векторизация введенного текста
    test_text = text.lower()
    mystem_analyzer = Mystem()
    test_text = mystem_analyzer.lemmatize(test_text)
    test_text = ''.join(test_text)
    test_text_vectorized = vectorizer.transform([test_text])

    # Замер времени предсказания класса
    start_time = time.time()
    # Предсказание метки класса для введенного текста
    predicted_label = svm_model.predict(test_text_vectorized)
    end_time = time.time()
    prediction_time = end_time - start_time

    return predicted_label[0], prediction_time

# Функция для предсказания класса с использованием модели наивного байесовского классификатора
def predict_detail_class_NB(text):
    # Загрузка данных из файла CSV
    df = pd.read_csv('./datasetTemas/random.csv')

    # Преобразование текста в числовой формат с помощью CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['message'])

    # Обучение модели наивного байесовского классификатора
    nb_model = MultinomialNB()
    nb_model.fit(X, df['label'])

    # Предобработка и векторизация введенного текста
    test_text = text.lower()
    mystem_analyzer = Mystem()
    test_text = mystem_analyzer.lemmatize(test_text)
    test_text = ''.join(test_text)
    test_text_vectorized = vectorizer.transform([test_text])

    # Замер времени предсказания класса
    start_time = time.time()
    # Предсказание метки класса для введенного текста
    predicted_label = nb_model.predict(test_text_vectorized)
    end_time = time.time()
    prediction_time = end_time - start_time

    return predicted_label[0], prediction_time

# Функция для предсказания класса с использованием модели дерева принятия решений
def predict_detail_class_DT(text):
    # Загрузка данных из файла CSV
    df = pd.read_csv('./datasetTemas/random.csv')

    # Преобразование текста в числовой формат с помощью CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['message'])

    # Обучение модели дерева принятия решений
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X, df['label'])

    # Предобработка и векторизация введенного текста
    test_text = text.lower()
    mystem_analyzer = Mystem()
    test_text = mystem_analyzer.lemmatize(test_text)
    test_text = ''.join(test_text)
    test_text_vectorized = vectorizer.transform([test_text])

    # Замер времени предсказания класса
    start_time = time.time()
    # Предсказание метки класса для введенного текста
    predicted_label = dt_model.predict(test_text_vectorized)
    end_time = time.time()
    prediction_time = end_time - start_time

    return predicted_label[0], prediction_time

# # Пример использования функций
# input_text = input("Введите текст для тестирования: ")
# predicted_class_svm, time_svm = predict_detail_class_SVM(input_text)
# predicted_class_nb, time_nb = predict_detail_class_NB(input_text)
# predicted_class_dt, time_dt = predict_detail_class_DT(input_text)
# print("Предсказанный класс детали (SVM):", predicted_class_svm)
# print("Время предсказания (SVM):", time_svm)
# print("Предсказанный класс детали (Naive Bayes):", predicted_class_nb)
# print("Время предсказания (Naive Bayes):", time_nb)
# print("Предсказанный класс детали (Decision Tree):", predicted_class_dt)
# print("Время предсказания (Decision Tree):", time_dt)

# # Графики времени предсказания
# algorithms = ['SVM + TF-IDF', 'Naive Bayes', 'Decision Tree']
# times = [time_svm, time_nb, time_dt]

# plt.figure(figsize=(8, 5))
# plt.bar(algorithms, times, color=['skyblue', 'lightgreen', 'lightcoral'])
# plt.xlabel('Algorithms')
# plt.ylabel('Prediction Time (s)')
# plt.title('Prediction Time of Text Classification Algorithms')
# plt.show()
