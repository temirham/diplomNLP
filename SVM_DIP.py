import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from pymystem3 import Mystem

def predict_detail_class(text):
    # Загрузка данных из файла CSV
    df = pd.read_csv('datasetTemas\dataset2.csv')
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

    # Предсказание метки класса для введенного текста
    predicted_label = svm_model.predict(test_text_vectorized)

    return predicted_label[0]

# Пример использования функции
# input_text = input("Введите текст для тестирования: ")
# predicted_class = predict_detail_class(input_text)
# print("Предсказанный класс детали:", predicted_class)
