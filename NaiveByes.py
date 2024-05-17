import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from pymystem3 import Mystem

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

    # Предсказание метки класса для введенного текста
    predicted_label = nb_model.predict(test_text_vectorized)

    return predicted_label[0]

# Пример использования функции
# input_text = input("Введите текст для тестирования: ")
# predicted_class = predict_detail_class_NB(input_text)
# print("Предсказанный класс детали:", predicted_class)
