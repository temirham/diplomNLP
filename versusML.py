# import pandas as pd
# import time
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score

# # Загрузка данных из файла CSV
# df = pd.read_csv('./datasetTemas/random.csv')

# # Разделение данных на обучающий и тестовый наборы
# X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# # Преобразование текста в числовой формат с помощью TF-IDF векторизации
# vectorizer = TfidfVectorizer()
# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_test_tfidf = vectorizer.transform(X_test)

# # Обучение модели SVM
# start_time = time.time()
# svm_model = SVC(kernel='linear')
# svm_model.fit(X_train_tfidf, y_train)
# end_time = time.time()
# svm_time = end_time - start_time

# # Предсказание на тестовом наборе данных
# y_pred_svm = svm_model.predict(X_test_tfidf)

# # Оценка точности модели SVM
# svm_accuracy = accuracy_score(y_test, y_pred_svm)

# # Обучение модели наивного байесовского классификатора
# start_time = time.time()
# nb_model = MultinomialNB()
# nb_model.fit(X_train_tfidf, y_train)
# end_time = time.time()
# nb_time = end_time - start_time

# # Предсказание на тестовом наборе данных
# y_pred_nb = nb_model.predict(X_test_tfidf)

# # Оценка точности модели наивного байесовского классификатора
# nb_accuracy = accuracy_score(y_test, y_pred_nb)

# # Обучение модели дерева принятия решений
# start_time = time.time()
# dt_model = DecisionTreeClassifier()
# dt_model.fit(X_train_tfidf, y_train)
# end_time = time.time()
# dt_time = end_time - start_time

# # Предсказание на тестовом наборе данных
# y_pred_dt = dt_model.predict(X_test_tfidf)

# # Оценка точности модели дерева принятия решений
# dt_accuracy = accuracy_score(y_test, y_pred_dt)

# # Функция для построения графиков
# def plot_results(algorithms, accuracies, times):
#     plt.figure(figsize=(12, 5))

#     # График точности предсказания
#     plt.subplot(1, 2, 1)
#     plt.bar(algorithms, accuracies, color='skyblue')
#     plt.xlabel('Algorithms')
#     plt.ylabel('Accuracy')
#     plt.title('Accuracy of Text Classification Algorithms')

#     # График времени обработки
#     plt.subplot(1, 2, 2)
#     plt.bar(algorithms, times, color='lightgreen')
#     plt.xlabel('Algorithms')
#     plt.ylabel('Processing Time (s)')
#     plt.title('Processing Time of Text Classification Algorithms')

#     plt.tight_layout()
#     plt.show()

# # Данные для графиков
# algorithms = ['SVM + TF-IDF', 'Naive Bayes', 'Decision Tree']
# accuracies = [svm_accuracy, nb_accuracy, dt_accuracy]
# times = [svm_time, nb_time, dt_time]

# # Создание графиков
# plot_results(algorithms, accuracies, times)
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from pymystem3 import Mystem

text = 'Эта деталь представляет собой профиль с типом C, длиной балки 80 мм и радиусом скругления балки 5 мм. Ширина полки профиля составляет 40 мм, толщина полотна 3 мм, а высота балки 60 мм. Толщина фланца равна 5 мм.'

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

# Загрузка данных из файла CSV
df = pd.read_csv('./datasetTemas/random.csv')

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Преобразование текста в числовой формат с помощью TF-IDF векторизации
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Обучение модели SVM
start_time = time.time()
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)
end_time = time.time()
svm_time = end_time - start_time

# Предсказание на тестовом наборе данных для модели SVM
start_time = time.time()
y_pred_svm = svm_model.predict(X_test_tfidf)
end_time = time.time()
svm_prediction_time = end_time - start_time

# Оценка точности модели SVM
svm_accuracy = accuracy_score(y_test, y_pred_svm)

# Обучение модели наивного байесовского классификатора
start_time = time.time()
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
end_time = time.time()
nb_time = end_time - start_time

# Предсказание на тестовом наборе данных для модели наивного байесовского классификатора
start_time = time.time()
y_pred_nb = nb_model.predict(X_test_tfidf)
end_time = time.time()
nb_prediction_time = end_time - start_time

# Оценка точности модели наивного байесовского классификатора
nb_accuracy = accuracy_score(y_test, y_pred_nb)

# Обучение модели дерева принятия решений
start_time = time.time()
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_tfidf, y_train)
end_time = time.time()
dt_time = end_time - start_time

# Предсказание на тестовом наборе данных для модели дерева принятия решений
start_time = time.time()
y_pred_dt = dt_model.predict(X_test_tfidf)
end_time = time.time()
dt_prediction_time = end_time - start_time

# Оценка точности модели дерева принятия решений
dt_accuracy = accuracy_score(y_test, y_pred_dt)

# Функция для построения графиков
def plot_results(algorithms, accuracies, training_times, prediction_times):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # График точности предсказания
    axes[0].bar(algorithms, accuracies, color='skyblue')
    axes[0].set_xlabel('Algorithms')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy of Text Classification Algorithms')

    # График времени обучения
    axes[1].bar(algorithms, training_times, color='lightgreen')
    axes[1].set_xlabel('Algorithms')
    axes[1].set_ylabel('Training Time (s)')
    axes[1].set_title('Training Time of Text Classification Algorithms')

    # График времени предсказания
    axes[2].bar(algorithms, prediction_times, color='lightcoral')
    axes[2].set_xlabel('Algorithms')
    axes[2].set_ylabel('Prediction Time (s)')
    axes[2].set_title('Prediction Time of Text Classification Algorithms')

    plt.tight_layout()
    plt.show()




# Данные для графиков
algorithms = ['SVM + TF-IDF', 'Naive Bayes', 'Decision Tree']
accuracies = [svm_accuracy, nb_accuracy, dt_accuracy]
training_times = [svm_time, nb_time, dt_time]
prediction_times = [svm_prediction_time, nb_prediction_time, dt_prediction_time]

# Создание графиков
plot_results(algorithms, accuracies, training_times, prediction_times)

