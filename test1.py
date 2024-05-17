import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Загрузка данных из файла CSV
df = pd.read_csv('./datasetTemas/random.csv')

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Преобразование текста в числовой формат с помощью TF-IDF векторизации
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Обучение модели SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)

# Предсказание на тестовом наборе данных
y_pred = svm_model.predict(X_test_tfidf)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)




# import pandas as pd
# import numpy as np
# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# # Загрузка данных из файла CSV
# df = pd.read_csv('./datasetTemas/random.csv')

# # Преобразование меток классов в числовой формат
# label_encoder = LabelEncoder()
# df['label'] = label_encoder.fit_transform(df['label'])

# # Разделение данных на обучающий и тестовый наборы
# X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# print(y_test)

# # Преобразование текста в числовой формат с помощью токенизатора
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(X_train)
# X_train_seq = tokenizer.texts_to_sequences(X_train)
# X_test_seq = tokenizer.texts_to_sequences(X_test)

# # Выравнивание последовательностей до одной длины
# max_length = max([len(seq) for seq in X_train_seq])
# X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
# X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

# # Создание нейронной сети
# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=64, input_length=max_length),
#     tf.keras.layers.GlobalAveragePooling1D(),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# # Компиляция модели
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Обучение модели
# model.fit(X_train_pad, y_train, epochs=10, batch_size=16, validation_data=(X_test_pad, y_test))

# # Оценка модели на тестовых данных
# loss, accuracy = model.evaluate(X_test_pad, y_test)
# print('Test Accuracy:', accuracy)




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Загрузка данных из файла CSV
df = pd.read_csv('./datasetTemas/random.csv')

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Преобразование текста в числовой формат с помощью CountVectorizer
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Обучение модели наивного байесовского классификатора
nb_model = MultinomialNB()
nb_model.fit(X_train_counts, y_train)

# Предсказание на тестовом наборе данных
y_pred = nb_model.predict(X_test_counts)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Загрузка данных из файла CSV
df = pd.read_csv('./datasetTemas/random.csv')

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Преобразование текста в числовой формат с помощью CountVectorizer
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)
# print(X_train_counts)

# Обучение модели дерева принятия решений
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_counts, y_train)

# Предсказание на тестовом наборе данных
y_pred = dt_model.predict(X_test_counts)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)





# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import accuracy_score

# # Загрузка данных из файла CSV
# df = pd.read_csv('dataset_with_random_words.csv')

# # Разделение данных на обучающий и тестовый наборы
# X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# # Преобразование текста в числовой формат с помощью TF-IDF векторизации
# vectorizer = TfidfVectorizer()
# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_test_tfidf = vectorizer.transform(X_test)

# # Обучение модели градиентного бустинга
# gb_model = GradientBoostingClassifier()
# gb_model.fit(X_train_tfidf, y_train)

# # Предсказание на тестовом наборе данных
# y_pred = gb_model.predict(X_test_tfidf)

# # Оценка точности модели
# accuracy = accuracy_score(y_test, y_pred)
# print('Accuracy:', accuracy)
