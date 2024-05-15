import nltk
# nltk.download('punkt')
# nltk.download('wordnet')

def main():
    """
    Главная функция экспертной системы.
    """

    # База знаний - словарь, где ключи - симптомы, а значения - возможные причины
    knowledge_base = {
        "не включается": ["Неисправен блок питания", "Повреждена материнская плата"],
        "выключается во время работы": ["Перегрев процессора", "Неисправен блок питания"],
        "медленно работает": ["Недостаточно оперативной памяти", "Заражение вирусом"],
        "искажается изображение на экране": ["Неисправна видеокарта", "Поврежден монитор"],
    }

    # Правила логического вывода - список функций, каждая из которых 
    # принимает список возможных причин и возвращает обновленный список
    inference_rules = [
        lambda causes: ["Неисправен блок питания"] if "не включается" in causes else causes,
        lambda causes: ["Перегрев процессора"] if "выключается во время работы" in causes and "Неисправен блок питания" not in causes else causes,
        # ... другие правила логического вывода
    ]

    # Функция для токенизации и лемматизации введенного текста
    def preprocess_text(text):
        tokens = nltk.word_tokenize(text.lower())
        lemmas = [nltk.WordNetLemmatizer().lemmatize(token) for token in tokens]
        return lemmas

    # Получение симптомов от пользователя с помощью NLTK
    symptoms_text = input("Опишите проблему своими словами: ")
    symptoms_lemmas = preprocess_text(symptoms_text)

    # Поиск подходящего ключа в базе знаний
    possible_causes = []
    for symptom, causes in knowledge_base.items():
        if symptom.lower() in symptoms_text.lower():
            possible_causes.extend(causes)

    # Применение правил логического вывода
    for rule in inference_rules:
        possible_causes = rule(possible_causes)

    # Вывод результата
    if possible_causes:
        print(f"Возможные причины проблемы: {', '.join(possible_causes)}")
    else:
        print("К сожалению, по описанным симптомам невозможно определить причину проблемы.")

if __name__ == "__main__":
    main()
