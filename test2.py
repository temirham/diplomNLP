import pandas as pd
from itertools import permutations

def generate_dataset(tokens, label):
    # Создаем генератор для всех возможных комбинаций токенов
    combinations_generator = (' '.join(token_order) for token_order in permutations(tokens))
    
    # Создаем DataFrame, используя генератор
    df = pd.DataFrame(combinations_generator, columns=['message'])
    
    # Добавляем столбец с меткой класса
    df['label'] = label
    
    return df

# Пример использования функции
tokens = ['диаметр отверстие', 'толщина', 'диаметр головка', 'ширина по плоскость', 'обозначение резьба', 'гайка']
label = 'гайка'
df = generate_dataset(tokens, label)

# Сохраняем датасет в файл CSV
df.to_csv('./datasetTemas/nut.csv', index=False)
