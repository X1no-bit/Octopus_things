import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


# 1. Загрузка данных
def load_data(filepath):
    """Загрузка сырых данных из CSV"""
    df = pd.read_csv(filepath)
    print(f"Данные загружены. Размер: {df.shape}")
    return df


# 2. Очистка данных
def clean_data(df):
    """Удаление ненужных колонок и обработка пропусков"""

    # Удаление нерелевантных колонок
    cols_to_drop = ['match_id', 'player_slot', 'game_time']  # пример
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Обработка пропусков
    df = df.dropna(subset=['player_gold'])  # ключевая фича

    # Заполнение остальных пропусков
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna('unknown')

    print(f"После очистки: {df.shape}")
    return df


# 3. Фичевая инженерия
def feature_engineering(df):
    """Создание новых признаков"""

    # Относительные показатели
    df['gold_per_min'] = df['player_gold'] / (df['game_duration'] / 60)
    df['xp_advantage'] = df['hero_xp'] - df['enemy_avg_xp']

    # Взаимодействия
    df['gold_xp_ratio'] = df['player_gold'] / (df['hero_xp'] + 1e-6)

    # Временные тренды (для временных рядов)
    df['gold_diff_1min'] = df['player_gold'].diff(60)  # изменение за 1 мин

    print(f"Добавлено {len(df.columns) - len(df.columns)} новых фич")
    return df


# 4. Нормализация
def normalize_data(df, target_col='player_gold'):
    """Масштабирование числовых признаков"""

    # Отделяем целевую переменную
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Выбираем только числовые колонки
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns

    # Стандартное масштабирование
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Для временных рядов можно MinMax (0-1)
    # scaler = MinMaxScaler()
    # X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    print(f"Масштабировано {len(numeric_cols)} признаков")
    return X, y, scaler


# 5. Создание временных окон (для LSTM)
def create_time_windows(X, y, window_size=10, steps_ahead=1):
    """Преобразование в 3D-формат для LSTM"""
    X_seq, y_seq = [], []

    for i in range(len(X) - window_size - steps_ahead):
        X_seq.append(X.iloc[i:i + window_size].values)
        y_seq.append(y.iloc[i + window_size + steps_ahead - 1])

    return np.array(X_seq), np.array(y_seq)


# 6. Разделение данных
def split_data(X, y, test_size=0.2, time_series=False):
    """Разделение на train/test с учетом временных рядов"""
    if time_series:
        split_idx = int(len(X) * (1 - test_size))
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ==================== ПРИМЕР ИСПОЛЬЗОВАНИЯ ====================

# Загрузка данных
raw_data = load_data('dota2_matches.csv')

# Очистка
cleaned_data = clean_data(raw_data)

# Фичевая инженерия
engineered_data = feature_engineering(cleaned_data)

# Нормализация
X, y, scaler = normalize_data(engineered_data)

# Для классических моделей (LR, XGBoost)
X_train, X_test, y_train, y_test = split_data(X, y, time_series=False)

# Для LSTM
X_seq, y_seq = create_time_windows(X, y, window_size=10, steps_ahead=5)
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = split_data(
    X_seq, y_seq, time_series=True)
Для первых экспериментов можно использовать готовые датасеты с Kaggle:

python
Copy
kaggle datasets download -d darinhawley/dota-2-matches
Для реального времени добавьте:

python
Copy
def preprocess_realtime(data, scaler):
    """Обработка одного нового сэмпла"""
    data = pd.DataFrame([data])
    data = feature_engineering(data)
    data = scaler.transform(data)
    return data.reshape(1, -1)  # для LR/XGBoost
    # или .reshape(1, window_size, -1) для LSTM
Для категориальных признаков (герои, предметы):

python
Copy
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
hero_encoded = encoder.fit_transform(df[['hero_id']])
Оптимизация памяти:

python
Copy
df = df.astype({'player_gold': 'float32', 'hero_xp': 'float32'})