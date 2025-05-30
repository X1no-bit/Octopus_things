import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging
from typing import Tuple, Union, Optional
from pandas.errors import ParserError

# Импорты для нейронной сети
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Dota2DataPreprocessor:
    """
    Полноценный предобработчик данных для матчей Dota 2.
    Поддерживает как классические модели, так и временные ряды (LSTM).
    """
    
    def __init__(self, target_col: str = 'radiant_win', time_series: bool = False,
                 window_size: int = 10, csv_sep: str = ',', encoding: str = 'utf-8'):
        """Инициализация предобработчика"""
        self.target_col = target_col
        self.time_series = time_series
        self.window_size = window_size
        self.csv_sep = csv_sep
        self.encoding = encoding
        self.scaler = None
        self.encoder = None
        self.preprocessor = None
        self.feature_names = None
        self._validate_parameters()

    def _validate_parameters(self):
        """Проверка корректности параметров"""
        if not isinstance(self.target_col, str):
            raise ValueError("target_col должен быть строкой")
        if not isinstance(self.time_series, bool):
            raise ValueError("time_series должен быть булевым значением")
        if self.window_size <= 0:
            raise ValueError("window_size должен быть положительным числом")
        if not isinstance(self.csv_sep, str):
            raise ValueError("csv_sep должен быть строкой")
        if not isinstance(self.encoding, str):
            raise ValueError("encoding должен быть строкой")

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Загрузка данных из JSON/CSV с обработкой ошибок"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Файл {filepath} не найден")

            logger.info(f"Попытка загрузить файл: {filepath}")

            if filepath.endswith('.json'):
                self.raw_data = pd.read_json(filepath)
            elif filepath.endswith('.csv'):
                try:
                    self.raw_data = pd.read_csv(
                        filepath,
                        sep=self.csv_sep,
                        encoding=self.encoding,
                        engine='python',
                        on_bad_lines='warn'
                    )
                except ParserError as e:
                    logger.warning(f"Ошибка парсинга CSV: {e}. Пробуем другой подход.")
                    self.raw_data = pd.read_csv(filepath, sep=self.csv_sep, encoding=self.encoding,
                                              error_bad_lines=False)

            logger.info(f"Успешно загружено {len(self.raw_data)} записей")
            return self.raw_data

        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {e}")
            raise

    def preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Предобработка данных (реализуйте вашу логику предобработки)"""
        # Здесь должна быть ваша логика предобработки данных
        # Например:
        # - Обработка пропущенных значений
        # - Кодирование категориальных признаков
        # - Масштабирование числовых признаков
        # - Выбор фичей и т.д.
        
        # Пример простой предобработки:
        X = data.drop(columns=[self.target_col])
        y = data[self.target_col]
        
        # Здесь должен быть ваш код предобработки
        # ...
        
        return X, y

    def train_test_split(self, test_size: float = 0.2) -> tuple:
        """Разделение данных на train/test"""
        if not hasattr(self, 'raw_data'):
            raise ValueError("Данные не загружены. Сначала вызовите load_data()")

        X, y = self.preprocess_data(self.raw_data)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        return X_train, X_val, y_train, y_val

class Dota2ModelTrainer:
    """
    Класс для построения и обучения нейронной сети на данных Dota 2
    """
    
    def __init__(self, input_shape: tuple, time_series: bool = False):
        """
        Инициализация тренера модели
        
        Параметры:
        - input_shape: форма входных данных
        - time_series: использовать ли архитектуру для временных рядов
        """
        self.input_shape = input_shape
        self.time_series = time_series
        self.model = self._build_model()
    
    def _build_model(self) -> Sequential:
        """Построение архитектуры нейронной сети"""
        model = Sequential()
        
        if self.time_series:
            # Архитектура для временных рядов (LSTM)
            model.add(LSTM(128, input_shape=self.input_shape, return_sequences=True, 
                          kernel_regularizer=l2(0.01)))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            
            model.add(LSTM(64, kernel_regularizer=l2(0.01)))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
        else:
            # Стандартная архитектура для табличных данных
            model.add(Dense(256, input_dim=self.input_shape[0], activation='relu', 
                         kernel_regularizer=l2(0.01)))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            
            model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            
            model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
        
        # Выходной слой
        model.add(Dense(1, activation='sigmoid'))
        
        # Компиляция модели
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', 
                    metrics=['accuracy', 'AUC'])
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=100, batch_size=64):
        """
        Обучение модели с использованием ранней остановки
        
        Параметры:
        - X_train, y_train: тренировочные данные
        - X_val, y_val: валидационные данные
        - epochs: максимальное количество эпох
        - batch_size: размер батча
        """
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history

def main():
    try:
        # 1. Загрузка и предобработка данных
        preprocessor = Dota2DataPreprocessor(target_col='radiant_win', time_series=False)
        data = preprocessor.load_data('dota2_matches.csv')  # Укажите ваш файл с данными
        
        # 2. Разделение данных
        X_train, X_val, y_train, y_val = preprocessor.train_test_split(test_size=0.2)
        
        # 3. Подготовка данных для нейронной сети
        if preprocessor.time_series:
            # Для временных рядов преобразуем данные в 3D форму (samples, timesteps, features)
            X_train = X_train.reshape((X_train.shape[0], preprocessor.window_size, -1))
            X_val = X_val.reshape((X_val.shape[0], preprocessor.window_size, -1))
        
        # 4. Создание и обучение модели
        input_shape = X_train.shape[1:]  # Форма входных данных
        trainer = Dota2ModelTrainer(input_shape, time_series=preprocessor.time_series)
        
        print("Начало обучения модели...")
        history = trainer.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=64)
        
        # 5. Оценка модели
        print("\nОценка модели на валидационных данных:")
        val_loss, val_acc, val_auc = trainer.model.evaluate(X_val, y_val, verbose=0)
        print(f"Validation Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}")
        
        # 6. Сохранение модели (опционально)
        trainer.model.save('dota2_model.h5')
        print("Модель сохранена как 'dota2_model.h5'")
        
    except Exception as e:
        logger.error(f"Ошибка в основном потоке выполнения: {e}")
        raise

if __name__ == "__main__":
    main()