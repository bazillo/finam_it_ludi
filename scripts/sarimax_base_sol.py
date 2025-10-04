import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')



class MultiStepSARIMAXForecaster:
    def __init__(self):
        self.models = {}
        self.news_vectorizer = None
        self.news_lda = None
        
    def prepare_targets(self, candles_df, horizon=20):
        """Подготовка целевых переменных - 20 кумулятивных доходностей"""
        tickers = candles_df['ticker'].unique()
        all_data = []
        
        for ticker in tickers:
            ticker_data = candles_df[candles_df['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('begin')
            
            # Создаем целевые переменные r1, r2, ..., r20
            # где r_i = close_{t+i} / close_t - 1
            current_close = ticker_data['close']
            for i in range(1, horizon + 1):
                future_close = ticker_data['close'].shift(-i)
                ticker_data[f'p{i}'] = (future_close / current_close) - 1
            
            all_data.append(ticker_data)
            
        return pd.concat(all_data, ignore_index=True)
    
    def prepare_features(self, candles_df, news_df=None):
        """Подготовка признаков"""
        features_list = []
        
        for ticker in candles_df['ticker'].unique():
            ticker_data = candles_df[candles_df['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('begin')
            
            # Ценовые признаки
            ticker_data['returns'] = ticker_data['close'].pct_change()
            ticker_data['volatility'] = ticker_data['returns'].rolling(5).std()
            ticker_data['volume_change'] = ticker_data['volume'].pct_change()
            ticker_data['high_low_ratio'] = ticker_data['high'] / ticker_data['low']
            
            # Лаговые признаки
            for lag in [1, 2, 3, 5]:
                ticker_data[f'returns_lag_{lag}'] = ticker_data['returns'].shift(lag)
                ticker_data[f'volume_lag_{lag}'] = ticker_data['volume_change'].shift(lag)
            
            # Скользящие средние
            ticker_data['sma_5'] = ticker_data['close'].rolling(5).mean()
            ticker_data['sma_10'] = ticker_data['close'].rolling(10).mean()
            ticker_data['close_to_sma5'] = ticker_data['close'] / ticker_data['sma_5']
            
            features_list.append(ticker_data)
        
        features_df = pd.concat(features_list, ignore_index=True)
        
        # Новостные признаки (если есть)
        if news_df is not None and len(news_df) > 0:
            news_features = self._extract_news_features(news_df)
            # Здесь нужно добавить логику объединения с основными признаками
            # по дате/времени
            
        return features_df
    
    def _extract_news_features(self, news_df):
        """Извлечение признаков из новостей"""
        news_texts = news_df['title'] + " " + news_df['publication']
        
        if self.news_vectorizer is None:
            self.news_vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
            news_tfidf = self.news_vectorizer.fit_transform(news_texts)
            
            self.news_lda = LatentDirichletAllocation(n_components=5, random_state=42)
            topic_features = self.news_lda.fit_transform(news_tfidf)
        else:
            news_tfidf = self.news_vectorizer.transform(news_texts)
            topic_features = self.news_lda.transform(news_tfidf)
            
        return pd.DataFrame(topic_features, 
                          columns=[f'news_topic_{i}' for i in range(5)],
                          index=news_df['publish_date'])
    
    def train(self, candles_df, news_df=None, horizon=20):
        """Обучение отдельных моделей для каждого горизонта прогноза"""
        # Подготовка данных с целевыми переменными
        data_with_targets = self.prepare_targets(candles_df, horizon)
        features = self.prepare_features(candles_df, news_df)
        
        tickers = candles_df['ticker'].unique()
        
        for ticker in tqdm(tickers,leave=False):
            print(f"Training models for {ticker}")
            ticker_data = data_with_targets[data_with_targets['ticker'] == ticker].copy()
            ticker_data = ticker_data.dropna()
            
            if len(ticker_data) < 30:
                print(f"Not enough data for {ticker}")
                continue
            
            # Обучаем отдельную модель для каждого горизонта
            ticker_models = {}
            for i in tqdm(range(1, horizon + 1),leave=False):
                target_col = f'p{i}'
                
                if target_col not in ticker_data.columns:
                    continue
                    
                # Используем исторические доходности как признаки
                feature_cols = ['returns', 'volatility', 'volume_change', 
                              'returns_lag_1', 'returns_lag_2', 'returns_lag_3']
                
                available_features = [col for col in feature_cols if col in ticker_data.columns]
                
                if len(available_features) == 0:
                    # Если нет признаков, используем только целевую переменную
                    model = SARIMAX(ticker_data[target_col], 
                                  order=(1, 0, 1),
                                  seasonal_order=(0, 0, 0, 0),
                                  enforce_stationarity=False)
                else:
                    # SARIMAX с экзогенными переменными
                    exog_features = ticker_data[available_features].fillna(0)
                    model = SARIMAX(ticker_data[target_col],
                                  exog=exog_features,
                                  order=(1, 0, 1),
                                  seasonal_order=(0, 0, 0, 0),
                                  enforce_stationarity=False)
                
                try:
                    fitted_model = model.fit(disp=False)
                    ticker_models[i] = fitted_model
                    print(f"  Horizon {i}: AIC = {fitted_model.aic:.2f}")
                except Exception as e:
                    print(f"  Horizon {i}: Failed - {e}")
                    # Простая модель как fallback
                    ticker_models[i] = None
            
            self.models[ticker] = ticker_models
    
    def predict(self, candles_df, news_df=None, horizon=20):
        """Прогнозирование r1, r2, ..., r20 для каждого тикера"""
        predictions = {}
        
        for ticker in candles_df['ticker'].unique():
            if ticker not in self.models:
                # Fallback: нулевые прогнозы
                predictions[ticker] = {f'p{i}': 0.0 for i in range(1, horizon + 1)}
                continue
            
            # Последние данные для прогноза
            ticker_data = candles_df[candles_df['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('begin').tail(10)  # Берем последние 10 точек
            
            if len(ticker_data) == 0:
                continue
            
            # Подготовка признаков для последней точки
            last_point = ticker_data.iloc[-1:].copy()
            features = self.prepare_features(last_point, news_df)
            
            ticker_predictions = {}
            for i in range(1, horizon + 1):
                if i in self.models[ticker] and self.models[ticker][i] is not None:
                    try:
                        # Прогноз на конкретный горизонт
                        if hasattr(self.models[ticker][i], 'exog_names_in') and self.models[ticker][i].exog_names_in:
                            # С экзогенными переменными
                            exog_features = features[self.models[ticker][i].exog_names_in].fillna(0)
                            forecast = self.models[ticker][i].get_forecast(steps=1, exog=exog_features)
                        else:
                            # Без экзогенных переменных
                            forecast = self.models[ticker][i].get_forecast(steps=1)
                        
                        pred_value = forecast.predicted_mean.iloc[0]
                        ticker_predictions[f'p{i}'] = pred_value
                    except Exception as e:
                        print(f"Prediction failed for {ticker} horizon {i}: {e}")
                        ticker_predictions[f'p{i}'] = 0.0
                else:
                    ticker_predictions[f'p{i}'] = 0.0
            
            predictions[ticker] = ticker_predictions
        
        return predictions

    def save_predictions(self, predictions, output_file='predictions.csv'):
        """Сохранение прогнозов в CSV файл"""
        results = []
        for ticker, preds in predictions.items():
            row = {'ticker': ticker}
            row.update(preds)
            results.append(row)
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        return results_df

def main_pipeline():
    """Основной пайплайн"""
    # Загрузка данных
    candles = pd.concat([pd.read_csv('data/raw/participants/candles.csv', parse_dates=['begin']),pd.read_csv('data/raw/participants/candles_2.csv', parse_dates=['begin'])])
    news = pd.concat([pd.read_csv('data/raw/participants/news.csv', parse_dates=['publish_date']), pd.read_csv('data/raw/participants/news_2.csv', parse_dates=['publish_date'])])
    
    
    # Инициализация и обучение
    forecaster = MultiStepSARIMAXForecaster()
    forecaster.train(candles, news, horizon=20)
    

    predictions = forecaster.predict(candles, horizon=20)
    
    results_df = forecaster.save_predictions(predictions, output_file='data/predictions.csv')
    
    return forecaster, results_df


# Запуск пайплайна
if __name__ == "__main__":
    model, predictions_df = main_pipeline()
    print(f"\nFirst 5 predictions:")
    print(predictions_df.head())