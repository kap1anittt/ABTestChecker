import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.stats.proportion import proportions_ztest
from typing import Dict, Tuple, List

class ABTestValidator:
    def __init__(self):
        """Инициализация валидатора A/B тестов"""
        pass
        
    def load_data(self, file) -> pd.DataFrame:
        """
        Загружает данные из CSV файла.
        
        Args:
            file: Файл CSV для загрузки
            
        Returns:
            pd.DataFrame: Загруженные данные
        """
        df = pd.read_csv(file)
        required_columns = ['user_id', 'group', 'conversion', 'country']
        
        # Проверка наличия всех необходимых столбцов
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Отсутствует обязательный столбец: {col}")
        
        # Базовая валидация
        if len(df['group'].unique()) != 2:
            raise ValueError("Должно быть ровно 2 группы (A и B)")
            
        # Конвертация группы в категориальный тип
        df['group'] = df['group'].astype('category')
        
        # Конвертация conversion в бинарный тип (0 или 1)
        df['conversion'] = df['conversion'].astype(int)
        
        return df
    
    def check_srm(self, df: pd.DataFrame) -> Dict:
        """
        Выполняет SRM проверку (Sample Ratio Mismatch)
        
        Args:
            df: DataFrame с данными теста
            
        Returns:
            Dict: Результаты проверки
        """
        # Подсчет наблюдений в каждой группе
        group_counts = df['group'].value_counts()
        
        # Ожидаемые частоты (50/50)
        total = len(df)
        expected = np.array([total / 2, total / 2])
        observed = np.array(group_counts)
        
        # Выполнение chi-square теста
        chi2, p_value = stats.chisquare(observed, expected)
        
        # Различие в процентах
        actual_ratio = observed[0] / total * 100
        
        return {
            'group_counts': dict(group_counts),
            'chi2': chi2,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'actual_ratio': actual_ratio
        }
    
    def plot_group_distribution(self, df: pd.DataFrame):
        """
        Создает график распределения пользователей по группам
        
        Args:
            df: DataFrame с данными теста
        """
        group_counts = df['group'].value_counts().reset_index()
        group_counts.columns = ['group', 'count']
        
        fig = px.bar(
            group_counts, 
            x='group', 
            y='count',
            title='Распределение пользователей по группам',
            labels={'count': 'Количество пользователей', 'group': 'Группа'}
        )
        
        return fig
    
    def check_country_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Проверяет однородность распределения стран между группами
        
        Args:
            df: DataFrame с данными теста
            
        Returns:
            Dict: Результаты проверки
        """
        # Создание таблицы сопряженности (contingency table)
        contingency_table = pd.crosstab(df['group'], df['country'])
        
        # Выполнение chi-square теста
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        return {
            'contingency_table': contingency_table,
            'chi2': chi2,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'degrees_of_freedom': dof
        }
    
    def plot_country_distribution(self, df: pd.DataFrame):
        """
        Создает график распределения стран по группам
        
        Args:
            df: DataFrame с данными теста
        """
        # Подсчет количества пользователей по странам и группам
        country_group_counts = df.groupby(['group', 'country']).size().reset_index(name='count')
        
        fig = px.bar(
            country_group_counts,
            x='country',
            y='count',
            color='group',
            barmode='group',
            title='Распределение стран по группам',
            labels={'count': 'Количество пользователей', 'country': 'Страна', 'group': 'Группа'}
        )
        
        return fig
    
    def calculate_test_results(self, df: pd.DataFrame) -> Dict:
        """
        Рассчитывает результаты A/B теста
        
        Args:
            df: DataFrame с данными теста
            
        Returns:
            Dict: Результаты расчета
        """
        # Группировка по группе и расчет конверсии
        results = df.groupby('group')['conversion'].agg(['count', 'sum']).reset_index()
        results['conversion_rate'] = results['sum'] / results['count']
        
        # Определение групп A и B
        groups = results['group'].tolist()
        if 'A' in groups and 'B' in groups:
            group_a = results[results['group'] == 'A']
            group_b = results[results['group'] == 'B']
        else:
            # Если группы не названы A и B, берем первую и вторую
            group_a = results.iloc[0]
            group_b = results.iloc[1]
        
        # Расчет uplift
        conv_a = group_a['conversion_rate'].values[0]
        conv_b = group_b['conversion_rate'].values[0]
        uplift = (conv_b - conv_a) / conv_a * 100
        
        # Статистический тест
        count_a = group_a['count'].values[0]
        count_b = group_b['count'].values[0]
        
        success_a = group_a['sum'].values[0]
        success_b = group_b['sum'].values[0]
        
        counts = np.array([success_a, success_b])
        nobs = np.array([count_a, count_b])
        
        z_stat, p_value = proportions_ztest(counts, nobs)
        
        # 95% доверительный интервал для uplift
        # Упрощенный расчет ошибки разницы пропорций
        se_a = np.sqrt(conv_a * (1 - conv_a) / count_a)
        se_b = np.sqrt(conv_b * (1 - conv_b) / count_b)
        se_diff = np.sqrt(se_a**2 + se_b**2)
        
        # Расчет доверительного интервала для разницы
        diff = conv_b - conv_a
        ci_lower = diff - 1.96 * se_diff
        ci_upper = diff + 1.96 * se_diff
        
        # Приведение к процентам
        rel_ci_lower = ci_lower / conv_a * 100
        rel_ci_upper = ci_upper / conv_a * 100
        
        return {
            'conversion_a': conv_a,
            'conversion_b': conv_b,
            'uplift': uplift,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'abs_diff': diff,
            'abs_ci': (ci_lower, ci_upper),
            'rel_ci': (rel_ci_lower, rel_ci_upper)
        }
    
    def plot_conversion_results(self, df: pd.DataFrame):
        """
        Создает график конверсий по группам
        
        Args:
            df: DataFrame с данными теста
        """
        # Группировка по группе и расчет конверсии
        results = df.groupby('group')['conversion'].agg(['count', 'sum']).reset_index()
        results['conversion_rate'] = results['sum'] / results['count'] * 100  # в процентах
        
        fig = px.bar(
            results,
            x='group',
            y='conversion_rate',
            title='Конверсия по группам',
            labels={'conversion_rate': 'Конверсия, %', 'group': 'Группа'},
            text_auto='.2f'
        )
        
        fig.update_layout(yaxis_range=[0, max(results['conversion_rate']) * 1.2])
        
        return fig 