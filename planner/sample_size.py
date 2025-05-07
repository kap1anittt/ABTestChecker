import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower
from scipy.optimize import brentq
import plotly.graph_objects as go
from typing import Dict, Tuple, Optional

class ABTestPlanner:
    def __init__(self):
        """Инициализация планировщика A/B тестов"""
        self.power_analysis = TTestIndPower()
        
    def calculate_sample_size(
        self, 
        baseline_rate: float, 
        mde: float, 
        alpha: float = 0.05, 
        power: float = 0.8
    ) -> Dict[str, float]:
        """
        Расчет необходимого размера выборки для A/B теста.
        
        Args:
            baseline_rate (float): Базовая конверсия (например, 0.1 для 10%)
            mde (float): Минимально детектируемый эффект (относительный прирост)
            alpha (float): Уровень значимости (по умолчанию 0.05)
            power (float): Мощность теста (по умолчанию 0.8)
            
        Returns:
            Dict[str, float]: Словарь с результатами расчетов
        """
        # Расчет абсолютного эффекта
        absolute_effect = baseline_rate * mde
        
        # Расчет стандартного отклонения для биномиального распределения
        std_dev = np.sqrt(baseline_rate * (1 - baseline_rate))
        
        # Расчет размера эффекта (Cohen's d)
        effect_size = absolute_effect / std_dev
        
        # Расчет размера выборки на группу
        sample_size = self.power_analysis.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            ratio=1.0,
            alternative='two-sided'
        )
        
        return {
            'sample_size_per_group': int(np.ceil(sample_size)),
            'total_sample_size': int(np.ceil(sample_size * 2)),
            'effect_size': effect_size,
            'absolute_effect': absolute_effect
        }

    def calculate_mde(
        self, 
        baseline_rate: float, 
        sample_size: int, 
        alpha: float = 0.05, 
        power: float = 0.8
    ) -> Dict[str, float]:
        """
        Расчет минимально детектируемого эффекта (MDE) при заданном размере выборки.
        
        Args:
            baseline_rate (float): Базовая конверсия
            sample_size (int): Размер выборки на группу
            alpha (float): Уровень значимости
            power (float): Мощность теста
            
        Returns:
            Dict[str, float]: Словарь с результатами расчетов
        """
        # Стандартное отклонение для биномиального распределения
        std_dev = np.sqrt(baseline_rate * (1 - baseline_rate))
        
        # Расчет эффекта через решение уравнения мощности
        effect_size = self.power_analysis.solve_power(
            nobs1=sample_size,
            alpha=alpha,
            power=power,
            ratio=1.0,
            alternative='two-sided'
        )
        
        # Расчет абсолютного и относительного MDE
        absolute_mde = effect_size * std_dev
        relative_mde = absolute_mde / baseline_rate
        
        return {
            'effect_size': effect_size,
            'absolute_mde': absolute_mde,
            'relative_mde': relative_mde
        }

    def plot_power_curves(
        self,
        baseline_rate: float,
        mde: float,
        alpha: float = 0.05,
        max_sample_size: int = 10000
    ) -> None:
        """
        Построение графиков зависимости мощности от размера выборки.
        
        Args:
            baseline_rate (float): Базовая конверсия
            mde (float): Минимально детектируемый эффект
            alpha (float): Уровень значимости
            max_sample_size (int): Максимальный размер выборки для графика
        """
        # Создаем диапазон размеров выборки
        sample_sizes = np.linspace(100, max_sample_size, 50)
        
        # Расчет стандартного отклонения
        std_dev = np.sqrt(baseline_rate * (1 - baseline_rate))
        effect_size = (baseline_rate * mde) / std_dev
        
        # Расчет мощности для каждого размера выборки
        powers = [
            self.power_analysis.power(
                effect_size=effect_size,
                nobs1=n,
                alpha=alpha,
                ratio=1.0
            )
            for n in sample_sizes
        ]
        
        # Создание графика
        fig = go.Figure()
        
        # Добавление линии мощности
        fig.add_trace(go.Scatter(
            x=sample_sizes,
            y=powers,
            mode='lines',
            name='Power'
        ))
        
        # Добавление горизонтальной линии для power=0.8
        fig.add_hline(y=0.8, line_dash="dash", line_color="red",
                     annotation_text="Power = 0.8")
        
        # Настройка макета
        fig.update_layout(
            title='Power vs Sample Size',
            xaxis_title='Sample Size per Group',
            yaxis_title='Power',
            yaxis_range=[0, 1],
            showlegend=True
        )
        
        fig.show()

    def plot_mde_curve(
        self,
        baseline_rate: float,
        sample_size_range: Tuple[int, int],
        alpha: float = 0.05,
        power: float = 0.8
    ) -> None:
        """
        Построение графика зависимости MDE от размера выборки.
        
        Args:
            baseline_rate (float): Базовая конверсия
            sample_size_range (Tuple[int, int]): Диапазон размеров выборки (min, max)
            alpha (float): Уровень значимости
            power (float): Мощность теста
        """
        # Создаем диапазон размеров выборки
        sample_sizes = np.linspace(sample_size_range[0], sample_size_range[1], 50)
        
        # Расчет MDE для каждого размера выборки
        mdes = [
            self.calculate_mde(
                baseline_rate=baseline_rate,
                sample_size=int(n),
                alpha=alpha,
                power=power
            )['relative_mde']
            for n in sample_sizes
        ]
        
        # Создание графика
        fig = go.Figure()
        
        # Добавление линии MDE
        fig.add_trace(go.Scatter(
            x=sample_sizes,
            y=mdes,
            mode='lines',
            name='MDE'
        ))
        
        # Настройка макета
        fig.update_layout(
            title='Minimum Detectable Effect vs Sample Size',
            xaxis_title='Sample Size per Group',
            yaxis_title='Relative MDE',
            showlegend=True
        )
        
        fig.show()

