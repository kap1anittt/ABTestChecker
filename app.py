import streamlit as st
from planner.sample_size import ABTestPlanner
from planner.test_validator import ABTestValidator
import pandas as pd
import os

def main():
    st.set_page_config(page_title="A/B Test Validator", layout="wide")
    
    st.title("A/B Test Validator")
    st.write("Инструмент для планирования и валидации A/B тестов")
    
    # Инициализация
    planner = ABTestPlanner()
    validator = ABTestValidator()
    
    # Создаем боковую панель для ввода параметров
    with st.sidebar:
        st.header("Параметры теста")
        
        baseline_rate = st.number_input(
            "Базовая конверсия (например, 0.1 для 10%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01
        )
        
        mde = st.number_input(
            "Минимально детектируемый эффект (например, 0.05 для 5%)",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01
        )
        
        alpha = st.number_input(
            "Уровень значимости (alpha)",
            min_value=0.01,
            max_value=0.1,
            value=0.05,
            step=0.01
        )
        
        power = st.number_input(
            "Мощность теста (power)",
            min_value=0.7,
            max_value=0.99,
            value=0.8,
            step=0.05
        )
        
        # Раздел загрузки файла
        st.header("Загрузка данных")
        uploaded_file = st.file_uploader("Загрузите CSV файл с данными теста", type=["csv"])
    
    # Создаем вкладки для разных функций
    tab1, tab2, tab3 = st.tabs(["Планирование", "Валидация", "Результаты"])
    
    # Вкладка планирования теста
    with tab1:
        st.header("Планирование A/B теста")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Расчет размера выборки")
            if st.button("Рассчитать размер выборки"):
                results = planner.calculate_sample_size(
                    baseline_rate=baseline_rate,
                    mde=mde,
                    alpha=alpha,
                    power=power
                )
                
                st.write("Результаты расчета:")
                st.write(f"Размер выборки на группу: {results['sample_size_per_group']}")
                st.write(f"Общий размер выборки: {results['total_sample_size']}")
                st.write(f"Размер эффекта (Cohen's d): {results['effect_size']:.4f}")
                st.write(f"Абсолютный эффект: {results['absolute_effect']:.4f}")
                
                # Построение графика мощности
                planner.plot_power_curves(
                    baseline_rate=baseline_rate,
                    mde=mde,
                    alpha=alpha,
                    max_sample_size=results['sample_size_per_group'] * 2
                )
        
        with col2:
            st.subheader("Расчет MDE")
            sample_size = st.number_input(
                "Размер выборки на группу",
                min_value=100,
                max_value=1000000,
                value=1000,
                step=100
            )
            
            if st.button("Рассчитать MDE"):
                results = planner.calculate_mde(
                    baseline_rate=baseline_rate,
                    sample_size=sample_size,
                    alpha=alpha,
                    power=power
                )
                
                st.write("Результаты расчета:")
                st.write(f"Размер эффекта (Cohen's d): {results['effect_size']:.4f}")
                st.write(f"Абсолютный MDE: {results['absolute_mde']:.4f}")
                st.write(f"Относительный MDE: {results['relative_mde']:.4f}")
                
                # Построение графика MDE
                planner.plot_mde_curve(
                    baseline_rate=baseline_rate,
                    sample_size_range=(sample_size//2, sample_size*2),
                    alpha=alpha,
                    power=power
                )
    
    # Вкладка валидации теста
    with tab2:
        st.header("Валидация групп теста")
        
        if uploaded_file is not None:
            try:
                df = validator.load_data(uploaded_file)
                
                # Отображение основной информации о данных
                st.subheader("Информация о данных")
                st.write(f"Всего записей: {len(df)}")
                st.write(f"Уникальных пользователей: {df['user_id'].nunique()}")
                st.write(f"Группы в тесте: {', '.join(df['group'].unique())}")
                st.write(f"Страны в данных: {', '.join(df['country'].unique())}")
                
                col1, col2 = st.columns(2)
                
                # SRM проверка
                with col1:
                    st.subheader("SRM проверка")
                    srm_results = validator.check_srm(df)
                    
                    # Отображение результатов
                    st.write(f"Распределение групп: {srm_results['group_counts']}")
                    st.write(f"Chi-square статистика: {srm_results['chi2']:.4f}")
                    st.write(f"p-value: {srm_results['p_value']:.4f}")
                    
                    if srm_results['is_significant']:
                        st.error("⚠️ Обнаружен Sample Ratio Mismatch! Распределение групп статистически значимо отличается от 50/50.")
                    else:
                        st.success("✅ SRM проверка пройдена. Распределение групп не отличается статистически значимо от 50/50.")
                    
                    # График распределения групп
                    st.plotly_chart(validator.plot_group_distribution(df), use_container_width=True)
                
                # Проверка по странам
                with col2:
                    st.subheader("Проверка однородности стран")
                    country_results = validator.check_country_distribution(df)
                    
                    # Отображение результатов
                    st.write("Таблица распределения стран по группам:")
                    st.dataframe(country_results['contingency_table'])
                    
                    st.write(f"Chi-square статистика: {country_results['chi2']:.4f}")
                    st.write(f"p-value: {country_results['p_value']:.4f}")
                    
                    if country_results['is_significant']:
                        st.error("⚠️ Распределение стран между группами неравномерно!")
                    else:
                        st.success("✅ Распределение стран однородно между группами.")
                    
                    # График распределения стран
                    st.plotly_chart(validator.plot_country_distribution(df), use_container_width=True)
            
            except Exception as e:
                st.error(f"Ошибка при обработке файла: {e}")
        else:
            st.info("Загрузите файл CSV с данными теста для валидации групп.")
    
    # Вкладка результатов теста
    with tab3:
        st.header("Результаты A/B теста")
        
        if uploaded_file is not None:
            try:
                df = validator.load_data(uploaded_file)
                
                # Расчет результатов
                results = validator.calculate_test_results(df)
                
                # Отображение результатов
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Статистические результаты")
                    st.write(f"Конверсия в группе A: {results['conversion_a']*100:.2f}%")
                    st.write(f"Конверсия в группе B: {results['conversion_b']*100:.2f}%")
                    st.write(f"Абсолютная разница: {results['abs_diff']*100:.2f}%")
                    st.write(f"Относительное изменение (uplift): {results['uplift']:.2f}%")
                    
                    # Доверительный интервал
                    st.write(f"95% доверительный интервал для абсолютной разницы: "\
                            f"[{results['abs_ci'][0]*100:.2f}%, {results['abs_ci'][1]*100:.2f}%]")
                    
                    st.write(f"95% доверительный интервал для uplift: "\
                            f"[{results['rel_ci'][0]:.2f}%, {results['rel_ci'][1]:.2f}%]")
                    
                    # Значимость
                    st.write(f"p-value: {results['p_value']:.4f}")
                    
                    if results['is_significant']:
                        st.success("✅ Результат статистически значим (p < 0.05)")
                    else:
                        st.error("⚠️ Результат статистически не значим (p ≥ 0.05)")
                
                with col2:
                    # График конверсий
                    st.plotly_chart(validator.plot_conversion_results(df), use_container_width=True)
            
            except Exception as e:
                st.error(f"Ошибка при обработке результатов: {e}")
        else:
            st.info("Загрузите файл CSV с данными теста для расчета результатов.")
    
    # Создаем пример файла данных
    if st.sidebar.button("Сгенерировать пример данных"):
        example_data = generate_example_data()
        st.sidebar.download_button(
            label="Скачать пример CSV",
            data=example_data,
            file_name="example_abtest_data.csv",
            mime="text/csv"
        )

def generate_example_data():
    """Генерирует пример данных A/B теста"""
    import numpy as np
    import pandas as pd
    
    # Параметры генерации
    n_users = 1000
    countries = ['RU', 'US', 'DE', 'UK', 'FR']
    conv_rate_a = 0.1  # 10% конверсия в группе A
    conv_rate_b = 0.11  # 11% конверсия в группе B (10% uplift)
    
    # Создаем DataFrame
    data = []
    
    for i in range(n_users):
        user_id = f"user_{i}"
        # Распределение 50/50
        group = 'A' if i < n_users / 2 else 'B'
        # Генерация страны с разным распределением
        p_country = [0.4, 0.3, 0.15, 0.1, 0.05]
        country = np.random.choice(countries, p=p_country)
        # Генерация конверсии
        conversion_rate = conv_rate_a if group == 'A' else conv_rate_b
        # Небольшой сдвиг по странам для реализма
        if country == 'US':
            conversion_rate += 0.02
        elif country == 'DE':
            conversion_rate -= 0.01
        
        conversion = 1 if np.random.random() < conversion_rate else 0
        
        data.append({
            'user_id': user_id,
            'group': group,
            'conversion': conversion,
            'country': country
        })
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False)

if __name__ == "__main__":
    main() 