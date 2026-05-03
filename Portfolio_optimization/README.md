# Quant Portfolio AI Pro

Система машинного обучения и квант-анализа для оптимизации инвестиционного портфеля на данных Московской Биржи (MOEX).

## Описание

Дипломная работа реализует профессиональную платформу для:
- Загрузки исторических данных с MOEX через официальный API
- Расчёта ключевых финансовых метрик (Sharpe, VaR, CVaR, Max Drawdown)
- Оптимизации портфеля шестью различными методами
- Визуализации результатов и генерации экспертных заключений

## Математические модели

1. **Max Sharpe Ratio (MPT Markowitz)** — классическая теория портфеля Марковица (1952)
2. **Risk Parity (Ray Dalio All Weather)** — паритет рисков, равный вклад каждого актива в общий риск
3. **Global Minimum Volatility** — портфель минимальной волатильности
4. **Maximum Diversification** — максимизация коэффициента диверсификации
5. **Minimum CVaR** — минимизация условного Value at Risk (хвостовые риски)
6. **Hierarchical Risk Parity (Lopez de Prado, 2016)** — иерархическая кластеризация активов

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

```bash
streamlit run app.py
```

## Структура проекта

```
Portfolio_optimization/
├── app.py                 # Главное приложение Streamlit
├── requirements.txt       # Зависимости
├── assets/
│   └── style.css        # Стили оформления
└── src/
    ├── __init__.py      # Инициализация пакета
    ├── data_loader.py   # Загрузка данных с MOEX
    ├── models.py        # Математические модели
    ├── visuals.py       # Визуализация (Plotly)
    └── expert_engine.py # Экспертный анализ
```

## Требования

- Python >= 3.9
- streamlit >= 1.30.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- plotly >= 5.18.0
- apimoex >= 1.3.0

## Автор

[Ваше ФИО]  
[Название ВУЗа]  
[Год]
