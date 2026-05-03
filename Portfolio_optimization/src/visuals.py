import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from scipy import stats

class Visualizer:
    TEMPLATE = "plotly_dark"
    BG_COLOR = "rgba(0,0,0,0)"

    @staticmethod
    def _get_y_zoom_buttons(price_series):
        """Создаёт кнопки быстрого зума по оси Y"""
        min_p = price_series.min()
        max_p = price_series.max()
        mid = (min_p + max_p) / 2
        full_range = max_p - min_p
        
        return [
            dict(
                label="+25%",
                method="relayout",
                args=["yaxis.range", [mid - full_range*0.375, mid + full_range*0.375]]
            ),
            dict(
                label="+50%", 
                method="relayout",
                args=["yaxis.range", [mid - full_range*0.75, mid + full_range*0.75]]
            ),
            dict(
                label="100%",
                method="relayout", 
                args=["yaxis.autorange", True]
            ),
            dict(
                label="Сброс всё",
                method="relayout",
                args=[{"xaxis.autorange": True, "yaxis.autorange": True}]
            )
        ]

    @staticmethod
    def plot_price_history(prices, y_title="Цена закрытия (RUB)"):
        """График с зумом по цене и времени"""
        fig = px.line(
            prices, x=prices.index, y=prices.columns,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(
            template=Visualizer.TEMPLATE,
            paper_bgcolor=Visualizer.BG_COLOR,
            plot_bgcolor=Visualizer.BG_COLOR,
            xaxis_title='Дата', yaxis_title=y_title,
            legend_title="Тикеры", hovermode="x unified",
            margin=dict(t=30, b=10, l=10, r=10)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', fixedrange=False)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', fixedrange=False)
        return fig

    @staticmethod
    def plot_individual_price(prices, ticker):
        """Индивидуальный график с кнопками зума по цене"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=prices.index, y=prices[ticker], mode='lines', name=ticker,
            line=dict(color='#00f2fe', width=2.5),
            fill='tozeroy', fillcolor='rgba(0, 242, 254, 0.1)',
            hovertemplate="<b>Дата:</b> %{x}<br><b>Цена:</b> %{y:.2f} ₽<extra></extra>"
        ))
        
        current = prices[ticker].iloc[-1]
        pmin, pmax = prices[ticker].min(), prices[ticker].max()
        pmean = prices[ticker].mean()
        
        fig.update_layout(
            template=Visualizer.TEMPLATE,
            paper_bgcolor=Visualizer.BG_COLOR,
            plot_bgcolor=Visualizer.BG_COLOR,
            title=dict(
                text=f"{ticker}: <b>{current:.2f} ₽</b> "
                     f"<span style='font-size:13px;color:#64748b'>[{pmin:.1f}–{pmax:.1f} ₽]</span>",
                font=dict(size=17, color="#f8fafc")
            ),
            xaxis_title="", yaxis_title="Цена (RUB)",
            hovermode="x unified", margin=dict(t=60, b=10, l=10, r=10),
            
            # 🔑 Кнопки зума по цене (updatemenus)
            updatemenus=[dict(
                type="buttons",
                direction="left",
                x=0, y=1.15, xanchor="left", yanchor="top",
                pad={"r": 10, "t": 10},
                showactive=True,
                buttons=Visualizer._get_y_zoom_buttons(prices[ticker])
            )]
        )
        
        # Ось X: время с ползунком
        fig.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1М", step="month", stepmode="backward"),
                    dict(count=6, label="6М", step="month", stepmode="backward"),
                    dict(count=1, label="1Г", step="year", stepmode="backward"),
                    dict(step="all", label="ВСЕ")
                ]),
                bgcolor="#1e293b", activecolor="#3b82f6", font=dict(size=10)
            ),
            rangeslider=dict(visible=True, thickness=0.08, bgcolor="rgba(30,41,59,0.3)"),
            type="date", showgrid=True, gridcolor='rgba(255,255,255,0.05)', fixedrange=False
        )
        
        # 🔑 Ось Y: разрешаем зум + добавляем среднюю линию
        fig.update_yaxes(
            showgrid=True, gridcolor='rgba(255,255,255,0.05)',
            fixedrange=False, autorange=True
        )
        fig.add_hline(
            y=pmean, line_dash="dot", line_color="#94a3b8", opacity=0.3,
            annotation_text=f"Средняя: {pmean:.1f} ₽",
            annotation_position="top right", annotation_font_size=9
        )
        
        # Подсказка
        fig.add_annotation(
            text="Зум: колёсико | Выделение | Кнопки выше",
            x=0.5, y=-0.15, xref="paper", yref="paper",
            showarrow=False, font=dict(size=10, color="#64748b"),
            bgcolor="rgba(30,41,59,0.8)", bordercolor="#334155",
            borderwidth=1, borderpad=4, align="center"
        )
        
        return fig

    # === Остальные методы без изменений ===
    @staticmethod
    def plot_allocation(weights, tickers):
        df = pd.DataFrame({'Тикер': tickers, 'Вес': weights}).sort_values('Вес', ascending=False)
        df = df[df['Вес'] > 0.001]
        fig = px.pie(df, values='Вес', names='Тикер', hole=0.6, color_discrete_sequence=px.colors.sequential.Tealgrn_r)
        fig.update_traces(textposition='outside', textinfo='percent+label', marker=dict(line=dict(color='#0f172a', width=2)))
        fig.update_layout(template=Visualizer.TEMPLATE, paper_bgcolor=Visualizer.BG_COLOR, plot_bgcolor=Visualizer.BG_COLOR, showlegend=False, annotations=[dict(text='Аллокация', x=0.5, y=0.5, font_size=20, showarrow=False, font_color="#94a3b8")])
        return fig

    @staticmethod
    def plot_distribution_and_qq(daily_returns):
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=daily_returns, histnorm='probability density', name='Эмпирическое', opacity=0.75, marker_color='#0ea5e9'))
        x_norm = np.linspace(daily_returns.min(), daily_returns.max(), 100)
        y_norm = stats.norm.pdf(x_norm, daily_returns.mean(), daily_returns.std())
        fig.add_trace(go.Scatter(x=x_norm, y=y_norm, mode='lines', name='Нормальное (Гаусс)', line=dict(color='#f43f5e', width=3, dash='dot')))
        fig.update_layout(template=Visualizer.TEMPLATE, paper_bgcolor=Visualizer.BG_COLOR, plot_bgcolor=Visualizer.BG_COLOR, xaxis_title='Дневная доходность', yaxis_title='Плотность вероятности', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        return fig

    @staticmethod
    def plot_rolling_metrics(daily_returns, window=60):
        rolling_vol = daily_returns.rolling(window).std() * np.sqrt(252) * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol, mode='lines', line=dict(color='#10b981', width=2), fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.1)'))
        fig.update_layout(title=f'Скользящая историческая волатильность ({window} дней)', template=Visualizer.TEMPLATE, paper_bgcolor=Visualizer.BG_COLOR, plot_bgcolor=Visualizer.BG_COLOR, yaxis_title='Волатильность (%)', xaxis_title='Дата')
        return fig

    @staticmethod
    def plot_correlation_heatmap(returns):
        corr = returns.corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="Temps", zmin=-1, zmax=1)
        fig.update_layout(template=Visualizer.TEMPLATE, paper_bgcolor=Visualizer.BG_COLOR, plot_bgcolor=Visualizer.BG_COLOR, margin=dict(t=30, b=10, l=10, r=10))
        return fig