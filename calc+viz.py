import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import panel as pn
pn.extension('echarts')
from common import *
from indicators import airi_score, final_rsa, final_psa, final_incap, final_insaf, invcap, invsaf


# Indicator history charts
def create_ind_his_graph(y_final, y_values, filename):
    # X-axis
    today = datetime.today()
    if today.weekday() == 0:  # Monday is 0
        end_date = today
    else:
        end_date = today - timedelta(days=today.weekday())

    dates = [end_date - timedelta(weeks=i) for i in range(len(y_values))]
    date_labels = [date.strftime('%b %d') for date in reversed(dates)]

    fig = go.Figure(#go.Indicator(
        # mode="number+delta",
        # value=y_final
        # number={"valueformat": ".2f", "suffix": "%"},
        # delta={"reference": y_values[-2], "valueformat": ".2f", "suffix": "%"},
        # domain={'y': [0.7, 1], 'x': [0.7, 0.1]}  # Position at top left corner
    )#)
    # TODO fig = px.line(title="layout.hovermode='x unified'")

    fig.add_trace(go.Scatter(
        x=date_labels,
        y=y_values,
        mode='lines',
        line=dict(color='blue')
    ))

    fig.update_layout(
        yaxis=dict(
            side='right',
            gridcolor='lightgrey',
            autorange=True,
            range=[min(y_values) - (max(y_values) - min(y_values)) * 0.1, max(y_values)]),
        xaxis=dict(
            range=[0, len(y_values)],
            gridcolor='lightgrey',
            showline=True,
            linecolor='black',
            linewidth=2,
            mirror=True),
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)'  # Set background color to transparent
    )

    fig.show()
    return fig.write_html(f"visualizations/{filename}.html")


create_ind_his_graph(final_incap, invcap, "invcap")
create_ind_his_graph(final_insaf, invsaf, "invsaf")



## Gauge chart for index home page
gauge = pn.indicators.Gauge(name='AIRI', value=airi_score, bounds=(0, 100), format='{value}%',
                    colors=[(0.2, 'green'), (0.8, 'gold'), (1, 'red')], annulus_width=20,
                    start_angle=180, end_angle=0)

gauge.save('visualizations/gauge_chart.html')

# ALTERNATIVE
dial = pn.indicators.Dial(
    name='AIRI', value=airi_score, bounds=(0, 100), format='{value}%',
    colors=[(0.2, 'green'), (0.8, 'gold'), (1, 'red')], annulus_width=10
)
dial = pn.indicators.Dial(
    name='AIRI', value=airi_score, bounds=(0, 100), format='{value}%',
    colors=[(0.2, 'green'), (0.8, 'gold'), (1, 'red')]
)
dial.save('visualizations/gauge_chart_alt.html')


#### ALT bu güzel olan:
def create_gauge(filename):
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=airi_score,
        number={'font': {'size': 50}, "valueformat": ".2f"},
        gauge={'axis': {'range': [0, 100],
                        #'dtick' : 1,
                        'tickvals': [0, 20, 40, 60, 80, 100],
                        'ticktext': [0, 20, 40, 60, 80, 100]},
               'steps': [{'range': [0, 20], 'color': "#FFFCF8", 'line': {'color': 'rgba(255, 0, 0, 0.4)', 'width': 4}},  # Red with 40% opacity
                         {'range': [20, 40], 'color': "#FFFCF8", 'line': {'color': 'rgba(255, 255, 0, 0.4)', 'width': 4}},  # Yellow with 40% opacity
                         {'range': [40, 60], 'color': "#FFFCF8", 'line': {'color': 'rgba(0, 255, 0, 0.4)', 'width': 4}},  # Green with 40% opacity
                         {'range': [60, 80], 'color': "#FFFCF8", 'line': {'color': 'rgba(255, 255, 0, 0.4)', 'width': 4}},  # Yellow with 40% opacity
                         {'range': [80, 100], 'color': "#FFFCF8", 'line': {'color': 'rgba(255, 0, 0, 0.4)', 'width': 4}}],   # Red with 40% opacity
                'bar': {'color': 'rgba(0,0,0,0)'}}))

    # Needle calculation
    angle = airi_score / 100 * 180
    needle_length = 0.4
    needle_base = 0.05

    needle_x = [0, np.cos(np.radians(180 - angle)) * needle_length, -needle_base / 2,
                0, needle_base / 2, np.cos(np.radians(180 - angle)) * needle_length]

    needle_y = [0, np.sin(np.radians(180 - angle)) * needle_length, 0.02,
                0, 0.02, np.sin(np.radians(180 - angle)) * needle_length]

    # Add the needle
    fig.add_trace(go.Scatter(x=needle_x, y=needle_y,
                             mode='lines',
                             fill='toself',
                             fillcolor='black',
                             line=dict(color='black', width=2),
                             showlegend=False))

    # Add a circle to mask the root of the needle
    fig.add_trace(go.Scatter(x=[0], y=[0],mode='markers',
                             marker=dict(size=90, color='white'),
                             showlegend=False))

    # Text labels
    for percentage, label in zip([6, 50, 94], ['Safety', 'Balance', 'Misalignment']):
        angle = percentage / 100 * 180
        x = 0.50 * np.cos(np.radians(180 - angle))
        y = 0.50 * np.sin(np.radians(180 - angle))
        text_angle = angle - 90

        # Add annotation for tangent text
        fig.add_annotation(x=x, y=y, text=label, showarrow=False,
            font=dict(size=16), xref="x", yref="y",
            textangle=text_angle, ax=0, ay=0)

    fig.update_traces(gauge_axis_tick0=0, gauge_axis_tickmode='auto',  gauge_axis_nticks=50, gauge_axis_ticklabelstep=2, gauge_axis_tickcolor='#E6E6E6', gauge_bordercolor='#E6E6E6', selector = dict(type='indicator'))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        width=600,
        xaxis=dict(showticklabels=False, range=[-0.6, 0.6]),
        yaxis=dict(showticklabels=False, range=[-0.1, 0.6]))

    fig.show()
    return fig.write_html(f"visualizations/{filename}.html")


create_gauge("gauge_chart")






# alttaki değil, üstteki iyi olan.

import numpy as np
import plotly.graph_objects as go

def create_fear_greed_gauge(airi_score, filename):
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=airi_score,
        number={'font': {'size': 40}},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickvals': [0, 20, 40, 60, 80, 100],
                'ticktext': [0, 20, 40, 60, 80, 100],
            },
            'steps': [
                {'range': [0, 20], 'color': 'rgba(255, 0, 0, 0.4)'},
                {'range': [20, 40], 'color': 'rgba(255, 255, 0, 0.4)'},
                {'range': [40, 60], 'color': 'rgba(0, 255, 0, 0.4)'},
                {'range': [60, 80], 'color': 'rgba(255, 255, 0, 0.4)'},
                {'range': [80, 100], 'color': 'rgba(255, 0, 0, 0.4)'}
            ],
            'bar': {'color': 'rgba(0,0,0,0)'},
            'bordercolor': 'black',
            'borderwidth': 2
        }
    ))

    # Needle calculation
    angle = airi_score / 100 * 180
    needle_length = 0.4

    needle_x = [
        0,
        np.cos(np.radians(180 - angle)) * needle_length,
    ]

    needle_y = [
        0,
        np.sin(np.radians(180 - angle)) * needle_length,
    ]

    # Add the needle
    fig.add_trace(go.Scatter(
        x=needle_x,
        y=needle_y,
        mode='lines',
        line=dict(color='black', width=3),
        showlegend=False
    ))

    # Text labels
    for percentage, label in zip([17, 50, 83], ['Fear', 'Neutral', 'Greed']):
        angle = percentage / 100 * 180
        x = 0.50 * np.cos(np.radians(180 - angle))
        y = 0.50 * np.sin(np.radians(180 - angle))
        text_angle = angle - 90

        # Add annotation for tangent text
        fig.add_annotation(
            x=x,
            y=y,
            text=label,
            showarrow=False,
            font=dict(size=16),
            xref="x",
            yref="y",
            textangle=text_angle,
            ax=0,
            ay=0
        )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        width=600,
        xaxis=dict(showticklabels=False, range=[-0.6, 0.6]),
        yaxis=dict(showticklabels=False, range=[-0.1, 0.6])
    )

    fig.show()
    return fig.write_html(f"visualizations/{filename}.html")

# Example usage
create_fear_greed_gauge(55, "fear_greed_gauge")

