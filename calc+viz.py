import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.tools import mpl_to_plotly
import plotly.io as pio
import mpld3
import base64
import panel as pn
pn.extension('echarts')
from common import *
from indicators import airi_score, final_rsa, final_psa, final_incap, final_insaf, invcap, invsaf


## Indicator history charts
def create_his_graph(y_values, filename):
    # X-axis
    today = datetime.today()
    if today.weekday() == 0:  # Monday is 0
        end_date = today
    else:
        end_date = today - timedelta(days=today.weekday())

    dates = [end_date - timedelta(weeks=i) for i in range(len(y_values))]
    date_labels = [date.strftime('%b %d') for date in reversed(dates)]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=date_labels,
        y=y_values,
        mode='lines',
        line=dict(color='blue')))


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
        paper_bgcolor='rgba(0,0,0,0)',  # Set background color to transparent
        hovermode="x unified",
        hoverlabel = dict(
            bgcolor="white",  # Background color for hover labels
            font_size=16,  # Font size for hover labels
            font_color="black")  # Font color for hover labels
    )

    fig.show()
    return fig.write_html(f"{filename}.html", config= {'displayModeBar': False})


create_his_graph(invcap, "invcap")
create_his_graph(invsaf, "invsaf")

sa_rsa = [40.12, 42.15, 43.67, 44.88, 46.20, 47.35, 48.58, 49.12, 49.87, 50.24, 50.67, 51.22, 50.89]
sa_psa = [62.34, 63.56, 65.12, 66.78, 67.89, 68.45, 68.90, 69.12, 69.45, 69.78, 70.12, 70.50, 70.77]
hist_airi = [0.25 * (sa_rsa[i] + sa_psa[i] + invcap[i] + invsaf[i]) for i in range(len(sa_rsa))]

create_his_graph(sa_rsa, "sa_rsa")
create_his_graph(sa_psa, "sa_psa")
create_his_graph(hist_airi, "hist_airi")




## Gauge chart for index home page
def deg_to_rad(deg):
    return deg * np.pi/180

values = [100, 80, 60, 40, 20, 0]
x_axis_deg = [0, 36, 72, 108, 144]
x_axis_vals = [deg_to_rad(i) for i in x_axis_deg]

fig = plt.figure(figsize=(18, 18), dpi=100)
ax = fig.add_subplot(projection="polar")
fig.patch.set_alpha(0)
ax.set_facecolor('none')

# Highlighted bar
highlighted_edgecolor = "#6b6b6b"
default_edgecolor = "#D3D3DA"
highlight_index = next(i for i, v in enumerate(values) if v > round(airi_score) > (v - 20))

# Bars with padding
num_bars = 5
bar_width = deg_to_rad(180/num_bars) * 0.95  # Adjust width to leave padding
gap = deg_to_rad(180/num_bars) * 0.05  # Small gap between bars

for i, (x, height) in enumerate(zip(x_axis_vals, values)):
    edgecolor = highlighted_edgecolor if i == highlight_index else default_edgecolor
    ax.bar(x=x + gap / 2, width=bar_width, height=1, bottom=1.5, linewidth=5, edgecolor=edgecolor, color="#FFFCF8", align="edge")

# Annotations
annotations = [("High\nRisk", 18, 2.0, -75),
               ("Rising\nRisk", 54, 2.0, -40),
               ("Balanced\nEfforts", 90, 2.0, 0),
               ("Slowed\nAI Dev", 126, 2.0, 40),
               ("Blocked\nAI Dev", 162, 2.0, 75)]

for i, (text, x, y, rot) in enumerate(annotations):
    color = "black" if i == highlight_index else highlighted_edgecolor
    plt.annotate(text, xy=(deg_to_rad(x), y), rotation=rot, fontweight="bold", fontsize=16, color=color,
                 rotation_mode='anchor', transform=ax.transData, ha="center")

# Little tick values
ticks = [100, "•", 80, "•", 60, "•", 40, "•", 20, "•", 0]
for i, (loc, val) in enumerate(zip([deg_to_rad(i) for i in [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]], ticks)):
    color = highlighted_edgecolor
    ha = "right" if i <= 4 else ("center" if i == 5 else "left")
    plt.annotate(val, xy=(loc, 1.3), ha=ha, fontsize="16", color=color)

# Needle
plt.annotate(round(airi_score), xytext=(0, 0), xy=(deg_to_rad((100-round(airi_score))/100*180), 1.8), fontsize=45, color="#FFFCF8", ha="center", va="center",
             arrowprops=dict(arrowstyle="wedge, tail_width=0.5", color="black"),
             bbox=dict(boxstyle="circle, pad=0.4", facecolor="black", linewidth=0.3))

ax.set_axis_off()
plt.savefig('gauge.png', transparent=True)

with open('gauge.png', 'rb') as img_file:
    base64_encoded_image = base64.b64encode(img_file.read()).decode('utf-8')

# HTML content
html_content = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Embedded PNG</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background-color: transparent;
        }}
        .container {{
            width: 100%;
            height: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: transparent;
        }}
        img {{
            width: 120%; /* Increase this value to make the image larger */
            height: 120%; /* Increase this value to make the image larger */
            object-fit: contain;
            max-width: none;
            max-height: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <img src="data:image/png;base64,{base64_encoded_image}" alt="Embedded PNG"/>
    </div>
</body>
</html>
'''

# Save HTML content to a file
with open('gauge.html', 'w') as html_file:
    html_file.write(html_content)

"""
import matplotlib.pyplot as plt
import base64

values = [100, 80, 60, 40, 20, 0]
x_axis_vals = [0, 0.63, 1.26, 1.89, 2.52]

fig = plt.figure(figsize=(18, 18), dpi=100)
ax = fig.add_subplot(projection="polar")

# Set background color of figure and axes to be transparent
fig.patch.set_alpha(0)  # Transparent background for figure
ax.set_facecolor('none')  # Transparent background for axes

ax.bar(x=x_axis_vals, width=0.62, height=1, bottom=1.5, linewidth=5, edgecolor="#D3D3DA", color="#FFFCF8", align="edge")

annotations = [("UNRULY", 0.19, 1.82, -75),
               ("MISALIGNED", 1.02, 1.5, -40),
               ("BALANCED", 1.76, 1.99, 0),
               ("SLOW", 2.32, 2.04, 40),
               ("STAGNANT", 3.03, 2.08, 75)]

for text, x, y, rot in annotations:
    plt.annotate(text, xy=(x, y), rotation=rot, fontweight="bold", fontsize=20)

for loc, val in zip([0, 0.63, 1.26, 1.89, 2.52, 3.15], values):
    plt.annotate(val, xy=(loc, 2.52), ha="right" if val<=40 else "left", fontsize="14")

plt.annotate(round(airi_score), xytext=(0, 0), xy=(1.1, 1.7), fontsize=45, color="#FFFCF8", ha="center",
             arrowprops=dict(arrowstyle="wedge, tail_width=0.5", color="black"),
             bbox=dict(boxstyle="circle", facecolor="black", linewidth=1.5))

ax.set_axis_off()
plt.savefig('gauge.png', transparent=True)

with open('gauge.png', 'rb') as img_file:
    base64_encoded_image = base64.b64encode(img_file.read()).decode('utf-8')

# Create HTML content
html_content = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Embedded PNG</title>
</head>
<body>
    <img src="data:image/png;base64,{base64_encoded_image}" alt="Embedded PNG"/>
</body>
</html>
'''

# Save HTML content to a file
with open('gauge.html', 'w') as html_file:
    html_file.write(html_content)

#####

import matplotlib.pyplot as plt
import base64

values = [100, 80, 60, 40, 20, 0]
x_axis_vals = [0, 0.63, 1.26, 1.89, 2.52]

# Assuming airi_score is defined somewhere earlier in your code
airi_score = 50  # Replace with the actual score

fig = plt.figure(figsize=(18, 18), dpi=100)
ax = fig.add_subplot(projection="polar")

# Set background color of figure and axes to be transparent
fig.patch.set_alpha(0)  # Transparent background for figure
ax.set_facecolor('none')  # Transparent background for axes

# Define a different edge color for the highlighted bar
highlighted_edgecolor = "#8c8c8c"
default_edgecolor = "#D3D3DA"

# Determine the index of the bar that should be highlighted
highlight_index = next(i for i, v in enumerate(values) if v >= round(airi_score))

# Plot bars with different edge colors
for i, (x, height) in enumerate(zip(x_axis_vals, values)):
    edgecolor = highlighted_edgecolor if i == highlight_index else default_edgecolor
    ax.bar(x=x, width=0.62, height=1, bottom=1.5, linewidth=5, edgecolor=edgecolor, color="#FFFCF8", align="edge")

annotations = [("UNRULY", 0.19, 1.82, -75),
               ("MISALIGNED", 1.02, 1.5, -40),
               ("BALANCED", 1.76, 1.99, 0),
               ("SLOW", 2.32, 2.04, 40),
               ("STAGNANT", 3.03, 2.08, 75)]

# Plot annotations with different colors based on the highlighted bar
for i, (text, x, y, rot) in enumerate(annotations):
    color = "black" if i == highlight_index else highlighted_edgecolor
    plt.annotate(text, xy=(x, y), rotation=rot, fontweight="bold", fontsize=20, color=color)

# Plot values on the bars
for i, (loc, val) in enumerate(zip([0, 0.63, 1.26, 1.89, 2.52, 3.15], values)):
    color = "black" if i == highlight_index else highlighted_edgecolor
    plt.annotate(val, xy=(loc, 2.52), ha="right" if val <= 40 else "left", fontsize="14", color=color)

plt.annotate(round(airi_score), xytext=(0, 0), xy=(1.1, 1.7), fontsize=45, color="#FFFCF8", ha="center",
             arrowprops=dict(arrowstyle="wedge, tail_width=0.5", color="black"),
             bbox=dict(boxstyle="circle", facecolor="black", linewidth=1.5))

ax.set_axis_off()
plt.savefig('gauge1.png', transparent=True)

with open('gauge1.png', 'rb') as img_file:
    base64_encoded_image = base64.b64encode(img_file.read()).decode('utf-8')

# Create HTML content
html_content = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Embedded PNG</title>
</head>
<body>
    <img src="data:image/png;base64,{base64_encoded_image}" alt="Embedded PNG"/>
</body>
</html>
'''

# Save HTML content to a file
with open('gauge1.html', 'w') as html_file:
    html_file.write(html_content)
"""
