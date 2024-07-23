import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import base64
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
            font_color="black"))  # Font color for hover labels

    fig.show()
    return fig.write_html(f"{filename}.html", config= {'displayModeBar': False})


sa_rsa = [40.12, 42.15, 43.67, 44.88, 46.20, 47.35, 48.58, 49.12, 49.87, 50.24, 50.67, 51.22, 50.89]
sa_psa = [62.34, 63.56, 65.12, 66.78, 67.89, 68.45, 68.90, 69.12, 69.45, 69.78, 70.12, 70.50, 70.77]
hist_airi = [0.25 * (sa_rsa[i] + sa_psa[i] + invcap[i] + invsaf[i]) for i in range(len(sa_rsa))]

create_his_graph(invcap, "invcap")
create_his_graph(invsaf, "invsaf")
create_his_graph(sa_rsa, "sa_rsa")
create_his_graph(sa_psa, "sa_psa")
create_his_graph(hist_airi, "hist_airi")



## Gauge chart for index home page
def deg_to_rad(deg):
    return deg * np.pi/180


values = [100, 80, 60, 40, 20, 0]
x_axis_deg = [0, 36, 72, 108, 144]
x_axis_vals = [deg_to_rad(i) for i in x_axis_deg]

fig = plt.figure(figsize=(10, 10), dpi=100)
ax = fig.add_subplot(projection="polar")
fig.patch.set_alpha(0)
ax.set_facecolor('none')

highlighted_edgecolor = "#6b6b6b"
default_edgecolor = "#D3D3DA"
highlight_index = next(i for i, v in enumerate(values) if v > round(airi_score) > (v - 20))

num_bars = 5
bar_width = deg_to_rad(180 / num_bars) * 0.95
gap = deg_to_rad(180 / num_bars) * 0.05

for i, (x, height) in enumerate(zip(x_axis_vals, values)):
    edgecolor = highlighted_edgecolor if i == highlight_index else default_edgecolor
    ax.bar(x=x + gap / 2, width=bar_width, height=1, bottom=1.5, linewidth=5, edgecolor=edgecolor, color="#FFFCF8",
           align="edge")

annotations = [("High\nRisk", 18, 2.0, -75),
               ("Rising\nRisk", 54, 2.0, -40),
               ("Balanced\nEfforts", 90, 2.0, 0),
               ("Slowed\nAI Dev", 126, 2.0, 40),
               ("Blocked\nAI Dev", 162, 2.0, 75)]

for i, (text, x, y, rot) in enumerate(annotations):
    color = "black" if i == highlight_index else highlighted_edgecolor
    plt.annotate(text, xy=(deg_to_rad(x), y), rotation=rot, fontweight="bold", fontsize=16, color=color,
                 rotation_mode='anchor', transform=ax.transData, ha="center")

ticks = [100, "•", 80, "•", 60, "•", 40, "•", 20, "•", 0]
for i, (loc, val) in enumerate(
        zip([deg_to_rad(i) for i in [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]], ticks)):
    color = highlighted_edgecolor
    ha = "right" if i <= 4 else ("center" if i == 5 else "left")
    plt.annotate(val, xy=(loc, 1.3), ha=ha, fontsize="16", color=color)

plt.annotate(round(airi_score), xytext=(0, 0), xy=(deg_to_rad((100 - round(airi_score)) / 100 * 180), 1.8),
             fontsize=45, color="#FFFCF8", ha="center", va="center",
             arrowprops=dict(arrowstyle="wedge, tail_width=0.5", color="black"),
             bbox=dict(boxstyle="circle, pad=0.4", facecolor="black", linewidth=0.3))

ax.set_axis_off()

# Save as PNG with transparent background
fig.savefig('gauge.png', transparent=True)

def erase_bg_crop_and_html(input_image, output_filename, resize_factor):
    with Image.open(input_image) as img:
        # Convert image to grayscale for whitespace detection
        gray_img = img.convert('L')

        # Convert grayscale image to numpy array
        np_img = np.array(gray_img)

        # Define a threshold for what counts as whitespace (0 is black, 255 is white)
        threshold = 240
        mask = np_img < threshold

        # Find bounding box of non-white areas
        coords = np.argwhere(mask)
        if coords.size == 0:
            raise ValueError("No non-white areas detected in the image.")

        top, left = coords.min(axis=0)
        bottom, right = coords.max(axis=0)

        # Calculate width and height of the bounding box
        bbox_width = right - left
        bbox_height = bottom - top

        # Determine size of the square crop
        square_size = max(bbox_width, bbox_height)

        # Center the bounding box in the square crop
        center_x = left + bbox_width / 2
        center_y = top + bbox_height / 2

        new_left = int(center_x - square_size / 2)
        new_top = int(center_y - square_size / 2)
        new_right = int(center_x + square_size / 2)
        new_bottom = int(center_y + square_size / 2)

        # Ensure the coordinates are within image bounds
        new_left = max(0, new_left)
        new_top = max(0, new_top)
        new_right = min(img.width, new_right)
        new_bottom = min(img.height, new_bottom)

        # Crop the image
        cropped_gauge = img.crop((new_left, new_top, new_right, new_bottom))

        # Resize the cropped image
        new_width = int(cropped_gauge.width * resize_factor)
        new_height = int(cropped_gauge.height * resize_factor)
        resized_gauge = cropped_gauge.resize((new_width, new_height), Image.LANCZOS)

        # Save the cropped image as PNG
        png_path = f"{output_filename}.png"
        resized_gauge.save(png_path)

        # Generate HTML
        buffered = BytesIO()
        resized_gauge.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        html_content = f"""
                <html>
                <body>
                    <img src="data:image/png;base64,{img_str}" alt="Gauge Graph">
                </body>
                </html>
                """

        html_path = f"{output_filename}.html"
        with open(html_path, 'w') as f:
            f.write(html_content)

    return png_path, html_path

# Crop and save as html
erase_bg_crop_and_html('gauge.png', "gauge_cropped", 0.55)



# X image
# Load images
background = Image.open('background.png')
gauge_cropped = Image.open('gauge_cropped.png')

# Resize background
new_width = int(gauge_cropped.width * 1.2)
new_height = new_width  # Ensure background is square
background = background.resize((new_width, new_height), Image.LANCZOS)

# Center the gauge_cropped on the background but lower it a bit
bg_width, bg_height = background.size
gauge_width, gauge_height = gauge_cropped.size
x_center = (bg_width - gauge_width) // 2
y_center = (bg_height - gauge_height) // 2 + 48  # Lower the gauge by 30 pixels
background.paste(gauge_cropped, (x_center, y_center), gauge_cropped)

# Draw text on the image
draw = ImageDraw.Draw(background)
font_path = "/Library/Fonts/Helvetica.ttc"  # Make sure this path is correct
font_title = ImageFont.truetype(font_path, 50)   # For title text
font_subtitle = ImageFont.truetype(font_path, 30) # For subtitle text
font_footer = ImageFont.truetype(font_path, 20)
title = "AI Risk Index"
subtitle = "Quantifying misaligned AI risk"
footer_left = "22 Jul 2024"
footer_right = "airiskindex.com"

# Margins
left_margin = 30
right_margin = 30

# Draw the title
draw.text((left_margin, 40), title, fill="black", font=font_title)
# Draw the subtitle
draw.text((left_margin, 95), subtitle, fill="#6b6b6b", font=font_subtitle)
# Draw the divider line
line_y = bg_height - 55
draw.line([(left_margin, line_y), (bg_width - right_margin, line_y)], fill="darkgrey", width=2)
# Draw the footer left
draw.text((left_margin, bg_height - 45), footer_left, fill="darkgrey", font=font_footer)
# Draw the footer right
bbox = draw.textbbox((0, 0), footer_right, font=font_footer)
text_width = bbox[2] - bbox[0]
draw.text((bg_width - text_width - right_margin, bg_height - 45), footer_right, fill="darkgrey", font=font_footer)

# Save the final image
background.save("gauge_x.png")
