import os
import numpy as np
import pandas as pd
import base64
import tempfile
import subprocess
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont
from warnings import filterwarnings
filterwarnings('ignore')

from indicators import airi_score, hist_invcap, hist_invsaf, hist_rsa, hist_psa, hist_airi

# Debugging print to check airi_score after import
print(f"Imported airi_score: {airi_score}")

def create_html(png_path, output_filename, img_type="web", new_width=None, new_height=None):
    try:
        with open(png_path, "rb") as image_file:
            img_str = base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error reading image file: {e}")
        return

    img_attrs = ''
    if img_type == "x":
        if new_width and new_height:
            img_attrs = f'width="{new_width}" height="{new_height}"'
        else:
            print("Invalid parameters for image type 'x'. Please provide new_width and new_height.")
            return

    html_content = f"""
    <html>
    <body>
        <img src="data:image/png;base64,{img_str}" {img_attrs}>
    </body>
    </html>
    """

    html_path = f"{output_filename}.html"
    with open(html_path, 'w') as f:
        f.write(html_content)
    print(f"HTML file created at {html_path}")

    return html_path


def commit_to_github(file_paths, branch_name="gh-pages", remote_name="origin"):
    try:
        subprocess.run(["git", "checkout", branch_name], check=True)

        # Add and commit only the HTML files
        for file_path in file_paths:
            if file_path.endswith(".html"):
                subprocess.run(["git", "add", file_path], check=True)
                commit_message = f"Add file: {os.path.basename(file_path)}"
                subprocess.run(["git", "commit", "-m", commit_message], check=True)

        # Push the branch to remote
        subprocess.run(["git", "push", "-u", remote_name, branch_name], check=True)
        print(f"Files committed to branch: {branch_name}")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


def create_his_graph(y_values, filename):
    try:
        end_date = datetime.today()
        end_date -= timedelta(days=end_date.weekday() % 7)
        dates = [end_date - timedelta(weeks=i) for i in range(len(y_values))]
        date_labels = [date.strftime('%b %d') for date in reversed(dates)]

        fig = go.Figure(data=go.Scatter(x=date_labels, y=y_values, mode='lines', line=dict(color='blue')))
        y_range = [min(y_values) - (max(y_values) - min(y_values)) * 0.1, max(y_values)]
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            yaxis=dict(side='right', gridcolor='lightgrey', autorange=True, range=y_range),
            xaxis=dict(gridcolor='lightgrey', showline=True, linecolor='black', linewidth=2, mirror=True),
            plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified", hoverlabel=dict(bgcolor="white", font_size=16, font_color="black"))

        html_path = os.path.join(f"{filename}.html")
        fig.write_html(html_path, config={'displayModeBar': False, 'scrollZoom': False, 'showAxisDragHandles': False})
        return html_path

    except Exception as e:
        print(f"Error creating graph: {e}")
        return None


def erase_bg_and_crop(input_image, resize_factor):
    try:
        with Image.open(input_image) as img:
            gray_img = img.convert('L')
            np_img = np.array(gray_img)
            threshold = 240
            mask = np_img < threshold
            coords = np.argwhere(mask)
            if coords.size == 0:
                raise ValueError("No non-white areas detected in the image.")

            top, left = coords.min(axis=0)
            bottom, right = coords.max(axis=0)
            bbox_width, bbox_height = right - left, bottom - top
            square_size = max(bbox_width, bbox_height)
            center_x, center_y = left + bbox_width / 2, top + bbox_height / 2
            new_left = max(0, int(center_x - square_size / 2))
            new_top = max(0, int(center_y - square_size / 2))
            new_right = min(img.width, int(center_x + square_size / 2))
            new_bottom = min(img.height, int(center_y + square_size / 2))

            cropped_gauge = img.crop((new_left, new_top, new_right, new_bottom))
            new_size = (int(cropped_gauge.width * resize_factor), int(cropped_gauge.height * resize_factor))

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                output_path = temp_file.name
                resized_gauge = cropped_gauge.resize(new_size, Image.LANCZOS)
                resized_gauge.save(output_path)

            return output_path
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def main():
    from indicators import airi_score  # Import here for debugging
    print(f"Imported airi_score in main: {airi_score}")

    # Switch to the gh-pages branch at the beginning
    subprocess.run(["git", "checkout", "gh-pages"], check=True)

    # Create historical line charts
    data_and_filenames = [(hist_invcap, "hist_invcap"),
                          (hist_invsaf, "hist_invsaf"),
                          (hist_rsa, "hist_rsa"),
                          (hist_psa, "hist_psa"),
                          (hist_airi, "hist_airi")]

    html_paths = []
    for data, filename in data_and_filenames:
        html_path = create_his_graph(data, filename)
        if html_path:
            html_paths.append(html_path)

    # Create gauge chart
    def deg_to_rad(deg):
        return deg * np.pi / 180

    values = [100, 80, 60, 40, 20, 0]
    x_axis_deg = [0, 36, 72, 108, 144]
    x_axis_vals = [deg_to_rad(i) for i in x_axis_deg]

    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(projection="polar")
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')

    highlighted_edgecolor = "#6b6b6b"
    default_edgecolor = "#D3D3DA"

    print(f"Imported airi_score 2: {airi_score}")

    for i, v in enumerate(values):
        print(f"Checking index {i}: v = {v}, v - 20 = {v - 20}, condition = {v >= airi_score > (v - 20)}")

    highlight_index = next((i for i, v in enumerate(values) if v >= airi_score > (v - 20)), None)

    if highlight_index is None:
        print("No valid highlight index found.")
        return  # Exit the function if no valid index is found

    print(f"Highlight index found: {highlight_index}")

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

    plt.tight_layout(pad=0)
    plt.axis('off')

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        gauge_file_path = temp_file.name
        plt.savefig(gauge_file_path, dpi=100, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

    # Create HTML gauge chart for desktop and mobile
    try:
        cropped_web_path = erase_bg_and_crop(gauge_file_path, 0.55)
        if cropped_web_path:
            html_path = create_html(cropped_web_path, "gauge_web", img_type="x", new_width=450, new_height=360)
            if html_path:
                html_paths.append(html_path)

        cropped_mobile_path = erase_bg_and_crop(gauge_file_path, 0.35)
        if cropped_mobile_path:
            html_path = create_html(cropped_mobile_path, "gauge_mobile", img_type="x", new_width=300, new_height=240)
            if html_path:
                html_paths.append(html_path)

    except Exception as e:
        print(f"Error processing images: {e}")

    # Create image for X post
    try:
        background = Image.open('background.png')
        gauge_cropped = Image.open(cropped_web_path)  # This assumes the image exists

        # Resize background
        new_width = int(gauge_cropped.width * 1.2)
        new_height = new_width  # Ensure background is square
        gauge_x = background.resize((new_width, new_height), Image.LANCZOS)

        # Center the gauge_cropped on the background and lower it a bit
        bg_width, bg_height = gauge_x.size
        gauge_width, gauge_height = gauge_cropped.size
        x_center = (bg_width - gauge_width) // 2
        y_center = (bg_height - gauge_height) // 2 + 48  # Lower the gauge by 48 pixels
        gauge_x.paste(gauge_cropped, (x_center, y_center), gauge_cropped)

        # Draw text on the image
        draw = ImageDraw.Draw(gauge_x)
        font_path = "/Library/Fonts/Helvetica.ttc"  # Make sure this path is correct
        font_title = ImageFont.truetype(font_path, 50)  # For title text
        font_subtitle = ImageFont.truetype(font_path, 30)  # For subtitle text
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
        draw.text((bg_width - text_width - right_margin, bg_height - 45), footer_right, fill="darkgrey",
                  font=font_footer)

        # Save the final image
        gauge_x_path = "gauge_x.png"
        gauge_x.save(gauge_x_path)

        # html_x_path = create_html(gauge_x_path, "gauge_x", img_type="x", new_width=new_width, new_height=new_height)
        # if html_x_path:
        #     html_paths.append(html_x_path)

    except Exception as e:
        print(f"Error creating X image: {e}")

    commit_to_github(html_paths, branch_name="gh-pages")

    os.remove(gauge_file_path)
    os.remove(cropped_web_path)
    os.remove(cropped_mobile_path)

    # Switch back to the main branch
    subprocess.run(["git", "checkout", "main"], check=True)


if __name__ == "__main__":
    main()
