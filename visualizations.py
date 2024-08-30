import os
import numpy as np
import base64
import tempfile
from io import BytesIO
import subprocess
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont
from warnings import filterwarnings
filterwarnings('ignore')

from indicators import airi_score, hist_invcap, hist_invsaf, hist_rsa, hist_psa, hist_airi, hist_date_records

x_post_date = "19 Aug 2024"

def create_html(input_data, output_filename, img_type="web", new_width=None, new_height=None):
    """
        Creates an HTML file embedding a base64 encoded image, either from a PNG file or a Matplotlib plot.

    Args:
        input_data (str or plt.Figure): The file path to the PNG image or a Matplotlib Figure object.
        output_filename (str): The desired name for the output HTML file (without extension).
        img_type (str, optional): The type of image to embed. Defaults to "web".
                                  If "x", new_width and new_height must be provided.
        new_width (int, optional): The new width for the image when img_type is "x". Defaults to None.
        new_height (int, optional): The new height for the image when img_type is "x". Defaults to None.

    Returns:
        str: The path to the created HTML file. Returns None if an error occurs during file creation.

    Raises:
        Exception: If there are issues reading the PNG file or writing the HTML file.
    """

    try:
        if isinstance(input_data, str):
            # Handle the case where input_data is a file path
            with open(input_data, "rb") as image_file:
                img_str = base64.b64encode(image_file.read()).decode("utf-8")
        elif isinstance(input_data, plt.Figure):
            # Handle the case where input_data is a Matplotlib Figure
            buffer = BytesIO()
            input_data.savefig(buffer, format='png', bbox_inches='tight', transparent=True)
            buffer.seek(0)
            img_str = base64.b64encode(buffer.read()).decode("utf-8")
        else:
            print("Invalid input_data type. Must be a file path or a Matplotlib Figure.")
            return None

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

    except Exception as e:
        print(f"Error creating HTML file: {e}")
        return None


def commit_to_github(file_paths, branch_name="gh-pages", remote_name="origin"):
    try:
        subprocess.run(["git", "checkout", branch_name], check=True)

        # Add and commit only the HTML and PNG files
        for file_path in file_paths:
            if file_path.endswith(".html") or file_path.endswith(".png"):
                subprocess.run(["git", "add", file_path], check=True)
                commit_message = f"Add file: {os.path.basename(file_path)}"
                subprocess.run(["git", "commit", "-m", commit_message], check=True)

        # Push the branch to remote
        subprocess.run(["git", "push", "-u", remote_name, branch_name], check=True)
        print(f"\nNumber of files to be committed: {len(file_paths)}")
        print(f"\nFiles committed to branch: {branch_name}")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


def git_add_all():
    try:
        result = subprocess.run(
            ["git", "add", "."],
            check=True,
            text=True,
            capture_output=True)
        print("Output:", result.stdout)
        print("All changes staged successfully.")
    except subprocess.CalledProcessError as e:
        print("Error staging changes:", e.stderr)


def push_to_origin(branch_name='gh-pages', remote_name='origin'):
    try:
        # Run the git push command
        result = subprocess.run(
            ["git", "push", "-u", remote_name, branch_name],
            check=True,
            text=True,
            capture_output=True)
        print("Output:", result.stdout)
        print("Pushed successfully to", remote_name, branch_name)
    except subprocess.CalledProcessError as e:
        print("Error pushing to remote:", e.stderr)


def create_his_graph(y_values, filename):
    try:
        fig = go.Figure(data=go.Scatter(x=hist_date_records, y=y_values, mode='lines', line=dict(color='blue')))
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
            new_size = int(cropped_gauge.width * resize_factor), int(cropped_gauge.height * resize_factor)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                output_path = temp_file.name
                resized_gauge = cropped_gauge.resize(new_size, Image.LANCZOS)
                resized_gauge.save(output_path)

            return output_path
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def main():
    # Switch to the gh-pages branch at the beginning
    subprocess.run(["git", "checkout", "gh-pages"], check=True)

    # Create historical line charts
    data_and_filenames = [(hist_invcap, "hist_invcap"),
                          (hist_invsaf, "hist_invsaf"),
                          (hist_rsa, "hist_rsa"),
                          (hist_psa, "hist_psa"),
                          (hist_airi, "hist_airi")]

    file_paths = []
    for data, filename in data_and_filenames:
        html_path = create_his_graph(data, filename)
        if html_path:
            file_paths.append(html_path)

    # Create gauge chart
    print("\n------CREATE GAUGE GRAPH------")
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

    print(f"airi_score: {airi_score}")
#   for i, v in enumerate(values):
#       print(f"Checking index {i}: v = {v}, v - 20 = {v - 20}, condition = {v >= airi_score > (v - 20)}")

    highlight_index = next((i for i, v in enumerate(values) if v >= airi_score > (v - 20)), None)
    if highlight_index is not None:
        highlight_val = values[highlight_index]
    if highlight_index is None:
        print("No valid highlight index found.")
        return  # Exit the function if no valid index is found

    print(f"Highlighted range for airi_score: ({highlight_val-20}-{highlight_val})")

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
        plt.annotate(text, xy=(deg_to_rad(x), y), rotation=rot, fontweight="bold", fontsize=22, color=color,
                     rotation_mode='anchor', transform=ax.transData, ha="center")

    ticks = [100, "•", 80, "•", 60, "•", 40, "•", 20, "•", 0]
    for i, (loc, val) in enumerate(
            zip([deg_to_rad(i) for i in [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]], ticks)):
        color = highlighted_edgecolor
        ha = "right" if i <= 4 else ("center" if i == 5 else "left")
        plt.annotate(val, xy=(loc, 1.3), ha=ha, va="center", fontsize="20", color=color)

    plt.annotate(round(airi_score), xytext=(0, 0), xy=(deg_to_rad((100 - round(airi_score)) / 100 * 180), 1.8),
                  fontsize=55, color="#FFFCF8", va="center", ha="center",
                  arrowprops=dict(arrowstyle="wedge, tail_width=0.5", color="black"),
                  bbox=dict(boxstyle="circle, pad=0.4", facecolor="black", linewidth=0.3))

    # Set theta limits to crop out bottom half
    ax.set_thetalim(0, np.pi)
    ax.set_axis_off()
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        gauge_raw_temp_file_path = temp_file.name
        plt.savefig(gauge_raw_temp_file_path, dpi=200, bbox_inches='tight', pad_inches=0, transparent=True)

    plt.close()

    # plt.savefig("raw_gauge.png", dpi=200, bbox_inches='tight', pad_inches=0, transparent=True)

    # Create HTML gauge chart for desktop and mobile
    print("\n------HTML GAUGE MAIN------")
    try:
        cropped_web_path = erase_bg_and_crop(gauge_raw_temp_file_path, 0.22)
        if cropped_web_path:
            html_path = create_html(cropped_web_path, "gauge_web", img_type="web")#, new_width=450, new_height=360)
            if html_path:
                file_paths.append(html_path)

        cropped_mobile_path = erase_bg_and_crop(gauge_raw_temp_file_path, 0.13)
        if cropped_mobile_path:
            html_path = create_html(cropped_mobile_path, "gauge_mobile", img_type="web")#, new_width=300, new_height=240)
            if html_path:
                file_paths.append(html_path)

    except Exception as e:
        print(f"Error processing images: {e}")

    """
        try:
        html_path_gauge_raw = create_html(fig, "gauge_web", 0.55)
        html_path_gauge_mobile = create_html(fig, "gauge_mobile", 0.35)
        if html_path_gauge_raw:
            file_paths.append(html_path_gauge_raw)
        if html_path_gauge_mobile:
            file_paths.append(html_path_gauge_mobile)
    except Exception as e:
        print(f"Error creating HTML files: {e}")
    """

#    Create image for X post
    print("\n------X IMAGE------")
    try:
        background = Image.open('background.png')
        gauge_raw = Image.open(gauge_raw_temp_file_path)

        # Resize background
        new_width = int(gauge_raw.width * 1.19)
        new_height = new_width  # Ensure background is square
        gauge_x = background.resize((new_width, new_height), Image.LANCZOS)

        # Center the gauge_cropped on the background and lower it a bit
        bg_width, bg_height = gauge_x.size
        gauge_width, gauge_height = gauge_raw.size
        x_center = (bg_width - gauge_width) // 2
        y_center = (bg_height - gauge_height) // 2 + 95  # Lower the gauge by 95 pixels
        gauge_x.paste(gauge_raw, (x_center, y_center), gauge_raw)

        # Draw text on the image
        draw = ImageDraw.Draw(gauge_x)
        font_path = "/Library/Fonts/Helvetica.ttc"
        font_title = ImageFont.truetype(font_path, 230)
        font_subtitle = ImageFont.truetype(font_path, 140)
        font_footer = ImageFont.truetype(font_path, 95)
        title = "AI Risk Index"
        subtitle = "Quantifying misaligned AI risk"
        footer_left = x_post_date
        footer_right = "airiskindex.com"

        left_margin = 140
        right_margin = 140

        # Draw title
        draw.text((left_margin, 180), title, fill="black", font=font_title)
        # Draw subtitle
        draw.text((left_margin, 440), subtitle, fill="#6b6b6b", font=font_subtitle)
        # Draw divider line
        line_y = bg_height - 255
        draw.line([(left_margin, line_y), (bg_width - right_margin, line_y)], fill="#6b6b6b", width=10)
        # Draw footer left
        draw.text((left_margin, bg_height - 210), footer_left, fill="#6b6b6b", font=font_footer)
        # Draw footer right
        bbox = draw.textbbox((0, 0), footer_right, font=font_footer)
        text_width = bbox[2] - bbox[0]
        draw.text((bg_width - text_width - right_margin, bg_height - 210), footer_right, fill="#6b6b6b",
                  font=font_footer)

        stripped_x_post_date = x_post_date.replace(" ", "")
        gauge_x_path = f"gauge_x_{stripped_x_post_date}.png"
        gauge_x.save(gauge_x_path)
        file_paths.append(gauge_x_path)
        print(f"X image created at {gauge_x_path}")

    except Exception as e:
        print(f"Error creating X image: {e}")

    print("\n------GIT COMMIT CHECK------")
    print("Files to be created per update:")
    print("- Historical AIRI + 4 indicators = 5")
    print("- Web + mobile gauge graphs = 2")
    print("- X image = 1")
    print(f"\n{len(file_paths)}/8 files in the file_paths list.")

    """
        # TODO do i need this image in html form for twitter bot?
        # Save the final image
        # gauge_x_path = "gauge_x.png"
        # #gauge_x.save(gauge_x_path)
        # html_x_path = create_html(gauge_x_path, "gauge_x", img_type="x", new_width=new_width, new_height=new_height)
        # if html_x_path:
        #     html_paths.append(html_x_path)

    except Exception as e:
        print(f"Error creating X image: {e}")
    
    """
    git_add_all()
    push_to_origin('gh-pages', 'origin')
    commit_to_github(file_paths, branch_name="gh-pages")
    push_to_origin('gh-pages', 'origin')

    os.remove(gauge_raw_temp_file_path)
    os.remove(cropped_web_path)
    os.remove(cropped_mobile_path)
    os.remove(html_path)
    os.remove(gauge_x_path)

    # Switch back to the main branch
    # subprocess.run(["git", "checkout", "main"], check=True)

if __name__ == "__main__":
    main()
