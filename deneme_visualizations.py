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

from indicators import airi_score, hist_invcap, hist_invsaf, hist_rsa, hist_psa, hist_airi, hist_date_records_formatted

x_post_date = "26 Aug 2024"

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


def create_his_graph(y_values, filename, ticktext):
    try:
        fig = go.Figure(data=go.Scatter(x=hist_date_records_formatted, y=y_values, mode='lines', line=dict(color='blue')))
        y_range = [min(y_values) - (max(y_values) - min(y_values)) * 0.1, max(y_values)]
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            yaxis=dict(side='right', gridcolor='lightgrey', autorange=True, range=y_range),
            xaxis=dict(gridcolor='lightgrey', showline=True, linecolor='black', linewidth=2, mirror=True,
                       ticktext=ticktext),
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
    # Create historical line charts
    data_and_filenames = [(hist_invcap, "hist_invcap"),
                          (hist_invsaf, "hist_invsaf"),
                          (hist_rsa, "hist_rsa"),
                          (hist_psa, "hist_psa"),
                          (hist_airi, "hist_airi")]

    for data, filename in data_and_filenames:
        create_his_graph(data, filename)


if __name__ == "__main__":
    main()
