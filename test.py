import os
import tempfile
import shutil
import base64
import subprocess
import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime, timedelta
from common import *
from indicators import airi_score, hist_invcap, hist_invsaf, hist_rsa, hist_psa, hist_airi


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

    html_path = f"temp_files/{output_filename}.html"
    with open(html_path, 'w') as f:
        f.write(html_content)
    print(f"HTML file created at {html_path}")

    return html_path


def commit_to_github(directory, branch_name="test", remote_name="origin"):
    try:
        # Stash any local changes to avoid conflicts
        subprocess.run(["git", "stash"], check=True)

        # Switch to the specified branch
        subprocess.run(["git", "checkout", branch_name], check=True)

        # Add and commit all HTML files in the directory
        for file in os.listdir(directory):
            if file.endswith(".html"):
                file_path = os.path.join(directory, file)
                subprocess.run(["git", "add", file_path], check=True)
                commit_message = f"Add HTML file: {file}"
                subprocess.run(["git", "commit", "-m", commit_message], check=True)

        # Push the branch to remote
        subprocess.run(["git", "push", "-u", remote_name, branch_name], check=True)
        print(f"HTML files committed to branch: {branch_name}")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
    finally:
        # Switch back to the main branch
        subprocess.run(["git", "checkout", "main"], check=True)
        # Restore stashed changes
        subprocess.run(["git", "stash", "pop"], check=True)
        # Cleanup the temp_files directory
        if os.path.exists(directory):
            shutil.rmtree(directory)


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

        # Define the path for the HTML file
        html_path = os.path.join("temp_files", f"{filename}.html")
        fig.write_html(html_path, config={'displayModeBar': False, 'scrollZoom': False, 'doubleClick': False,
                                          'showAxisDragHandles': False})
        return html_path

    except Exception as e:
        print(f"Error creating graph: {e}")
        return None


def main():
    # Create temp_files directory if not exists
    os.makedirs("temp_files", exist_ok=True)

    # Create and commit graphs
    data_and_filenames = [(hist_invcap, "hist_invcap"),
                          (hist_invsaf, "hist_invsaf"),
                          (hist_rsa, "hist_rsa"),
                          (hist_psa, "hist_psa"),
                          (hist_airi, "hist_airi")]

    for data, filename in data_and_filenames:
        html_path = create_his_graph(data, filename)
        if html_path:
            commit_to_github("temp_files", branch_name="test")

    # Gauge chart
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
                 fontsize=45, color="black", ha="center")

    plt.tight_layout(pad=0)
    plt.axis('off')

    gauge_file_path = "temp_files/gauge.png"
    plt.savefig(gauge_file_path, dpi=100, bbox_inches='tight', pad_inches=0, transparent=True)

    plt.close()

    # HTML for gauge
    create_html(gauge_file_path, "gauge_web", img_type="x", new_width=450, new_height=450)

    # Handle errors and Git operations
    try:
        gauge_cropped_web_path = "temp_files/gauge_cropped_web.png"
        if not os.path.exists(gauge_cropped_web_path):
            raise FileNotFoundError(f"File not found: {gauge_cropped_web_path}")
        gauge_cropped = Image.open(gauge_cropped_web_path)
        print("Gauge image processed successfully.")
    except Exception as e:
        print(f"Error processing image: {e}")

    # Commit and cleanup
    commit_to_github("temp_files", branch_name="test")


if __name__ == "__main__":
    main()
