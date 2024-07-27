"""

All code contributed to Exifa.net is © 2024 by Sahir Maharaj. 
The content is licensed under the Creative Commons Attribution 4.0 International License. 
This allows for sharing and adaptation, provided appropriate credit is given, and any changes made are indicated.

When using the code from Exifa.net, please credit as follows: "Code sourced from Exifa.net, authored by Sahir Maharaj, 2024."

For reporting bugs, requesting features, or further inquiries, please reach out to Sahir Maharaj at sahir@sahirmaharaj.com.

Connect with Sahir Maharaj on LinkedIn for updates and potential collaborations: https://www.linkedin.com/in/sahir-maharaj/

Hire Sahir Maharaj: https://topmate.io/sahirmaharaj/362667
"""

import streamlit as st
import replicate
import os
import pdfplumber
from docx import Document
import pandas as pd
from io import BytesIO
from transformers import AutoTokenizer
import exifread
import requests
from PIL import Image
import numpy as np
import plotly.express as px
import matplotlib.colors as mcolors
import plotly.graph_objs as go
import streamlit.components.v1 as components
import random

config = {
    "toImageButtonOptions": {
        "format": "png",
        "filename": "custom_image",
        "height": 720,
        "width": 480,
        "scale": 6,
    }
}

icons = {
    "assistant": "https://raw.githubusercontent.com/sahirmaharaj/exifa/2f685de7dffb583f2b2a89cb8ee8bc27bf5b1a40/img/assistant-done.svg",
    "user": "https://raw.githubusercontent.com/sahirmaharaj/exifa/2f685de7dffb583f2b2a89cb8ee8bc27bf5b1a40/img/user-done.svg",
}

particles_js = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Particles.js</title>
  <style>
  #particles-js {
    position: fixed;
    width: 100vw;
    height: 100vh;
    top: 0;
    left: 0;
    z-index: -1; /* Send the animation to the back */
  }
  .content {
    position: relative;
    z-index: 1;
    color: white;
  }
  
</style>
</head>
<body>
  <div id="particles-js"></div>
  <div class="content">
    <!-- Placeholder for Streamlit content -->
  </div>
  <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
  <script>
    particlesJS("particles-js", {
      "particles": {
        "number": {
          "value": 300,
          "density": {
            "enable": true,
            "value_area": 800
          }
        },
        "color": {
          "value": "#ffffff"
        },
        "shape": {
          "type": "circle",
          "stroke": {
            "width": 0,
            "color": "#000000"
          },
          "polygon": {
            "nb_sides": 5
          },
          "image": {
            "src": "img/github.svg",
            "width": 100,
            "height": 100
          }
        },
        "opacity": {
          "value": 0.5,
          "random": false,
          "anim": {
            "enable": false,
            "speed": 1,
            "opacity_min": 0.2,
            "sync": false
          }
        },
        "size": {
          "value": 2,
          "random": true,
          "anim": {
            "enable": false,
            "speed": 40,
            "size_min": 0.1,
            "sync": false
          }
        },
        "line_linked": {
          "enable": true,
          "distance": 100,
          "color": "#ffffff",
          "opacity": 0.22,
          "width": 1
        },
        "move": {
          "enable": true,
          "speed": 0.2,
          "direction": "none",
          "random": false,
          "straight": false,
          "out_mode": "out",
          "bounce": true,
          "attract": {
            "enable": false,
            "rotateX": 600,
            "rotateY": 1200
          }
        }
      },
      "interactivity": {
        "detect_on": "canvas",
        "events": {
          "onhover": {
            "enable": true,
            "mode": "grab"
          },
          "onclick": {
            "enable": true,
            "mode": "repulse"
          },
          "resize": true
        },
        "modes": {
          "grab": {
            "distance": 100,
            "line_linked": {
              "opacity": 1
            }
          },
          "bubble": {
            "distance": 400,
            "size": 2,
            "duration": 2,
            "opacity": 0.5,
            "speed": 1
          },
          "repulse": {
            "distance": 200,
            "duration": 0.4
          },
          "push": {
            "particles_nb": 2
          },
          "remove": {
            "particles_nb": 3
          }
        }
      },
      "retina_detect": true
    });
  </script>
</body>
</html>
"""

st.set_page_config(page_title="Exifa.net", page_icon="✨", layout="wide")

welcome_messages = [
    "Hello! I'm Exifa, an AI assistant designed to make image metadata meaningful. Ask me anything!",
    "Hi! I'm Exifa, an AI-powered assistant for extracting and explaining EXIF data. How can I help you today?",
    "Hey! I'm Exifa, your AI-powered guide to understanding the metadata in your images. What would you like to explore?",
    "Hi there! I'm Exifa, an AI-powered tool built to help you make sense of your image metadata. How can I help you today?",
    "Hello! I'm Exifa, an AI-driven tool designed to help you understand your images' metadata. What can I do for you?",
    "Hi! I'm Exifa, an AI-driven assistant designed to make EXIF data easy to understand. How can I help you today?",
    "Welcome! I'm Exifa, an intelligent AI-powered tool for extracting and explaining EXIF data. How can I assist you today?",
    "Hello! I'm Exifa, your AI-powered guide for understanding image metadata. Ask me anything!",
    "Hi! I'm Exifa, an intelligent AI assistant ready to help you understand your images' metadata. What would you like to explore?",
    "Hey! I'm Exifa, an AI assistant for extracting and explaining EXIF data. How can I help you today?",
]

message = random.choice(welcome_messages)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": message}]
if "exif_df" not in st.session_state:
    st.session_state["exif_df"] = pd.DataFrame()
if "url_exif_df" not in st.session_state:
    st.session_state["url_exif_df"] = pd.DataFrame()
if "show_expanders" not in st.session_state:
    st.session_state.show_expanders = True
if "reset_trigger" not in st.session_state:
    st.session_state.reset_trigger = False
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = None
if "image_url" not in st.session_state:
    st.session_state["image_url"] = ""
if "follow_up" not in st.session_state:
    st.session_state.follow_up = False
if "show_animation" not in st.session_state:
    st.session_state.show_animation = True


def clear_url():
    st.session_state["image_url"] = ""


def clear_files():
    st.session_state["uploaded_files"] = None
    st.session_state["file_uploader_key"] = not st.session_state.get(
        "file_uploader_key", False
    )


def download_image(data):
    st.download_button(
        label="⇩ Download Image",
        data=data,
        file_name="image_no_exif.jpg",
        mime="image/jpeg",
    )


def clear_chat_history():

    st.session_state.reset_trigger = not st.session_state.reset_trigger
    st.session_state.show_expanders = True

    st.session_state.show_animation = True

    st.session_state.messages = [{"role": "assistant", "content": message}]

    st.session_state["exif_df"] = pd.DataFrame()
    st.session_state["url_exif_df"] = pd.DataFrame()
    uploaded_files = ""

    if "uploaded_files" in st.session_state:
        del st.session_state["uploaded_files"]
    if "image_url" in st.session_state:
        st.session_state["image_url"] = ""
    st.cache_data.clear()

    st.success("Chat History Cleared!")


def clear_exif_data(image_input):
    if isinstance(image_input, BytesIO):
        image_input.seek(0)
        image = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        raise ValueError("Unsupported image input type")
    data = list(image.getdata())
    image_without_exif = Image.new(image.mode, image.size)
    image_without_exif.putdata(data)

    buffered = BytesIO()
    image_without_exif.save(buffered, format="JPEG", quality=100, optimize=True)
    buffered.seek(0)
    return buffered.getvalue()


with st.sidebar:

    image_url = (
        "https://raw.githubusercontent.com/sahirmaharaj/exifa/main/img/Exifa.gif"
    )

    st.markdown(
        f"""
        <div style='display: flex; align-items: center;'>
            <img src='{image_url}' style='width: 50px; height: 50px; margin-right: 30px;'>
            <h1 style='margin: 0;'>Exifa.net</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    expander = st.expander("🗀 File Input")
    with expander:

        image_url = st.text_input(
            "Enter image URL for EXIF analysis:",
            key="image_url",
            on_change=clear_files,
            value=st.session_state.image_url,
        )

        file_uploader_key = "file_uploader_{}".format(
            st.session_state.get("file_uploader_key", False)
        )

        uploaded_files = st.file_uploader(
            "Upload local files:",
            type=["txt", "pdf", "docx", "csv", "jpg", "png", "jpeg"],
            key=file_uploader_key,
            on_change=clear_url,
            accept_multiple_files=True,
        )

        if uploaded_files is not None:
            st.session_state["uploaded_files"] = uploaded_files
    expander = st.expander("⚒ Model Configuration")
    with expander:

        if "REPLICATE_API_TOKEN" in st.secrets:
            replicate_api = st.secrets["REPLICATE_API_TOKEN"]
        else:
            replicate_api = st.text_input("Enter Replicate API token:", type="password")
            if not (replicate_api.startswith("r8_") and len(replicate_api) == 40):
                st.warning("Please enter your Replicate API token.", icon="⚠️")
                st.markdown(
                    "**Don't have an API token?** Head over to [Replicate](https://replicate.com/account/api-tokens) to sign up for one."
                )
        os.environ["REPLICATE_API_TOKEN"] = replicate_api
        st.subheader("Adjust model parameters")
        temperature = st.slider(
            "Temperature", min_value=0.01, max_value=5.0, value=0.3, step=0.01
        )
        top_p = st.slider("Top P", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
        max_new_tokens = st.number_input(
            "Max New Tokens", min_value=1, max_value=1024, value=512
        )
        min_new_tokens = st.number_input(
            "Min New Tokens", min_value=0, max_value=512, value=0
        )
        presence_penalty = st.slider(
            "Presence Penalty", min_value=0.0, max_value=2.0, value=1.15, step=0.05
        )
        frequency_penalty = st.slider(
            "Frequency Penalty", min_value=0.0, max_value=2.0, value=0.2, step=0.05
        )
        stop_sequences = st.text_area("Stop Sequences", value="<|im_end|>", height=100)
    if uploaded_files and not st.session_state["exif_df"].empty:
        with st.expander("🗏 EXIF Details"):
            st.dataframe(st.session_state["exif_df"])
    if image_url and not st.session_state["url_exif_df"].empty:
        with st.expander("🗏 EXIF Details"):
            st.dataframe(st.session_state["url_exif_df"])
    base_prompt = """
    
    You are an expert EXIF Analyser. The user will provide an image file and you will explain the file EXIF in verbose detail.
    
    Pay careful attention to the data of the EXIF image and create a profile for the user who took this image. 
    
    1. Make inferences on things like location, budget, experience, etc. (2 paragraphs) 
    2. Make as many inferences as possible about the exif data in the next 3 paragraphs.
    
    3. Please follow this format, style, pacing and structure. 
    4. In addition to the content above, provide 1 more paragraph about the users financial standing based on the equipment they are using and estimate their experience.
    
    DO NOT skip any steps.
    
    FORMAT THE RESULT IN MULTIPLE PARAGRAPHS
    
    Do not keep talking and rambling on - Get to the point. 
    
    """

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.type == "application/pdf":
                with pdfplumber.open(uploaded_file) as pdf:
                    pages = [page.extract_text() for page in pdf.pages]
                file_text = "\n".join(pages) if pages else ""
            elif uploaded_file.type == "text/plain":
                file_text = str(uploaded_file.read(), "utf-8")
            elif (
                uploaded_file.type
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ):
                doc = Document(uploaded_file)
                file_text = "\n".join([para.text for para in doc.paragraphs])
            elif uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
                file_text = df.to_string(index=False)
            elif uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
                import tempfile

                with tempfile.NamedTemporaryFile(delete=False) as temp:
                    temp.write(uploaded_file.read())
                    temp.flush()
                    temp.close()
                    with open(temp.name, "rb") as file:
                        tags = exifread.process_file(file)
                    exif_data = {}
                    for tag in tags.keys():
                        if tag not in [
                            "JPEGThumbnail",
                            "TIFFThumbnail",
                            "Filename",
                            "EXIF MakerNote",
                        ]:
                            exif_data[tag] = str(tags[tag])
                    df = pd.DataFrame(exif_data, index=[0])
                    df.insert(loc=0, column="Image Feature", value=["Value"] * len(df))
                    df = df.transpose()
                    df.columns = df.iloc[0]
                    df = df.iloc[1:]

                    st.session_state["exif_df"] = df

                    file_text = "\n".join(
                        [
                            f"{tag}: {tags[tag]}"
                            for tag in tags.keys()
                            if tag
                            not in (
                                "JPEGThumbnail",
                                "TIFFThumbnail",
                                "Filename",
                                "EXIF MakerNote",
                            )
                        ]
                    )
                    os.unlink(temp.name)
            base_prompt += "\n" + file_text
    if image_url:
        try:
            response = requests.head(image_url)
            if response.headers["Content-Type"] in [
                "image/jpeg",
                "image/png",
                "image/jpg",
            ]:
                response = requests.get(image_url)
                response.raise_for_status()
                image_data = BytesIO(response.content)
                image = Image.open(image_data)
                image.load()

                tags = exifread.process_file(image_data)

                exif_data = {}
                for tag in tags.keys():
                    if tag not in [
                        "JPEGThumbnail",
                        "TIFFThumbnail",
                        "Filename",
                        "EXIF MakerNote",
                    ]:
                        exif_data[tag] = str(tags[tag])
                df = pd.DataFrame(exif_data, index=[0])
                df.insert(loc=0, column="Image Feature", value=["Value"] * len(df))
                df = df.transpose()
                df.columns = df.iloc[0]
                df = df.iloc[1:]

                st.session_state["url_exif_df"] = df

                file_text = "\n".join(
                    [
                        f"{tag}: {tags[tag]}"
                        for tag in tags.keys()
                        if tag
                        not in (
                            "JPEGThumbnail",
                            "TIFFThumbnail",
                            "Filename",
                            "EXIF MakerNote",
                        )
                    ]
                )
                base_prompt += "\n" + file_text
            else:

                pass
        except requests.RequestException:

            pass

        def load_image(file):
            if isinstance(file, str):
                response = requests.get(file)
                response.raise_for_status()
                return Image.open(BytesIO(response.content))
            elif isinstance(file, bytes):
                return Image.open(BytesIO(file))
            else:
                return Image.open(file)

        uploaded_file = image

        with st.expander("⛆ RGB Channel"):

            def get_channel_image(image, channels):

                data = np.array(image)

                channel_data = np.zeros_like(data)

                for channel in channels:
                    channel_data[:, :, channel] = data[:, :, channel]
                return Image.fromarray(channel_data)

            channels = st.multiselect(
                "Select channels:",
                ["Red", "Green", "Blue"],
                default=["Red", "Green", "Blue"],
            )

            if channels:
                channel_indices = [
                    0 if channel == "Red" else 1 if channel == "Green" else 2
                    for channel in channels
                ]
                combined_image = get_channel_image(image, channel_indices)
                st.image(combined_image, use_column_width=True)
            else:
                st.image(image, use_column_width=True)
        with st.expander("〽 HSV Distribution"):

            def get_hsv_histogram(image):

                hsv_image = image.convert("HSV")
                data = np.array(hsv_image)

                hue_hist, _ = np.histogram(data[:, :, 0], bins=256, range=(0, 256))
                saturation_hist, _ = np.histogram(
                    data[:, :, 1], bins=256, range=(0, 256)
                )
                value_hist, _ = np.histogram(data[:, :, 2], bins=256, range=(0, 256))

                histogram_df = pd.DataFrame(
                    {
                        "Hue": hue_hist,
                        "Saturation": saturation_hist,
                        "Value": value_hist,
                    }
                )

                return histogram_df

            hsv_histogram_df = get_hsv_histogram(image)

            st.line_chart(hsv_histogram_df)
        with st.expander("☄ Color Distribution"):
            if image_url:
                image = load_image(image_url)
                if image:

                    def color_distribution_sunburst(data):
                        data = np.array(data)
                        red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]
                        color_intensity = {"color": [], "intensity": [], "count": []}
                        for name, channel in zip(
                            ["Red", "Green", "Blue"], [red, green, blue]
                        ):
                            unique, counts = np.unique(channel, return_counts=True)
                            color_intensity["color"].extend([name] * len(unique))
                            color_intensity["intensity"].extend(unique)
                            color_intensity["count"].extend(counts)
                        df = pd.DataFrame(color_intensity)
                        fig = px.sunburst(
                            df,
                            path=["color", "intensity"],
                            values="count",
                            color="color",
                            color_discrete_map={
                                "Red": "#ff6666",
                                "Green": "#85e085",
                                "Blue": "#6666ff",
                            },
                        )
                        return fig

                    fig = color_distribution_sunburst(image)
                    st.plotly_chart(fig, use_container_width=True)
        with st.expander("🕸 3D Color Space"):

            def plot_3d_color_space(data, skip_factor):
                sample = data[::skip_factor, ::skip_factor].reshape(-1, 3)

                normalized_colors = sample / 255.0

                trace = go.Scatter3d(
                    x=sample[:, 0],
                    y=sample[:, 1],
                    z=sample[:, 2],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=["rgb({},{},{})".format(r, g, b) for r, g, b in sample],
                        opacity=0.8,
                    ),
                )
                layout = go.Layout(
                    scene=dict(
                        xaxis=dict(title="Red"),
                        yaxis=dict(title="Green"),
                        zaxis=dict(title="Blue"),
                        camera=dict(eye=dict(x=1.25, y=1.25, z=1.25)),
                    ),
                    margin=dict(l=0, r=0, b=0, t=30),
                )
                fig = go.Figure(data=[trace], layout=layout)
                return fig

            skip_factor = 8

            if isinstance(uploaded_file, Image.Image):
                data = np.array(uploaded_file)
            else:
                data = np.array(Image.open(uploaded_file))
            fig = plot_3d_color_space(data, skip_factor)
            st.plotly_chart(fig, use_container_width=True, config=config)
        with st.expander("‏𖦹 Pixel Density Polar"):

            def pixel_density_polar_plot(image):
                image_data = np.array(image)
                hsv_data = mcolors.rgb_to_hsv(image_data / 255.0)
                hue = hsv_data[:, :, 0].flatten()

                hist, bins = np.histogram(hue, bins=360, range=(0, 1))
                theta = np.linspace(0, 360, len(hist), endpoint=False)

                fig = px.bar_polar(
                    r=hist,
                    theta=theta,
                    template="seaborn",
                    color_discrete_sequence=["red"],
                )
                fig.update_traces(marker=dict(line=dict(color="red", width=1)))
                fig.update_layout()

                return fig

            if uploaded_file is not None:
                if isinstance(uploaded_file, Image.Image):
                    image = uploaded_file
                else:
                    image = Image.open(uploaded_file)
                fig = pixel_density_polar_plot(image)
                st.plotly_chart(fig, use_container_width=True, config=config)
        with st.expander("ᨒ 3D Surface (Color Intensities)"):

            def surface_plot_image_intensity(data):
                intensity = np.mean(data, axis=2)
                sample_size = int(intensity.shape[0] * 0.35)
                intensity_sample = intensity[:sample_size, :sample_size]
                fig = go.Figure(
                    data=[go.Surface(z=intensity_sample, colorscale="Viridis")]
                )
                fig.update_layout(autosize=True)
                return fig

            if isinstance(uploaded_file, Image.Image):
                data = np.array(uploaded_file)
            else:
                data = np.array(Image.open(uploaded_file))
            fig = surface_plot_image_intensity(data)
            st.plotly_chart(fig, use_container_width=True, config=config)
        with st.expander("🖌 Color Palette"):

            def extract_color_palette(image, num_colors=6):
                image = image.resize((100, 100))
                result = image.quantize(colors=num_colors)
                palette = result.getpalette()
                color_counts = result.getcolors()

                colors = [palette[i * 3 : (i + 1) * 3] for i in range(num_colors)]
                counts = [
                    count
                    for count, _ in sorted(
                        color_counts, reverse=True, key=lambda x: x[0]
                    )
                ]
                return colors, counts

            def plot_color_palette(colors, counts):
                fig = go.Figure()
                for i, (color, count) in enumerate(zip(colors, counts)):
                    hex_color = "#%02x%02x%02x" % tuple(color)
                    fig.add_trace(
                        go.Bar(
                            x=[1],
                            y=[hex_color],
                            orientation="h",
                            marker=dict(color=hex_color),
                            hoverinfo="text",
                            hovertext=f"<b>HEX:</b> {hex_color}<br><b>Count:</b> {count}",
                            name="",
                        )
                    )
                fig.update_layout(
                    xaxis=dict(showticklabels=False),
                    yaxis=dict(showticklabels=True),
                    showlegend=False,
                    template="plotly_dark",
                    height=400,
                )
                return fig

            num_colors = st.slider("Number of Colors", 2, 10, 6)

            if isinstance(uploaded_file, Image.Image):
                image = uploaded_file.convert("RGB")
            else:
                image = Image.open(uploaded_file).convert("RGB")
            colors, counts = extract_color_palette(image, num_colors)
            fig = plot_color_palette(colors, counts)
            st.plotly_chart(fig, use_container_width=True, config=config)
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            clean_img = clear_exif_data(image)
            with col1:
                st.button("🗑 Clear Chat History", on_click=clear_chat_history)
            with col2:
                download_image(clean_img)
        st.session_state.reset_trigger = True
    if st.session_state.show_expanders:

        if uploaded_files and not st.session_state["exif_df"].empty:

            with st.expander("⛆ RGB Channel"):

                for uploaded_file in uploaded_files:
                    if uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:

                        def load_image(image_file):
                            return Image.open(image_file)

                        image = load_image(uploaded_file)

                def get_channel_image(image, channels):
                    data = np.array(image)

                    channel_data = np.zeros_like(data)

                    for channel in channels:
                        channel_data[:, :, channel] = data[:, :, channel]
                    return Image.fromarray(channel_data)

                channels = st.multiselect(
                    "Select channels:",
                    ["Red", "Green", "Blue"],
                    default=["Red", "Green", "Blue"],
                )

                if channels:
                    channel_indices = [
                        0 if channel == "Red" else 1 if channel == "Green" else 2
                        for channel in channels
                    ]
                    combined_image = get_channel_image(image, channel_indices)
                    st.image(combined_image, use_column_width=True)
                else:
                    st.image(image, use_column_width=True)
            with st.expander("〽 HSV Distribution"):

                def get_hsv_histogram(image):
                    hsv_image = image.convert("HSV")
                    data = np.array(hsv_image)

                    hue_hist, _ = np.histogram(data[:, :, 0], bins=256, range=(0, 256))
                    saturation_hist, _ = np.histogram(
                        data[:, :, 1], bins=256, range=(0, 256)
                    )
                    value_hist, _ = np.histogram(
                        data[:, :, 2], bins=256, range=(0, 256)
                    )

                    histogram_df = pd.DataFrame(
                        {
                            "Hue": hue_hist,
                            "Saturation": saturation_hist,
                            "Value": value_hist,
                        }
                    )

                    return histogram_df

                hsv_histogram_df = get_hsv_histogram(image)

                st.line_chart(hsv_histogram_df)
            with st.expander("☄ Color Distribution"):

                def color_distribution_sunburst(data):
                    data = np.array(data)

                    red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]
                    color_intensity = {"color": [], "intensity": [], "count": []}
                    for name, channel in zip(
                        ["Red", "Green", "Blue"], [red, green, blue]
                    ):
                        unique, counts = np.unique(channel, return_counts=True)
                        color_intensity["color"].extend([name] * len(unique))
                        color_intensity["intensity"].extend(unique)
                        color_intensity["count"].extend(counts)
                    df = pd.DataFrame(color_intensity)
                    fig = px.sunburst(
                        df,
                        path=["color", "intensity"],
                        values="count",
                        color="color",
                        color_discrete_map={
                            "Red": "#ff6666",
                            "Green": "#85e085",
                            "Blue": "#6666ff",
                        },
                    )
                    return fig

                image = load_image(uploaded_file)
                fig = color_distribution_sunburst(image)
                st.plotly_chart(fig, use_container_width=True, config=config)
            with st.expander("🕸 3D Color Space"):

                def plot_3d_color_space(data, skip_factor):
                    sample = data[::skip_factor, ::skip_factor].reshape(-1, 3)

                    normalized_colors = sample / 255.0

                    trace = go.Scatter3d(
                        x=sample[:, 0],
                        y=sample[:, 1],
                        z=sample[:, 2],
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=[
                                "rgb({},{},{})".format(r, g, b) for r, g, b in sample
                            ],
                            opacity=0.8,
                        ),
                    )
                    layout = go.Layout(
                        scene=dict(
                            xaxis=dict(title="Red"),
                            yaxis=dict(title="Green"),
                            zaxis=dict(title="Blue"),
                            camera=dict(eye=dict(x=1.25, y=1.25, z=1.25)),
                        ),
                        margin=dict(l=0, r=0, b=0, t=30),
                    )
                    fig = go.Figure(data=[trace], layout=layout)
                    return fig

                skip_factor = 8

                data = np.array(Image.open(uploaded_file))
                fig = plot_3d_color_space(data, skip_factor)
                st.plotly_chart(fig, use_container_width=True, config=config)
            with st.expander("𖦹 Pixel Density Polar"):

                def pixel_density_polar_plot(data):
                    image_data = np.array(Image.open(data))
                    hsv_data = mcolors.rgb_to_hsv(image_data / 255.0)
                    hue = hsv_data[:, :, 0].flatten()

                    hist, bins = np.histogram(hue, bins=360, range=(0, 1))
                    theta = np.linspace(0, 360, len(hist), endpoint=False)

                    fig = px.bar_polar(
                        r=hist,
                        theta=theta,
                        template="seaborn",
                        color_discrete_sequence=["red"],
                    )
                    fig.update_traces(marker=dict(line=dict(color="red", width=1)))
                    fig.update_layout()

                    return fig

                if uploaded_file is not None:
                    fig = pixel_density_polar_plot(uploaded_file)
                    st.plotly_chart(fig, use_container_width=True, config=config)
            with st.expander("ᨒ 3D Surface (Color Intensities)"):

                def surface_plot_image_intensity(data):
                    intensity = np.mean(data, axis=2)
                    sample_size = int(intensity.shape[0] * 0.35)
                    intensity_sample = intensity[:sample_size, :sample_size]
                    fig = go.Figure(
                        data=[go.Surface(z=intensity_sample, colorscale="Viridis")]
                    )
                    fig.update_layout(autosize=True)
                    return fig

                data = np.array(Image.open(uploaded_file))
                fig = surface_plot_image_intensity(data)

                st.plotly_chart(fig, use_container_width=True, config=config)
            with st.expander("🖌 Color Palette"):

                def extract_color_palette(image, num_colors=6):
                    image = image.resize((100, 100))
                    result = image.quantize(colors=num_colors)
                    palette = result.getpalette()
                    color_counts = result.getcolors()

                    colors = [palette[i * 3 : (i + 1) * 3] for i in range(num_colors)]
                    counts = [
                        count
                        for count, _ in sorted(
                            color_counts, reverse=True, key=lambda x: x[0]
                        )
                    ]

                    return colors, counts

                def plot_color_palette(colors, counts):
                    fig = go.Figure()
                    for i, (color, count) in enumerate(zip(colors, counts)):
                        hex_color = "#%02x%02x%02x" % tuple(color)
                        fig.add_trace(
                            go.Bar(
                                x=[1],
                                y=[hex_color],
                                orientation="h",
                                marker=dict(color=hex_color),
                                hoverinfo="text",
                                hovertext=f"<b>HEX:</b> {hex_color}<br><b>Count:</b> {count}",
                                name="",
                            )
                        )
                    fig.update_layout(
                        xaxis=dict(showticklabels=False),
                        yaxis=dict(showticklabels=True),
                        showlegend=False,
                        template="plotly_dark",
                        height=400,
                    )
                    return fig

                num_colors = st.slider("Number of Colors", 2, 10, 6)
                image = Image.open(uploaded_file).convert("RGB")
                colors, counts = extract_color_palette(image, num_colors)
                fig = plot_color_palette(colors, counts)
                st.plotly_chart(fig, use_container_width=True, config=config)
            st.session_state.reset_trigger = True

            col1, col2 = st.columns(2)
            with col1:
                st.button("🗑 Clear Chat History", on_click=clear_chat_history)
            with col2:
                clear = clear_exif_data(image)
                download_image(clear)


@st.experimental_dialog("How to use Exifa.net", width=1920)
def show_video(item):
    video_url = "https://www.youtube.com/watch?v=CS7rkWu7LNY"
    st.video(video_url, loop=False, autoplay=True, muted=False)


for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=icons[message["role"]]):
        st.write(message["content"])
        if message == st.session_state["messages"][0]:
            if st.button("How can I use Exifa?"):
                show_video("")
st.sidebar.caption(
    "Built by [Sahir Maharaj](https://www.linkedin.com/in/sahir-maharaj/). Like this? [Hire me!](https://topmate.io/sahirmaharaj/362667)"
)

linkedin = "https://raw.githubusercontent.com/sahirmaharaj/exifa/main/img/linkedin.gif"
topmate = "https://raw.githubusercontent.com/sahirmaharaj/exifa/main/img/topmate.gif"
email = "https://raw.githubusercontent.com/sahirmaharaj/exifa/main/img/email.gif"
newsletter = (
    "https://raw.githubusercontent.com/sahirmaharaj/exifa/main/img/newsletter.gif"
)
share = "https://raw.githubusercontent.com/sahirmaharaj/exifa/main/img/share.gif"

uptime = "https://uptime.betterstack.com/status-badges/v1/monitor/196o6.svg"

st.sidebar.caption(
    f"""
        <div style='display: flex; align-items: center;'>
            <a href = 'https://www.linkedin.com/in/sahir-maharaj/'><img src='{linkedin}' style='width: 35px; height: 35px; margin-right: 25px;'></a>
            <a href = 'https://topmate.io/sahirmaharaj/362667'><img src='{topmate}' style='width: 32px; height: 32px; margin-right: 25px;'></a>
            <a href = 'mailto:sahir@sahirmaharaj.com'><img src='{email}' style='width: 28px; height: 28px; margin-right: 25px;'></a>
            <a href = 'https://www.linkedin.com/build-relation/newsletter-follow?entityUrn=7163516439096733696'><img src='{newsletter}' style='width: 28px; height: 28px; margin-right: 25px;'></a>
            <a href = 'https://www.kaggle.com/sahirmaharajj'><img src='{share}' style='width: 28px; height: 28px; margin-right: 25px;'></a>
            
        </div>
        <br>
        <a href = 'https://exifa.betteruptime.com/'><img src='{uptime}'></a>
        &nbsp; <a href="https://www.producthunt.com/posts/exifa-net?embed=true&utm_source=badge-featured&utm_medium=badge&utm_souce=badge-exifa&#0045;net" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=474560&theme=dark" alt="Exifa&#0046;net - Your&#0032;AI&#0032;assistant&#0032;for&#0032;understanding&#0032;EXIF&#0032;data | Product Hunt" style="width: 125px; height: 27px;" width="125" height="27" /></a>
        
        """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_tokenizer():
    return AutoTokenizer.from_pretrained("huggyllama/llama-7b")


def get_num_tokens(prompt):
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(prompt)
    return len(tokens)


def generate_arctic_response_follow_up():

    follow_up_response = ""

    last_three_messages = st.session_state.messages[-3:]
    for message in last_three_messages:
        follow_up_response += "\n\n {}".format(message)
    prompt = [
        "Please generate one question based on the conversation thus far that the user might ask next. Ensure the question is short, less than 8 words, stays on the topic of EXIF and its importance and dangers, and is formatted with underscores instead of spaces, e.g., What_does_EXIF_mean? Conversation Info = {}. Please generate one question based on the conversation thus far that the user might ask next. Ensure the question is short, less than 8 words, stays on the topic of EXIF and its importance and dangers, and is formatted with underscores instead of spaces".format(
            follow_up_response
        )
    ]
    prompt.append("assistant\n")
    prompt_str = "\n".join(prompt)

    full_response = []
    for event in replicate.stream(
        "snowflake/snowflake-arctic-instruct",
        input={
            "prompt": prompt_str,
            "prompt_template": r"{prompt}",
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "stop_sequences": stop_sequences,
        },
    ):
        full_response.append(str(event).strip())
    complete_response = "".join(full_response)

    return complete_response


def generate_arctic_response():

    prompt = [base_prompt] if base_prompt else []
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            prompt.append("user\n" + dict_message["content"])
        else:
            prompt.append("assistant\n" + dict_message["content"])
    prompt.append("assistant\n")
    prompt_str = "\n".join(prompt)

    if get_num_tokens(prompt_str) >= 1000000:
        st.error("Conversation length too long. Please keep it under 1000000 tokens.")
        st.button(
            "🗑 Clear Chat History",
            on_click=clear_chat_history,
            key="clear_chat_history",
        )
        st.stop()
    for event in replicate.stream(
        "snowflake/snowflake-arctic-instruct",
        input={
            "prompt": prompt_str,
            "prompt_template": r"{prompt}",
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "stop_sequences": stop_sequences,
        },
    ):
        yield str(event)


def display_question():
    st.session_state.follow_up = True


if prompt := st.chat_input(disabled=not replicate_api):

    st.session_state.show_animation = False

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message(
        "user",
        avatar="https://raw.githubusercontent.com/sahirmaharaj/exifa/main/img/user.gif",
    ):
        st.write(prompt)
if st.session_state.follow_up:

    st.session_state.show_animation = False

    unique_key = "chat_input_" + str(hash("Snowflake Arctic is cool"))

    complete_question = generate_arctic_response_follow_up()
    formatted_question = complete_question.replace("_", " ").strip()

    st.session_state.messages.append({"role": "user", "content": formatted_question})
    with st.chat_message(
        "user",
        avatar="https://raw.githubusercontent.com/sahirmaharaj/exifa/main/img/user.gif",
    ):
        st.write(formatted_question)
    st.session_state.follow_up = False

    with st.chat_message(
        "assistant",
        avatar="https://raw.githubusercontent.com/sahirmaharaj/exifa/main/img/assistant.gif",
    ):
        response = generate_arctic_response()
        full_response = st.write_stream(response)
        message = {"role": "assistant", "content": full_response}

        st.session_state.messages.append(message)

        full_response_prompt = generate_arctic_response_follow_up()
        message_prompt = {"content": full_response_prompt}
        st.button(
            str(message_prompt["content"]).replace("_", " ").strip(),
            on_click=display_question,
        )
if st.session_state.messages[-1]["role"] != "assistant":

    st.session_state.show_animation = False

    with st.chat_message(
        "assistant",
        avatar="https://raw.githubusercontent.com/sahirmaharaj/exifa/main/img/assistant.gif",
    ):
        response = generate_arctic_response()
        full_response = st.write_stream(response)
        message = {"role": "assistant", "content": full_response}

        full_response_prompt = generate_arctic_response_follow_up()
        message_prompt = {"content": full_response_prompt}
        st.button(
            str(message_prompt["content"]).replace("_", " ").strip(),
            on_click=display_question,
        )

        st.session_state.messages.append(message)
if st.session_state.reset_trigger:

    unique_key = "chat_input_" + str(hash("Snowflake Arctic is cool"))

    complete_question = generate_arctic_response_follow_up()

    st.session_state.show_animation = False
if "has_snowed" not in st.session_state:

    st.snow()
    st.session_state["has_snowed"] = True
if st.session_state.show_animation:
    components.html(particles_js, height=370, scrolling=False)
