# helpers.py
import os
import base64
from PIL import Image
import streamlit as st

def add_big_sidebar_logo(width_px: int = 400):
    base_path = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(base_path, "helper.py", "pages", "Assets", "Quant_Vision_Logo.png")

    if not os.path.exists(logo_path):
        st.sidebar.error(f"Logo not found: {logo_path}")
        return

    # Get actual aspect ratio (height / width)
    w, h = Image.open(logo_path).size
    aspect = h / w if w else 0.6  # safe fallback

    with open(logo_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    # Calculate a reasonable container height for your target width
    target_h = int(width_px * aspect)

    st.markdown(
        f"""
        <style>
            /* Insert a responsive block above the multipage nav */
            [data-testid="stSidebarNav"]::before {{
                content: "";
                display: block;

                /* spacing around the logo block */
                margin: 16px 12px 20px 12px;

                /* make the block span the sidebar width minus side margins */
                width: calc(100% - 24px);

                /* height based on your requested width and the real aspect ratio */
                height: {target_h}px;

                /* show the image without cropping */
                background-image: url("data:image/png;base64,{b64}");
                background-repeat: no-repeat;
                background-position: center;

                /* scale the image to fit inside the box (no cropping) */
                background-size: contain;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

