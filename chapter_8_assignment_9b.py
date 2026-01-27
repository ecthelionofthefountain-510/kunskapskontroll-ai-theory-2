# =====Imports =====
import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
from keras.applications import ResNet50
from keras.applications.resnet50 import decode_predictions as resnet_decode
from keras.applications.resnet50 import preprocess_input as resnet_preprocess
from PIL import Image



# ====== Model specs ======
@dataclass(frozen=True)
class ModelSpec:
    name: str
    image_size: int
    preprocess: callable
    decode: callable
    build: callable


def _mobilenetv2_spec() -> ModelSpec:
    return ModelSpec(
        name="MobileNetV2 (ImageNet)",
        image_size=224,
        preprocess=preprocess_input,
        decode=decode_predictions,
        build=lambda: MobileNetV2(weights="imagenet"),
    )


def _resnet50_spec() -> ModelSpec:
    return ModelSpec(
        name="ResNet50 (ImageNet)",
        image_size=224,
        preprocess=resnet_preprocess,
        decode=resnet_decode,
        build=lambda: ResNet50(weights="imagenet"),
    )

# =====Available models =====
MODEL_SPECS = {
    "mobilenetv2": _mobilenetv2_spec(),
    "resnet50": _resnet50_spec(),
}


@st.cache_resource
def load_model(model_key: str):
    spec = MODEL_SPECS[model_key]
    return spec.build()


def prepare_image(img: Image.Image, image_size: int, preprocess) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize((image_size, image_size))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess(arr)
    return arr


def main() -> None:

    # ===== Main =====
    st.set_page_config(page_title="Assignment 9 – Image classification", layout="centered")

    st.title("Assignment 9 – Pre trained models for image classification")

    # ====== Sidebar =====
    st.sidebar.header("Settings")

    model_key = st.sidebar.selectbox(
        "Choose model:", options=list(MODEL_SPECS.keys()),
        format_func=lambda k: MODEL_SPECS[k].name,
        index=0,
    )

    st.sidebar.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.sidebar.divider()

    uploaded = st.sidebar.file_uploader(
        "Choose a picture:",
    )

    if uploaded is None:
        st.info("Choose an image file to get started.")
        return

    try:
        # ====== Read image  =====
        data = uploaded.getvalue() 
        if not data:
            st.error("The uploaded file appears to be empty.")
            return

        img = Image.open(io.BytesIO(data))
        st.image(img, caption=uploaded.name, use_container_width=True)

        # ===== Prediction ======
        spec = MODEL_SPECS[model_key]
        with st.spinner(f"Loading model: {spec.name} …"):
            model = load_model(model_key)

        with st.spinner("Running prediction …"):
            x = prepare_image(img, spec.image_size, spec.preprocess)
            preds = model(x, training=False).numpy()
            top5 = spec.decode(preds, top=5)[0]
    except Exception as e:
        st.error("Something went wrong while reading the image or running the model.")
        st.exception(e)
        return

    # ===== Resultat =====
    st.subheader("Prediction")

    best_class_id, best_label, best_prob = top5[0]
    st.success(f"Highest probability: {best_label} ({best_prob:.1%})")

    rows = [{"Label": label, "Probability": float(prob)} for _, label, prob in top5]
    df = pd.DataFrame(rows)

    st.caption("Top-5 (tabell)")
    table_df = df.copy()
    table_df["Probability"] = (table_df["Probability"] * 100).round(1).astype(str) + "%"
    st.dataframe(table_df, hide_index=True, use_container_width=True)


# ====== Entry point ======
if __name__ == "__main__":
    main()

