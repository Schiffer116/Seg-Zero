import torch
import streamlit as st
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from infer_function import infer

@st._dg_singleton
def load_model():
    reasoning_model_path = "pretrained_models/VisionReasoner-7B"
    segmentation_model_path = "facebook/sam2-hiera-large"

    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        reasoning_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    segmentation_model = SAM2ImagePredictor.from_pretrained(segmentation_model_path)

    reasoning_model.eval()

    # default processer
    processor = AutoProcessor.from_pretrained(reasoning_model_path, padding_side="left")
    return reasoning_model, segmentation_model, processor


reasoning_model, segmentation_model, processor = load_model()
st.set_page_config(layout="wide")

# Layout: Split screen
left_col, right_col = st.columns([1, 1])

# Left Side – Inputs
with left_col:
    st.header("Image and Prompt Input")
    input_mode = st.radio("Choose Image Input Mode:", ["Upload", "Camera"])

    image = uploaded_file = None
    if input_mode == "Upload":
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
    else:
        image = st.camera_input("Take a photo")

    # Prompt input
    prompt = st.text_area("Write your prompt here:")


with right_col:
    if (uploaded_file or image) and prompt:
        thought, output = infer(reasoning_model, segmentation_model, processor, prompt, image)
        st.image(output, use_container_width=True)

        st.subheader("Prompt Response:")
        st.write(f"Thinking: {thought}")
    else:
        st.info("Please provide both an image and a prompt.")
