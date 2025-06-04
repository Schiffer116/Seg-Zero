from qwen_vl_utils import process_vision_info
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

from PIL import Image as PILImage
import numpy as np
from infer_multi_object import extract_bbox_points_think

def infer(reasoning_model, segmentation_model, processor, prompt, image):
    QUESTION_TEMPLATE = \
        "Please find \"{Question}\" with bboxs and points." \
        "Compare the difference between object(s) and find the most closely matched object(s)." \
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
        "Output the bbox(es) and point(s) inside the interested object(s) in JSON format." \
        "i.e., <think> thinking process here </think>" \
        "<answer>{Answer}</answer>"

    image = PILImage.open(image)
    image = image.convert("RGB")
    original_width, original_height = image.size
    resize_size = 840
    x_factor, y_factor = original_width/resize_size, original_height/resize_size

    messages = []
    message = [{
        "role": "user",
        "content": [
        {
            "type": "image",
            "image": image.resize((resize_size, resize_size), PILImage.BILINEAR)
        },
        {
            "type": "text",
            "text": QUESTION_TEMPLATE.format(
                Question=prompt.lower().strip("."),
                Answer="[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}, {\"bbox_2d\": [225,296,706,786], \"point_2d\": [302,410]}]"
            )
        }
    ]
    }]
    messages.append(message)

    # Preparation for inference
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

    #pdb.set_trace()
    image_inputs, video_inputs = process_vision_info(messages)
    #pdb.set_trace()
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = reasoning_model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print(output_text[0])
    # pdb.set_trace()
    bboxes, points, think = extract_bbox_points_think(output_text[0], x_factor, y_factor)
    print(points, len(points))

    print("Thinking process: ", think)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        mask_all = np.zeros((image.height, image.width), dtype=bool)
        segmentation_model.set_image(image)
        for bbox, point in zip(bboxes, points):
            masks, scores, _ = segmentation_model.predict(
                point_coords=[point],
                point_labels=[1],
                box=bbox
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            mask = masks[0].astype(bool)
            mask_all = np.logical_or(mask_all, mask)

    mask_overlay = np.zeros_like(image)
    mask_overlay[mask_all] = [255, 0, 0]

    blended = (0.6 * np.array(image) + 0.4 * mask_overlay).astype(np.uint8)

    masked_pil_image = PILImage.fromarray(blended)

    return think, masked_pil_image
