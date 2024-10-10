# This python file contains the code to set up the text to image stable diffusion model 
# and generate the belended image frames
# It was developed with assitance from ChatGTP and my thesis supervisor.

import numpy as np
from datetime import datetime
from os import mkdir, makedirs
from os.path import join, isdir
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
from safetensors.torch import load_file
from compel import Compel, ReturnedEmbeddingsType
import cv2
import re

def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def generate_interpolated_images(model_id, adapter_names, prompts, lora1_path, lora2_path, output_dir=None):
    # weight_1: float, weight_2: float, 
    num_interpolations = 100
    num_inference_steps = 8
    # Load the pipeline
    pipeline = AutoPipelineForText2Image.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")

    seed = 42
    generator = torch.manual_seed(seed)

    # Load in the LoRA weights
    pipeline.load_lora_weights(lora1_path, adapter_name=adapter_names[0])
    pipeline.load_lora_weights(lora2_path, adapter_name=adapter_names[1])

    # Set up Compel
    compel = Compel(
        tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
        text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True]
    )

    # Create output directory 
    if output_dir is None:
        timestampStr = datetime.now().strftime("%d-%b-%Y-%H-%M-%S")
        output_dir = timestampStr
        if not isdir(output_dir):
            makedirs(output_dir)

    frames = []
    image_paths = []

    with torch.no_grad():
        prompt1, prompt2 = prompts
        pipeline.set_adapters(adapter_names, adapter_weights=[0.5, 0.5])

        for i, alpha in enumerate(torch.linspace(0, 1, num_interpolations), 1):
            conditioning1, pooled1 = compel.build_conditioning_tensor(prompt1)
            conditioning1 = conditioning1.unsqueeze(0)
            conditioning2, pooled2 = compel.build_conditioning_tensor(prompt2)
            conditioning2 = conditioning2.unsqueeze(0)
            interpolated_conditioning = (1 - alpha) * conditioning1 + alpha * conditioning2
            interpolated_pool = (1 - alpha) * pooled1 + alpha * pooled2
            image = pipeline(
                prompt_embeds=interpolated_conditioning,
                pooled_prompt_embeds=interpolated_pool,
                num_inference_steps=num_inference_steps,
                generator=generator
            ).images[0]           
            filename = f"image_{i:03d}.jpg"  
            image_filename = join(output_dir, filename)
            
            if not isdir(output_dir):
                makedirs(output_dir)

            # Save the image
            image.save(image_filename)
            image_paths.append(image_filename)
            
            frames.append(image)
            torch.cuda.empty_cache()

    return output_dir, image_paths

