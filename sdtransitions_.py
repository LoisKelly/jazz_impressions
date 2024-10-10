# This python file contains the code to set up the image to image stable diffusion model 
# and generate the transition frames
# It was developed with assitance from ChatGTP and my thesis supervisor.

import numpy as np
from datetime import datetime
from os import mkdir
from os.path import join
import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image
from safetensors.torch import load_file
from compel import Compel, ReturnedEmbeddingsType
import os
import cv2
from skimage import transform
from skimage import img_as_ubyte
import random

# https://scikit-image.org/docs/stable/auto_examples/transform/plot_swirl.html
def swirl_transform(image, strength, radius, center=None):
    height = width = 1024
    if center is None:
        center = (width // 2, height // 2)

    swirled_image = transform.swirl(np.array(image), center=center, strength=strength, radius=radius)
    swirled_image = img_as_ubyte(swirled_image)

    return swirled_image

# To make the transitions move smoothly
def smoothstep(t):
    return t * t * (3 - 2 * t)

# Using the swirl transformation function to set up the swirl transition function that swirls the image. 
def swirl_transition(image1, image2, total_steps=30, strength=5.0, radius=None):
    frames = [image1, image1, image1]
    height = width = 1024
    if radius is None:
        radius = min(width, height) / 2

    for step in range(total_steps - 6):
        # Smooth transition for swirl strength and alpha blending
        t = step / total_steps
        smooth_t = smoothstep(t)  # Apply smoothstep for smooth interpolation
        
        swirl_in_strength = strength * (1 - smooth_t)
        swirl_out_strength = -strength * smooth_t
        
        frame1 = swirl_transform(image1, swirl_in_strength, radius)
        frame2 = swirl_transform(image2, swirl_out_strength, radius)
        
        alpha = smooth_t
        blended_frame = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
        frames.append(Image.fromarray(blended_frame))

    frames.append(image2)
    frames.append(image2)
    frames.append(image2)
    return frames

# Function to generate the transition frames. 
def generate_frames(input_images_folder, lora1_path, lora2_path, model_id, steps_per_image=20, frame_rate=15):

# Loading the image to image stable diffution model and initialising the pipeline
    pipeline_img = AutoPipelineForImage2Image.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16"
    ).to("gpu")
    pipeline_img.enable_model_cpu_offload()
    pipeline_img.enable_xformers_memory_efficient_attention()
    
    generator = torch.manual_seed(31)
    tokenizer = pipeline_img.tokenizer
    text_encoder = pipeline_img.text_encoder

    compel = Compel(
        tokenizer=[pipeline_img.tokenizer, pipeline_img.tokenizer_2],
        text_encoder=[pipeline_img.text_encoder, pipeline_img.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True]
    )
# Adding the diffusion prompts to add interest to the video
    prompt1 = "paint swirls in the air, modernist, vibrant colours"
    conditioning1, pooled1 = compel.build_conditioning_tensor(prompt1)
    conditioning1 = conditioning1.unsqueeze(0)
    prompt2 = "music blooms from the eyes, the images swirl, the colours dance"
    conditioning2, pooled2 = compel.build_conditioning_tensor(prompt2) 
    conditioning2 = conditioning2.unsqueeze(0)

# Creating the output directory using the timestamp
    frames = []
    timestampStr = datetime.now().strftime("%d-%b-%Y-%H-%M-%S")
    mkdir(timestampStr)
    dims = 1024
    paths = sorted([img for img in os.listdir(input_images_folder) if img.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))])
    current_image = Image.open(os.path.join(input_images_folder, paths[random.int]))
    total_frames_needed = frame_rate * 5  # 75 frames for 5 seconds at 15 fps
    alpha_steps = torch.linspace(0, 1, total_frames_needed)
    strength_steps = torch.concat((torch.linspace(0.1, 1, total_frames_needed // 2),
                                    torch.linspace(1, 0.1, total_frames_needed // 2)))
    
# Looping through all of the images to create transitions between each until total frames reached
    for image_ptr, next_image_path in enumerate(paths[1:]):
        next_image = Image.open(os.path.join(input_images_folder, next_image_path))
        transition_frames = swirl_transition(current_image, next_image, steps_per_image)
        
        for frame_ptr, frame in enumerate(transition_frames):
            if len(frames) >= total_frames_needed:
                break

            with torch.no_grad():
                alpha_ptr = (image_ptr * steps_per_image) + frame_ptr
                alpha = alpha_steps[alpha_ptr]
                strength = float(strength_steps[frame_ptr])
                interpolated_conditioning = (1 - alpha) * conditioning1 + alpha * conditioning2
                interpolated_pool = (1 - alpha) * pooled1 + alpha * pooled2
                image = pipeline_img(prompt_embeds=interpolated_conditioning, generator=torch.manual_seed(alpha_ptr),
                                    pooled_prompt_embeds=interpolated_pool, num_inference_steps=50, strength=strength, guidance_scale=16,
                                    image=frame).images[0]
                frames.append(image)
                image.save(join(timestampStr, f"generated_image_{alpha:.5f}.jpg"))
                torch.cuda.empty_cache()

        if len(frames) >= total_frames_needed:
            break
        
        current_image = frames[-1]
    
    # Save frames as video
    frame_size = (frames[0].width, frames[0].height)
    out = cv2.VideoWriter(f'{timestampStr}/output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, frame_size)

    for frame in frames:
        frame_bgr = np.array(frame.convert('RGB'))[:, :, ::-1]  # Convert to BGR for OpenCV
        out.write(frame_bgr)
    
    out.release()
    cv2.destroyAllWindows()

    return frames
