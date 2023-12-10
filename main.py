from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet.controlnet_pipeline import FastComposerControlNetPipeline
from fastcomposer.utils import parse_args
from diffusers.utils import load_image
import torch 
from controlnet.pose_utils import OpenposeDetector

import cv2
from PIL import Image
import numpy as np
import os 
import pdb
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def load_pipeline(args, device):
    from fastcomposer.model import FastComposerModel
    from fastcomposer.transforms import PadToSquare
    from torchvision import transforms as T
    from transformers import CLIPTokenizer
    from collections import OrderedDict

    model = FastComposerModel.from_pretrained(args)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer.add_tokens(["img"], special_tokens=True)
    image_token_id = tokenizer.convert_tokens_to_ids("img")

    controlnet_name = 'lllyasviel/sd-controlnet-openpose' if args.use_poses else 'lllyasviel/sd-controlnet-canny'
    controlnet = ControlNetModel.from_pretrained(controlnet_name, torch_dtype=torch.float16)

    pipe = FastComposerControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path, controlnet=controlnet, torch_dtype=weight_dtype
    ).to(device)
    pipe.safety_checker = None 

    state_dict = torch.load(args.finetuned_model_path, map_location="cpu")
    new_state_dict = dict()

    for key, val in state_dict.items():
        if key.startswith("vae.encoder.") or key.startswith("vae.decoder."):
            key = key.replace("query", "to_q").replace("key", 
                "to_k").replace("value", "to_v").replace("proj_attn", "to_out.0")
            
        new_state_dict[key] = val

    model.load_state_dict(new_state_dict, strict=True)
    model = model.to(dtype=weight_dtype, device=device)

    pipe.object_transforms = torch.nn.Sequential(
        OrderedDict(
            [
                ("pad_to_square", PadToSquare(fill=0, padding_mode="constant")),
                (
                    "resize",
                    T.Resize(
                        (args.object_resolution, args.object_resolution),
                        interpolation=T.InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                ),
                ("convert_to_float", T.ConvertImageDtype(torch.float32)),
            ]
        )
    )
    pipe.unet = model.unet
    pipe.text_encoder = model.text_encoder
    pipe.postfuse_module = model.postfuse_module
    pipe.image_encoder = model.image_encoder
    pipe.image_token_id = image_token_id
    pipe.special_tokenizer = tokenizer

    del model
    return pipe

def generate_poses_image(image, CACHE_DIR):
    pose_image, _ = OpenposeDetector()(image)
    pose_image = Image.fromarray(pose_image)
    pose_image.save(os.path.join(CACHE_DIR,"pose_image.png"))
    return pose_image

def generate_canny_image(image, CACHE_DIR, low_threshold=100, high_threshold=200):
    canny_image = cv2.Canny(image, low_threshold, high_threshold)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    canny_image = Image.fromarray(canny_image)
    canny_image.save(os.path.join(CACHE_DIR,"canny_image.png"))
    return canny_image

def main_fastcomposer_controlnet():
    args = parse_args()
    CACHE_DIR = args.output_images_dir
    os.makedirs(CACHE_DIR, exist_ok=True)
    image = Image.open(args.control_image_path)
    image.save(os.path.join(CACHE_DIR,"control_image_original.png"))
    image = np.array(image)

    condition_image = generate_poses_image(image, CACHE_DIR) if args.use_poses else generate_canny_image(image, CACHE_DIR)

    """
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    """

    reference_subject_images = [Image.open(args.reference_image_path)] # [Image.open("taylor.jpg")]

    pipe = load_pipeline(args, "cuda")

    prompt = ["a portrait of a woman img smiling, best quality, extremely detailed"]
    # prompt = [t + prompt for t in ["Sandra Oh", "Kim Kardashian", "rihanna", "taylor swift"]]
    generator = [torch.Generator(device="cpu").manual_seed(i) for i in range(len(prompt)*args.num_images_per_prompt)]

    output = pipe(
        prompt,
        condition_image,
        negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * len(prompt),
        generator=generator,
        num_inference_steps=50,
        height=512,
        width=512,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=args.num_images_per_prompt,
        alpha_=args.alpha,
        reference_subject_images=reference_subject_images,
        # control_guidance_start=0.0,
        # control_guidance_end=0.8,
    )

    grid_image = image_grid(output.images, 1, 1)
    grid_image.save(os.path.join(CACHE_DIR,"output_image.png"))


def main_sd_controlnet():
    args = parse_args()
    CACHE_DIR = "controlnet_cache"
    image = load_image(
        "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
    )
    image.save(os.path.join(CACHE_DIR,"original_image.png"))
    image = np.array(image)

    condition_image = generate_poses_image(image, CACHE_DIR) if args.use_poses else generate_canny_image()

    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )

    pipe = pipe.to("cuda")

    prompt = ", best quality, extremely detailed"
    prompt = [t + prompt for t in ["Sandra Oh", "Kim Kardashian", "rihanna", "taylor swift"]]
    generator = [torch.Generator(device="cpu").manual_seed(2) for i in range(len(prompt))]

    output = pipe(
        prompt,
        condition_image,
        negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * len(prompt),
        generator=generator,
        num_inference_steps=50,
    )

    grid_image = image_grid(output.images, 2, 2)

    grid_image.save(os.path.join(CACHE_DIR,"output_image.png"))


if __name__ == "__main__":
    main_fastcomposer_controlnet()