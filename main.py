from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet.controlnet_pipeline import FastComposerControlNetPipeline
from fastcomposer.utils import parse_args
from diffusers.utils import load_image
import torch 
from controlnet.pose_utils import OpenposeDetector
from similarity_score import similarity_score, similarity_score_double
import scipy
import pandas
import cv2
from PIL import Image
from mtcnn import MTCNN
import numpy as np
from keras_facenet import FaceNet
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
    print(args.alpha)
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

    # prompt = ["a portrait of a woman img smiling, best quality, extremely detailed"]
    prompt = ["a picture of a person img dancing, best quality, extremely detailed"]
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

def eval_in_bulk():
    args = parse_args()
    ref_images_dir = './celebA/ref'
    ref_images = []
    for _, _, files in os.walk(ref_images_dir):
        ref_images.extend(files)
    
    ref_images = sorted(ref_images)
    
    CACHE_DIR = f'./celebA/{args.output_images_dir}'
    os.makedirs(CACHE_DIR, exist_ok=True)
    image = Image.open(args.control_image_path)
    image.save(os.path.join(CACHE_DIR,"control_image_original.png"))
    image = np.array(image)

    condition_image = generate_poses_image(image, CACHE_DIR) if args.use_poses else generate_canny_image(image, CACHE_DIR)

    prompt = ["a picture of a person img dancing, best quality, extremely detailed"]
    pipe = load_pipeline(args, "cuda")
    generator = [torch.Generator(device="cpu").manual_seed(i) for i in range(len(prompt)*args.num_images_per_prompt)]
    # run on all of celeb-A
    for img in ref_images:
        reference_subject_images = [Image.open(f'./celebA/ref/{img}')]
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
        )

        grid_image = image_grid(output.images, 1, 1)
        grid_image.save(os.path.join(CACHE_DIR,img))
    return str(args.alpha)


def eval_in_bulk_multiimage():
    args = parse_args()
    ref_images = [
        ('Barbara Liskov', './twoperson_data/barbara.jpg'),
        ('Albert Einstein', './twoperson_data/einstein.jpg'),
        ('Elon Musk', './twoperson_data/elon.jpg'),
        ('Fei-fei Li', './twoperson_data/feifeili.jpg'),
        ('Geoffery Hinton', './twoperson_data/hinton.jpg'),
        ('Isaac Newton', './twoperson_data/newton.jpg'),
        ('Song Han','./twoperson_data/songhan.jpg'),
        ('Yann LeCun','./twoperson_data/yannlecun.jpg'),
        ('Yoshua Bengio','./twoperson_data/yoshua.jpg')]
    
    CACHE_DIR = f'./twoperson_outs/{args.output_images_dir}'
    os.makedirs(CACHE_DIR, exist_ok=True)
    image = Image.open(args.control_image_path)
    image.save(os.path.join(CACHE_DIR,"control_image_original.png"))
    image = np.array(image)

    condition_image = generate_poses_image(image, CACHE_DIR) if args.use_poses else generate_canny_image(image, CACHE_DIR)

    pipe = load_pipeline(args, "cuda")
    # run on all of celeb-A
    for i in range(9):
        for j in range(i):
            person1, file1 = ref_images[i]
            person2, file2 = ref_images[j]
            prompt = [f"a picture of {person1} and {person2} standing next to each other, best quality, extremely detailed"]
            generator = [torch.Generator(device="cpu").manual_seed(i) for i in range(len(prompt)*args.num_images_per_prompt)]
            reference_subject_images = [Image.open(file1), Image.open(file2)]
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
            )

            grid_image = image_grid(output.images, 1, 1)
            grid_image.save(os.path.join(CACHE_DIR, f'{person1} and {person2}.png'))
    return str(args.alpha)

def compute_eval_score(ref_images_dir, out_images_dir, ):
    ref_images = []
    for _, _, files in os.walk(ref_images_dir):
        ref_images.extend(files)
    evals = []

    detector = MTCNN() # create the detector, using default weights
    model = FaceNet()
    for image in ref_images:
        output = os.path.join(out_images_dir, f'{image}')
        ref = os.path.join(ref_images_dir, image)
        sim = similarity_score(detector, model, output, ref)
        evals.append(sim)
    
    return np.array(evals)

def compute_eval_multiimage(out_images_dir):
    ref_images = [
        ('Barbara Liskov', './twoperson_data/barbara.jpg'),
        ('Albert Einstein', './twoperson_data/einstein.jpg'),
        ('Elon Musk', './twoperson_data/elon.jpg'),
        ('Fei-fei Li', './twoperson_data/feifeili.jpg'),
        ('Geoffery Hinton', './twoperson_data/hinton.jpg'),
        ('Isaac Newton', './twoperson_data/newton.jpg'),
        ('Song Han','./twoperson_data/songhan.jpg'),
        ('Yann LeCun','./twoperson_data/yannlecun.jpg'),
        ('Yoshua Bengio','./twoperson_data/yoshua.jpg')]
    
    CACHE_DIR = f'./twoperson_outs/{out_images_dir}'

    detector = MTCNN() # create the detector, using default weights
    model = FaceNet()
    evals = []

    for i in range(9):
        for j in range(i):
            person1, file1 = ref_images[i]
            person2, file2 = ref_images[j]

            output = os.path.join(CACHE_DIR, f'{person1} and {person2}.png')
            sim = similarity_score_double(detector, model, output, file1, file2)
            evals.append(sim)
    return np.array(evals)

if __name__ == "__main__":
    # main_fastcomposer_controlnet()
    # alphaval = eval_in_bulk_multiimage()
    
    to_print = []
    
    to_print.append('L2')
    for alphaval in ['0.0', '0.1', '0.2', 
              '0.3', '0.4', '0.5', 
              '0.6', '0.7', '0.8', 
              '0.9', '1.0']:
        to_print.append('alpha')
        to_print.append(alphaval)
        evals = compute_eval_multiimage( f'poses_alpha{alphaval}')
        to_print.append('average')
        to_print.append(str(np.mean(evals)))
        to_print.append('standard error of mean')
        to_print.append(str(scipy.stats.sem(evals)))

        with open(f'./celebA/poses_alpha{alphaval}/eval_L2.txt', 'w') as f:
            f.write('\n'.join(to_print))
        
        df = pandas.DataFrame(evals)
        df.to_csv(f'./celebA/poses_alpha{alphaval}/eval_values_L2.csv')
    print('\n'.join(to_print))