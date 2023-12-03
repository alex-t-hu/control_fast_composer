CAPTION="a man <|image|> and a man <|image|> are reading book together"
DEMO_NAME="newton_einstein"
OUTPUT_DIR="control_output9/${DEMO_NAME}"

export PYTHONPATH="/home/alex/devel/control_fast_composer/fastcomposer:$PYTHONPATH"
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --mixed_precision=fp16 \
    fastcomposer/control_inference.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --finetuned_model_path model/fastcomposer \
    --test_reference_folder data/${DEMO_NAME} \
    --test_caption "${CAPTION}" \
    --output_dir ${OUTPUT_DIR} \
    --mixed_precision fp16 \
    --image_encoder_type clip \
    --image_encoder_name_or_path openai/clip-vit-large-patch14 \
    --num_image_tokens 1 \
    --max_num_objects 2 \
    --object_resolution 224 \
    --generate_height 512 \
    --generate_width 512 \
    --num_images_per_prompt 1 \
    --num_rows 1 \
    --seed 42 \
    --guidance_scale 9 \
    --inference_steps 50 \
    --start_merge_step 10 \
    --control_reference_image /home/alex/devel/control_fast_composer/fastcomposer/gregilya.jpg \
    --scale 2.0 \
    --no_object_augmentation
   
cp scripts/run_control_inference.sh ${OUTPUT_DIR}