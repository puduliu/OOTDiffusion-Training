# CUDA_VISIBLE_DEVICES=0
# python train_ootd.py \
#   --model_type "hd" \
#   --train_epochs 50 \
#   --batch_size 24 \
#   --learning_rate 1e-5 \
#   --conditioning_dropout_prob 0.1 \
#   --mixed_precision "fp16" \
#   --img_height 512 \
#   --img_width 384 \
#   --num_workers 14 \
#   --dataset_dir "../DressCode" \
#   --vit_path "../models/clip-vit-large-patch14" \
#   --vae_path "../models/stable-diffusion-v1-5/vae" \
#   --unet_path "../models/stable-diffusion-v1-5/unet" \
#   --tokenizer_path "../models/stable-diffusion-v1-5/tokenizer" \
#   --text_encoder_path "../models/stable-diffusion-v1-5/text_encoder" \
#   --scheduler_path "../models/stable-diffusion-v1-5/scheduler/scheduler_config.json" \
#   --first_epoch 0 \

# CUDA_VISIBLE_DEVICES=4,0 python train_ootd_pl.py \
#   --model_type "hd" \
#   --train_epochs 50 \
#   --batch_size 1 \
#   --learning_rate 1e-5 \
#   --conditioning_dropout_prob 0.1 \
#   --mixed_precision "fp16" \
#   --img_height 512 \
#   --img_width 384 \
#   --num_workers 4 \
#   --dataset_dir "/media/jqzhu/941A7DD31A7DB33A/lpd/download/VITON-HD" \
#   --vit_path "../checkpoints/clip-vit-large-patch14" \
#   --vae_path "../checkpoints/stable-diffusion-v1-5/vae" \
#   --unet_path "../checkpoints/stable-diffusion-v1-5/unet" \
#   --tokenizer_path "../checkpoints/stable-diffusion-v1-5/tokenizer" \
#   --text_encoder_path "../checkpoints/stable-diffusion-v1-5/text_encoder" \
#   --scheduler_path "../checkpoints/stable-diffusion-v1-5/scheduler/scheduler_config.json" \
#   --first_epoch 0 \

CUDA_VISIBLE_DEVICES=3,5 python train_ootd_pl.py \
  --model_type "hd" \
  --train_epochs 50 \
  --batch_size 1 \
  --learning_rate 1e-5 \
  --conditioning_dropout_prob 0.1 \
  --mixed_precision "fp16" \
  --img_height 512 \
  --img_width 384 \
  --num_workers 4 \
  --dataset_dir "/media/jqzhu/941A7DD31A7DB33A/lpd/download/VITON-HD" \
  --vit_path "../checkpoints/clip-vit-large-patch14" \
  --vae_path "../checkpoints/stable-diffusion-v1-5/vae" \
  --unet_path "/media/jqzhu/941A7DD31A7DB33A/lpd/OOTDiffusion-Training/run/checkpoints/unet_vton" \
  --tokenizer_path "../checkpoints/stable-diffusion-v1-5/tokenizer" \
  --text_encoder_path "../checkpoints/stable-diffusion-v1-5/text_encoder" \
  --scheduler_path "../checkpoints/stable-diffusion-v1-5/scheduler/scheduler_config.json" \
  --first_epoch 0 \