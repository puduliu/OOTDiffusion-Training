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

# batch_size 最大可以设置为10, 设置为11就OOM了
CUDA_VISIBLE_DEVICES=0
python train_vton_log.py \
  --model_type "hd" \
  --train_epochs 50 \
  --batch_size 10 \
  --learning_rate 5e-5 \
  --conditioning_dropout_prob 0.1 \
  --mixed_precision "fp16" \
  --img_height 512 \
  --img_width 384 \
  --num_workers 4 \
  --dataset_dir "/home/zyserver/work/lpd/download/VITON-HD/zalando-hd-resized" \
  --vit_path "../checkpoints/clip-vit-large-patch14" \
  --vae_path "../checkpoints/stable-diffusion-v1-5/vae" \
  --unet_path "../checkpoints/stable-diffusion-v1-5/unet" \
  --tokenizer_path "../checkpoints/stable-diffusion-v1-5/tokenizer" \
  --text_encoder_path "../checkpoints/stable-diffusion-v1-5/text_encoder" \
  --scheduler_path "../checkpoints/stable-diffusion-v1-5/scheduler/scheduler_config.json" \
  --first_epoch 0 \




# TODO 训练的时候是不是得有个全新的unet_path,或者用预训练模型
  # --unet_path "/home/zyserver/work/lpd/OOTDiffusion-Training/checkpoints/stable-diffusion-v1-5/ootd_hd/unet_garm" \
# CUDA_VISIBLE_DEVICES=0 python train_ootd_pl_ip_adapter.py \
#   --model_type "hd" \
#   --train_epochs 50 \
#   --batch_size 2 \
#   --learning_rate 1e-5 \
#   --conditioning_dropout_prob 0.1 \
#   --mixed_precision "fp16" \
#   --img_height 512 \
#   --img_width 384 \
#   --num_workers 4 \
#   --dataset_dir "/home/zyserver/work/lpd/download/VITON-HD/zalando-hd-resized" \
#   --vit_path "../checkpoints/clip-vit-large-patch14" \
#   --vae_path "../checkpoints/stable-diffusion-v1-5/vae" \
#   --unet_path "../checkpoints/stable-diffusion-v1-5/unet" \
#   --tokenizer_path "../checkpoints/stable-diffusion-v1-5/tokenizer" \
#   --text_encoder_path "../checkpoints/stable-diffusion-v1-5/text_encoder" \
#   --scheduler_path "../checkpoints/stable-diffusion-v1-5/scheduler/scheduler_config.json" \
#   --first_epoch 0 \

# batch_size = 7最大了, 能不能确认下为什么 普通pytorch训练batch size可以设置为10，差别这么大
# CUDA_VISIBLE_DEVICES=0 python train_vton_pl.py \
#   --model_type "hd" \
#   --train_epochs 50 \
#   --batch_size 7 \
#   --learning_rate 5e-5 \
#   --conditioning_dropout_prob 0.1 \
#   --mixed_precision "fp16" \
#   --img_height 512 \
#   --img_width 384 \
#   --num_workers 14 \
#   --dataset_dir "/home/zyserver/work/lpd/download/VITON-HD/zalando-hd-resized" \
#   --vit_path "../checkpoints/clip-vit-large-patch14" \
#   --vae_path "../checkpoints/stable-diffusion-v1-5/vae" \
#   --unet_path "../checkpoints/stable-diffusion-v1-5/unet" \
#   --tokenizer_path "../checkpoints/stable-diffusion-v1-5/tokenizer" \
#   --text_encoder_path "../checkpoints/stable-diffusion-v1-5/text_encoder" \
#   --scheduler_path "../checkpoints/stable-diffusion-v1-5/scheduler/scheduler_config.json" \
#   --first_epoch 0 \