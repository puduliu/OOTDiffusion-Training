import sys
import os
from shutil import copyfile, rmtree
import argparse
from utils.dataset import DressCodeDataLoader, DressCodeDataset, VITONDataLoader, VITONDataset
sys.path.append(r'../ootd')
# models import
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from diffusers.optimization import get_scheduler 
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers import UniPCMultistepScheduler, PNDMScheduler
from pipelines_ootd.unet_vton_2d_condition import UNetVton2DConditionModel
from pipelines_ootd.unet_garm_2d_condition import UNetGarm2DConditionModel

# train tools import
from tqdm import tqdm
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
import math
from pathlib import Path
import logging

import pytorch_lightning as pl
from VTONModel import VTONModel
from SimpleLiteUNet import SimpleLiteUNet

#-----args-----
def get_args():
    
    parser = argparse.ArgumentParser()
    
    # training configs
    parser.add_argument("--model_type", type=str, default='hd', help="hd or dc.")
    parser.add_argument("--train_epochs", type=int, default=200)
    parser.add_argument("--first_epoch", type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--learning_rate",type=float,default=5e-5)
    parser.add_argument("--conditioning_dropout_prob",type=float,default=0.1,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["normal","fp16", "bf16"],
        help=(
            "Whether to use mixed precision."
        ),
    )
    parser.add_argument("--checkpoints_total_limit", type=int, default=4)
    
    # dataset configs
    parser.add_argument('--img_height', type=int, default=512)
    parser.add_argument('--img_width', type=int, default=384)
    parser
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--semantic_nc', type=int, default=13, help='# of human-parsing map classes')

    # paths
    parser.add_argument('--dataset_dir', type=str, default='../VITON-HD')
    parser.add_argument("--vit_path", type=str, default="../checkpoints/clip-vit-large-patch14")
    parser.add_argument("--vae_path", type=str, default="../checkpoints/stable-diffusion-v1-5/vae")
    # parser.add_argument("--unet_path", type=str, default="../checkpoints/stable-diffusion-v1-5/unet")
    parser.add_argument("--unet_path", type=str, default="../checkpoints/unet_vton")
    parser.add_argument("--tokenizer_path", type=str, default="../checkpoints/stable-diffusion-v1-5/tokenizer")
    parser.add_argument("--text_encoder_path", type=str, default="../checkpoints/stable-diffusion-v1-5/text_encoder")
    parser.add_argument("--scheduler_path", type=str, default="../checkpoints/stable-diffusion-v1-5/scheduler/scheduler_config.json")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to a checkpoint to resume training.")
    
    
    # lr scheduler configs
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    
    args, unknown = parser.parse_known_args()
    return args

args = get_args()

test_dataset = VITONDataset(args, "test") # TODO
test_loader = VITONDataLoader(args, test_dataset)
train_dataset = VITONDataset(args, "train")
train_dataloader =VITONDataLoader(args, train_dataset)
train_dataloader = train_dataloader.data_loader
print("----------------------------------------args.batch_size = ", args.batch_size)

#-----load models-----
vae = AutoencoderKL.from_pretrained(args.vae_path)
# unet_garm = SimpleLiteUNet()
# unet_garm = UNetGarm2DConditionModel.from_pretrained(args.unet_path,use_safetensors=True)
# unet_vton = UNetVton2DConditionModel.from_pretrained(args.unet_path,use_safetensors=True)

# TODO 本来训练都使用 ../checkpoints/stable-diffusion-v1-5/unet  原始的sd1.5的unet.
unet_garm = UNetGarm2DConditionModel.from_pretrained("/home/zyserver/work/lpd/OOTDiffusion-Training/run/checkpoints/unet_garm",use_safetensors=True)
unet_vton = UNetVton2DConditionModel.from_pretrained("/home/zyserver/work/lpd/OOTDiffusion-Training/run/checkpoints/unet_vton",use_safetensors=True)

noise_scheduler = PNDMScheduler.from_pretrained(args.scheduler_path)
auto_processor = AutoProcessor.from_pretrained(args.vit_path)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.vit_path)
tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_path)
text_encoder = CLIPTextModel.from_pretrained(args.text_encoder_path)

print("===============================Now using in_channels 1111:", unet_vton.conv_in.in_channels)

#-----models configs-----
# unet_vton(denoising UNet)in_channels=4 --> in_channels=8
if unet_vton.conv_in.in_channels == 4:
    with torch.no_grad():
        new_in_channels = 8
        # create a new conv layer with 8 input channels
        conv_new = torch.nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=unet_vton.conv_in.out_channels,
            kernel_size=3,
            padding=1,
        )
        torch.nn.init.kaiming_normal_(conv_new.weight) 
        conv_new.weight.data = conv_new.weight.data * 0.  
        conv_new.weight.data[:, :4] = unet_vton.conv_in.weight.data  
        conv_new.bias.data = unet_vton.conv_in.bias.data  
        unet_vton.conv_in = conv_new  
        print('Add 4 zero-initialized channels to the first convolutional layer of the denoising UNet to support our input with 8 channels')
else:
    print("in_channels = 8")

vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
print("===============================Now using in_channels 2222:", unet_vton.conv_in.in_channels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training type
weight_dtype=torch.float32
if args.mixed_precision == "fp16":
    weight_dtype = torch.float16
elif args.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16
print("------------------------------------------------------------weight_dtype = ", weight_dtype)
# vae.to(device,dtype=weight_dtype)
# unet_garm.to(device)
# unet_vton.to(device) # TODO unet_garm和unet_vton设置 dtype会报错，看下什么原因
# image_encoder.to(device,dtype=weight_dtype)
# text_encoder.to(device,dtype=weight_dtype)


from pytorch_lightning.callbacks import ModelCheckpoint
# **保存 Checkpoint（仅保存模型权重，减少显存占用）** TODO callback实际触发的回调是on_train_epoch_end
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",     # 指定模型保存路径
    filename="epoch={epoch}-step={step}-hd",  # 设置保存文件的命名格式
    save_weights_only=False,   # **True只保存权重，减少显存占用**   False显存会溢出
    every_n_epochs=1,          # **每 5 个 epoch 保存一次**
) # TODO ModelCheckpoint 并不会保存 模型结构信息（比如你把 in_channels=4 改成 in_channels=8）

from logger import ImageLogger
logger_freq = 1000
logger = ImageLogger(batch_frequency=logger_freq) # TODO 看一下image logger是否适配

accumulate_grad_batches=1 

# trainer = pl.Trainer(devices=2, strategy="ddp", precision=16, accelerator="gpu", 
#                      max_epochs=50, callbacks=[checkpoint_callback], accumulate_grad_batches=accumulate_grad_batches)

# trainer = pl.Trainer(gpus=2, strategy="ddp_sharded", precision=16, accelerator="gpu", 
#                      max_epochs=20, callbacks=[checkpoint_callback], progress_bar_refresh_rate=1, accumulate_grad_batches=accumulate_grad_batches)
# trainer = pl.Trainer(gpus=1, precision=16, accelerator="gpu", 
#                      max_epochs=50, checkpoint_callback=False, progress_bar_refresh_rate=1, accumulate_grad_batches=accumulate_grad_batches)

trainer = pl.Trainer(gpus=1, precision=16, accelerator="gpu", 
                     max_epochs=50, callbacks=[logger], progress_bar_refresh_rate=1, accumulate_grad_batches=accumulate_grad_batches)


# 实例化模型
model = VTONModel(
    unet_garm=unet_garm, 
    unet_vton=unet_vton, 
    vae=vae, 
    text_encoder=text_encoder, 
    tokenizer=tokenizer,
    image_processor=image_processor,
    image_encoder=image_encoder, 
    noise_scheduler=noise_scheduler, 
    auto_processor=auto_processor, 
    train_data_loader=train_dataloader, 
    learning_rate=1e-4,
    model_type="hd"
)
# model = create_model('./configs/anydoor.yaml').cpu() # TODO anydoor的模型的各个参数 比如通道数都完全知道并且准确了。我的初始化并不知道学习其初始化方法
# 启动训练
trainer.fit(model)

# 优化器状态（optimizer state）和调度器状态（lr scheduler state）也一并保存并加载，否则会从头重新优化；# TODO 要不要保存一整个模块呢
# trainer.fit(model, ckpt_path="checkpoints/epoch=10-step=xxxxx-hd.ckpt")
