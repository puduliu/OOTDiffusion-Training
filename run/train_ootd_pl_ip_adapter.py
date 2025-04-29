import sys
import os
from shutil import copyfile, rmtree
import argparse
from utils.dataset import DressCodeDataLoader, DressCodeDataset, VITONDataLoader, VITONDataset
sys.path.append(r'../ip_vton')
# models import
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from diffusers.optimization import get_scheduler 
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers import UniPCMultistepScheduler, PNDMScheduler
from pipelines_vton.unet_garm_2d_condition import UNet2DConditionModel as UNetGarm2DConditionModel#TODO 都是要修改导入的，因为要输出特征，确认下和源码有何不同
from pipelines_vton.unet_vton_2d_condition import UNet2DConditionModel as UNetVton2DConditionModel
# from pipelines_vton.pipeline_ip_vton import StableDiffusionPipeline

# train tools import
from tqdm import tqdm
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
import math
from pathlib import Path
import logging

import pytorch_lightning as pl
from VTONModel_clip_vtion_text_image_ip import VTONModel # TODO edit change
sys.path.append(r'../IP-Adapter')
from ip_adapter.ip_adapter import Resampler, IPAdapter, ImageProjModel

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

from ip_adapter.utils import is_torch2_available

if is_torch2_available():
    print("====================================torch2 is available")
    from ip_adapter.attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from ip_adapter.attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from ip_adapter.attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
else:
    print("====================================torch2 is not available")
    from ip_adapter.attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor
from safetensors import safe_open

# def set_ip_adapter(unet,num_tokens, device,dtype):
#     attn_procs = {}
#     for name in unet.attn_processors.keys():
#         cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
#         if name.startswith("mid_block"):
#             hidden_size = unet.config.block_out_channels[-1]
#         elif name.startswith("up_blocks"):
#             block_id = int(name[len("up_blocks.")])
#             hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
#         elif name.startswith("down_blocks"):
#             block_id = int(name[len("down_blocks.")])
#             hidden_size = unet.config.block_out_channels[block_id]
#         if cross_attention_dim is None:
#             attn_procs[name] = AttnProcessor()
#         else:
#             attn_procs[name] = IPAttnProcessor( # TODO 用自定义的注意力模块（IPAttnProcessor）替换掉原本 UNet 里的 cross-attention processor，从而实现图像引导?
#                 hidden_size=hidden_size,
#                 cross_attention_dim=cross_attention_dim,
#                 scale=1.0,
#                 num_tokens=num_tokens,
#             ).to(device, dtype=dtype)
#     unet.set_attn_processor(attn_procs)
#     # if hasattr(self.pipe, "controlnet"):
#     #     if isinstance(self.pipe.controlnet, MultiControlNetModel):
#     #         for controlnet in self.pipe.controlnet.nets:
#     #             controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
#     #     else:
#     #         self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

# def load_ip_adapter(unet, ip_ckpt):
#     if os.path.splitext(ip_ckpt)[-1] == ".safetensors":
#         state_dict = {"image_proj": {}, "ip_adapter": {}}
#         with safe_open(ip_ckpt, framework="pt", device="cpu") as f:
#             for key in f.keys():
#                 if key.startswith("image_proj."):
#                     state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
#                 elif key.startswith("ip_adapter."):
#                     state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
#     else:
#         state_dict = torch.load(ip_ckpt, map_location="cpu")
#     image_proj_model.load_state_dict(state_dict["image_proj"])
#     ip_layers = torch.nn.ModuleList(unet.attn_processors.values())
#     ip_layers.load_state_dict(state_dict["ip_adapter"])


# def init_proj(unet, num_tokens, device, dtype):
#     image_proj_model = Resampler(
#         dim=unet.config.cross_attention_dim,
#         depth=4,
#         dim_head=64,
#         heads=12,
#         num_queries=num_tokens,
#         embedding_dim=image_encoder.config.hidden_size,
#         output_dim=unet.config.cross_attention_dim,
#         ff_mult=4,
#     ).to(device, dtype=dtype)
#     return image_proj_model
    

args = get_args()

test_dataset = VITONDataset(args, "test") # TODO
test_loader = VITONDataLoader(args, test_dataset)
train_dataset = VITONDataset(args, "train")
train_dataloader =VITONDataLoader(args, train_dataset)
train_dataloader = train_dataloader.data_loader
print("----------------------------------------args.batch_size = ", args.batch_size)

#-----load models-----
vae = AutoencoderKL.from_pretrained(args.vae_path)

unet_garm = UNetGarm2DConditionModel.from_pretrained(args.unet_path,use_safetensors=True)
unet_vton = UNetVton2DConditionModel.from_pretrained(args.unet_path,use_safetensors=True)

# TODO 本来训练都使用 ../checkpoints/stable-diffusion-v1-5/unet  原始的sd1.5的unet.
# TODO 初始训练暂停后接着训练
# unet_garm = UNetGarm2DConditionModel.from_pretrained("/home/zyserver/work/lpd/OOTDiffusion-Training/run/checkpoints/unet_garm",use_safetensors=True)
# unet_vton = UNetVton2DConditionModel.from_pretrained("/home/zyserver/work/lpd/OOTDiffusion-Training/run/checkpoints/unet_vton",use_safetensors=True)

noise_scheduler = PNDMScheduler.from_pretrained(args.scheduler_path)
auto_processor = AutoProcessor.from_pretrained(args.vit_path)
clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.vit_path)
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

trainer = pl.Trainer(gpus=1, precision=16, accelerator="gpu", 
                     max_epochs=50, callbacks=[logger], progress_bar_refresh_rate=1, accumulate_grad_batches=accumulate_grad_batches)

ip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "../IP-Adapter", subfolder="models/image_encoder", torch_dtype=torch.float16
)

#  TODO check这个有没有用
unet_vton.config.encoder_hid_dim = ip_image_encoder.config.hidden_size
unet_vton.config.encoder_hid_dim_type = "ip_image_proj"
unet_vton.config["encoder_hid_dim"] = ip_image_encoder.config.hidden_size
unet_vton.config["encoder_hid_dim_type"] = "ip_image_proj"
print("=========================encoder_hid_dim_type = ", unet_vton.config.encoder_hid_dim_type)  # 应为 ip_image_proj
print("=========================encoder_hid_dim = ", unet_vton.config.encoder_hid_dim)  

# state_dict = torch.load("../IP-Adapter/models/ip-adapter_sd15.bin", map_location="cpu") # TODO 只有训练的时候加载了 ipadapter?

 #ip-adapter
image_proj_model = ImageProjModel(
    cross_attention_dim=unet_vton.config.cross_attention_dim, 
    clip_embeddings_dim=ip_image_encoder.config.projection_dim, #官方 IP-Adapter 用的，默认是基于 CLIP ViT-H/14（clip_embeddings_dim=1024）
    clip_extra_context_tokens=4,
)
# image_proj_model = image_proj_model.to(device=device, dtype=torch.float16)
print("===================================unet_vton.config.cross_attention_dim = ", unet_vton.config.cross_attention_dim) # TODO cross_attention_dim = 768
# init adapter modules
attn_procs = {}
unet_sd = unet_vton.state_dict()
for name in unet_vton.attn_processors.keys():
    cross_attention_dim = None if name.endswith("attn1.processor") else unet_vton.config.cross_attention_dim
    if name.startswith("mid_block"):
        hidden_size = unet_vton.config.block_out_channels[-1]
    elif name.startswith("up_blocks"):
        block_id = int(name[len("up_blocks.")])
        hidden_size = list(reversed(unet_vton.config.block_out_channels))[block_id]
    elif name.startswith("down_blocks"):
        block_id = int(name[len("down_blocks.")])
        hidden_size = unet_vton.config.block_out_channels[block_id]
    if cross_attention_dim is None:
        attn_procs[name] = AttnProcessor()
    else:
        layer_name = name.split(".processor")[0]
        weights = {
            "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
            "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
        }
        attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
        attn_procs[name].load_state_dict(weights)
unet_vton.set_attn_processor(attn_procs)
adapter_modules = torch.nn.ModuleList(unet_vton.attn_processors.values()) 
# TODO adapter_modules = ModuleList(unet_vton.attn_processors.values()) 这句拷贝的是模块的引用，
state_dict = torch.load("/home/lenovo/work/sjj/OOTDiffusion-Training/IP-Adapter/models/ip-adapter_sd15.bin", map_location="cpu")
print("-------------------------------------------state_dict.keys() = ", state_dict.keys())
image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True) # TODO 实际上是把权重load到UNet里了。
unet_vton.encoder_hid_proj = image_proj_model 

# TODO 这种加载方式是不是也可以 遍历 attn_processors 直接 load
# for name, module in unet_vton.attn_processors.items():
#     module.load_state_dict(state_dict["ip_adapter"][name])



# 实例化模型
model = VTONModel(
    unet_garm=unet_garm, 
    unet_vton=unet_vton, 
    vae=vae, 
    text_encoder=text_encoder, 
    tokenizer=tokenizer,
    image_processor=image_processor,
    image_encoder=clip_image_encoder, # TODO check 显存有没有爆，如果爆的话只使用ip_adapter的image encoder, 不注入prompt image到vton
    ip_image_encoder=ip_image_encoder,
    noise_scheduler=noise_scheduler, 
    auto_processor=auto_processor, 
    train_data_loader=train_dataloader, 
    learning_rate=1e-4, # TODO 学习率调大试试
    model_type="hd"
)

trainer.fit(model)


