import pdb
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
import os
import torch
import numpy as np
from PIL import Image
import cv2

import random
import time
import pdb

from ootd.pipelines_ootd.pipeline_ootd import OotdPipeline
from ootd.pipelines_ootd.unet_garm_2d_condition import UNetGarm2DConditionModel
from ootd.pipelines_ootd.unet_vton_2d_condition import UNetVton2DConditionModel
from diffusers import UniPCMultistepScheduler
from diffusers import AutoencoderKL

import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from transformers import CLIPTextModel, CLIPTokenizer

from run.utils.dataset import DressCodeDataLoader, DressCodeDataset, VITONDataLoader, VITONDataset
from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing

import torchvision

# 指标
from run.utils.val_metrics import compute_metrics
import json

import argparse
parser = argparse.ArgumentParser(description='run ootd')
parser.add_argument('--gpu_id', '-g', type=int, default=0, required=False)
parser.add_argument('--model_type', type=str, default="hd", required=False)
parser.add_argument('--category', '-c', type=int, default=0, required=False)
parser.add_argument('--scale', type=float, default=2.0, required=False)
parser.add_argument('--step', type=int, default=20, required=False)
parser.add_argument('--sample', type=int, default=4, required=False)
parser.add_argument('--seed', type=int, default=42, required=False)

# 添加这两行：
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--dataset_dir', type=str, default='/media/jqzhu/941A7DD31A7DB33A/lpd/download/VITON-HD')
parser.add_argument('--dataset', type=str, default='vitonhd')
parser.add_argument('--semantic_nc', type=int, default=13, help='# of human-parsing map classes')
parser.add_argument('--img_height', type=int, default=512)
parser.add_argument('--img_width', type=int, default=384)
parser.add_argument("--compute_metrics", default=False, action="store_true",
                    help="Compute metrics after generation")

parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Path to the output directory",
)

parser.add_argument(
    "--allow_tf32",
    action="store_true",
    help=(
        "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
        " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
    ),
)

parser.add_argument(
    "--mixed_precision",
    type=str,
    default=None,
    choices=["no", "fp16", "bf16"],
    help=(
        "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
        " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
        " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
    ),
)

parser.add_argument(
    "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
)
parser.add_argument("--test_order", type=str, required=True, choices=["unpaired", "paired"])
parser.add_argument('--dresscode_dataroot', type=str, help='DressCode dataroot')
parser.add_argument('--vitonhd_dataroot', type=str, help='VitonHD dataroot')
args = parser.parse_args()

print ("==============================gpu_id = ", args.gpu_id)
openpose_model = OpenPose(args.gpu_id)
parsing_model = Parsing(args.gpu_id)


category_dict = ['upperbody', 'lowerbody', 'dress']
category_dict_utils = ['upper_body', 'lower_body', 'dresses']

model_type = args.model_type # "hd" or "dc"
category = args.category # 0:upperbody; 1:lowerbody; 2:dress
print("=============================== category = ", category)
image_scale = args.scale # TODO image_scale -> image_guidance_scale
n_steps = args.step
n_samples = args.sample
seed = args.seed


# VIT_PATH = "/home/zyserver/work/lpd/OOTDiffusion-Training/checkpoints/clip-vit-large-patch14"
# VAE_PATH = "/home/zyserver/work/lpd/OOTDiffusion-Training/checkpoints/stable-diffusion-v1-5"
# UNET_PATH = "/home/zyserver/work/lpd/OOTDiffusion-Training/checkpoints/stable-diffusion-v1-5/ootd_hd"
# MODEL_PATH = "/home/zyserver/work/lpd/OOTDiffusion-Training/checkpoints/stable-diffusion-v1-5"


VIT_PATH = "/media/jqzhu/941A7DD31A7DB33A/lpd/OOTDiffusion-Training/checkpoints/clip-vit-large-patch14"
VAE_PATH = "/media/jqzhu/941A7DD31A7DB33A/lpd/OOTDiffusion-Training/checkpoints/stable-diffusion-v1-5"
UNET_PATH = "/media/jqzhu/941A7DD31A7DB33A/lpd/OOTDiffusion-Training/checkpoints/stable-diffusion-v1-5/ootd_hd"
MODEL_PATH = "/media/jqzhu/941A7DD31A7DB33A/lpd/OOTDiffusion-Training/checkpoints/stable-diffusion-v1-5"

gpu_id = 'cuda:' + str(args.gpu_id)

vae = AutoencoderKL.from_pretrained(
    VAE_PATH,
    subfolder="vae",
    torch_dtype=torch.float16,
)

unet_garm = UNetGarm2DConditionModel.from_pretrained(
    UNET_PATH,
    subfolder="unet_garm",
    torch_dtype=torch.float16,
    use_safetensors=True,
)
unet_vton = UNetVton2DConditionModel.from_pretrained(
    UNET_PATH,
    subfolder="unet_vton",
    torch_dtype=torch.float16,
    use_safetensors=True,
)

pipe = OotdPipeline.from_pretrained(
    MODEL_PATH,
    unet_garm=unet_garm,
    unet_vton=unet_vton,
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    safety_checker=None,
    requires_safety_checker=False,
).to(gpu_id)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

auto_processor = AutoProcessor.from_pretrained(VIT_PATH)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(VIT_PATH).to(gpu_id)

tokenizer = CLIPTokenizer.from_pretrained(
    MODEL_PATH,
    subfolder="tokenizer",
)
text_encoder = CLIPTextModel.from_pretrained(
    MODEL_PATH,
    subfolder="text_encoder",
).to(gpu_id)


def tokenize_captions(captions, max_length):
    inputs = tokenizer(
        captions, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids

def pil_to_tensor(images):
    images = np.array(images).astype(np.float32) / 255.0
    images = torch.from_numpy(images.transpose(2, 0, 1))
    return images
print("-------------------------------------------------------args.dataset_dir =", args.dataset_dir)
test_dataset = VITONDataset(args, "test") # TODO
test_dataloader = VITONDataLoader(args, test_dataset)
test_dataloader = test_dataloader.data_loader

# if seed == -1:
#     random.seed(time.time())
#     seed = random.randint(0, 2147483647)
# print('Initial seed: ' + str(seed))
# generator = torch.manual_seed(seed)

# 生成保存文件夹
save_dir = os.path.join(args.output_dir, args.test_order)
save_dir = os.path.join(save_dir, category_dict_utils[category])
os.makedirs(save_dir, exist_ok=True)
    
generator = torch.Generator("cuda").manual_seed(args.seed) # TODO ladi别人是手动设置seed的，不是随机的?

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm
with torch.no_grad():
    # Extract the images
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Running inference"):
            # for batch in test_dataloader:
                img_name = batch['img_name']
                garm_name = batch['garm_name']
                # print("========================img_name = ", img_name, "====garm_name = ", garm_name)
                # TODO 单纯打印测试使用
                
                image_garm = batch['img_garm'].to(gpu_id) # TODO 要换的衣服
                image_vton = batch['img_vton'].to(gpu_id) # TODO 对模特衣服进行mask后的目标图像
                image_ori = batch['img_ori'].to(gpu_id) # TODO 原图
                img_vton_mask = batch['img_vton_mask'] # 加载在cpu上，否则报错
                # print("===image_garm.type = ", type(image_garm), "====img_vton_mask.type",type(img_vton_mask)) # Tensor
                # print("===image_garm.device = ", image_garm.device, "====img_vton_mask.device",img_vton_mask.device) # cpu
                
                # print("#########################image_vton tensor range1111",image_vton.min(), image_vton.max())
                
                prompt = batch["prompt"]
                prompt_vton = [f'Model is wearing {item}' for item in prompt] # TODO 给 vton使用的提示词
                prompt_image = auto_processor(images=image_garm, return_tensors="pt").to(gpu_id)
                prompt_image = image_encoder(prompt_image.data['pixel_values']).image_embeds
                prompt_image = prompt_image.unsqueeze(1)
                if model_type == 'hd':
                    prompt_embeds = text_encoder(tokenize_captions([""], 2).to(gpu_id))[0] # TODO check下有啥不一样 A cloth，好像确实影响不大
                    # print("-------------------------------------------------------prompt_embeds.shape = ", prompt_embeds.shape)
                    prompt_embeds[:, 1:] = prompt_image[:] # TODO 这是直接替换?
                elif model_type == 'dc':
                    prompt_embeds = text_encoder(tokenize_captions([category_dict_utils[category]], 3).to(gpu_id))[0]
                    prompt_embeds = torch.cat([prompt_embeds, prompt_image], dim=1) # TODO 这是concat?
                else:
                    raise ValueError("model_type must be \'hd\' or \'dc\'!")

                images = pipe(prompt_embeds=prompt_embeds,
                            image_garm=image_garm,
                            image_vton=image_vton, 
                            mask=img_vton_mask,
                            image_ori=image_ori,
                            num_inference_steps=20,
                            image_guidance_scale=image_scale,
                            num_images_per_prompt=1, # TODO 设置为1就可以了
                            generator=generator,
                ).images
                
                # print("--------------------------------images.type = ", type(images[0])) # 已经是pil image 类型
                for i in range(len(images)): # 不同的batch size 导致一次生成多张图片?
                    # image= image.resize((768, 1024),Image.NEAREST)
                    images[i].save(os.path.join(save_dir,img_name[i])) # TODO 保存png格式
                
                # TODO check 已经是pil image为什么还要转tensor，两个工程的数据集加载应该不一样，这种感觉有点色差. 
                # for i in range(len(images)):
                #     x_sample = pil_to_tensor(images[i])
                #     torchvision.utils.save_image(x_sample,os.path.join("output",img_name[i]))


torch.cuda.empty_cache()

save_dir = os.path.join(args.output_dir, args.test_order)
# Compute metrics if requested
if args.compute_metrics:
    metrics = compute_metrics(save_dir, args.test_order, args.dataset, category_dict_utils[category], ['all'],
                                args.dresscode_dataroot, args.vitonhd_dataroot)

    with open(os.path.join(save_dir, f"metrics_{args.test_order}_{category_dict_utils[category]}.json"), "w+") as f:
        json.dump(metrics, f, indent=4)