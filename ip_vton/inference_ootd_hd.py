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

# from pipelines_vton.pipeline_ip_vton import StableDiffusionPipeline as IPVTONPipeline
from pipelines_vton.pipeline_ip_vton import StableDiffusionPipeline
from pipelines_vton.unet_garm_2d_condition import UNet2DConditionModel as UNetGarm2DConditionModel#TODO 都是要修改导入的，因为要输出特征，确认下和源码有何不同
from pipelines_vton.unet_vton_2d_condition import UNet2DConditionModel as UNetVton2DConditionModel
from diffusers import UniPCMultistepScheduler
from diffusers import AutoencoderKL
# from diffusers.pipelines import StableDiffusionPipeline, IPAdapterPipeline # TODO check 0.24是不是还没有IPAdapterPipeline?

import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from transformers import CLIPTextModel, CLIPTokenizer
from encoders.modules import FrozenDinoV2Encoder

VIT_PATH = "/media/jqzhu/941A7DD31A7DB33A/lpd/OOTDiffusion-Training/checkpoints/clip-vit-large-patch14"
VAE_PATH = "/media/jqzhu/941A7DD31A7DB33A/lpd/OOTDiffusion-Training/checkpoints/stable-diffusion-v1-5"
UNET_PATH = "/media/jqzhu/941A7DD31A7DB33A/lpd/OOTDiffusion-Training/checkpoints/stable-diffusion-v1-5/ootd_hd"
# UNET_PATH = "/media/jqzhu/941A7DD31A7DB33A/lpd/OOTDiffusion-Training/checkpoints/stable-diffusion-v1-5/unet"
MODEL_PATH = "/media/jqzhu/941A7DD31A7DB33A/lpd/OOTDiffusion-Training/checkpoints/stable-diffusion-v1-5"

class IPAdapterHD:

    def __init__(self, gpu_id):
        self.gpu_id = 'cuda:' + str(gpu_id)

        self.encoder_dinov2  = FrozenDinoV2Encoder(device="cuda", freeze=True).to(self.gpu_id)  # TODO冻结的?
 
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
    
        
        # self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        #     VIT_PATH,
        #     torch_dtype=torch.float16,
        #     ).to(self.gpu_id)
        # TODO SD 1.4 / 1.5	使用ViT-L/14 (clip-vit-large-patch14)，维度w为768
        
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(VIT_PATH).to(self.gpu_id)
        
        self.ip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "../IP-Adapter", subfolder="models/image_encoder", torch_dtype=torch.float16
        ).to(self.gpu_id)
        # TODO IP-Adapter 自带的图像编码器输出是 1024 维
        # TODO check ip adapter 如果你坚持想用标准 clip-vit-large-patch14 作为 image encoder，
        # 那你必须替换掉 UNet 中的 encoder_hid_proj 模块，让它接受 768 维输入
        
        #TODO image_encoder ootd有输入这个吗 
        self.pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_PATH,
            unet_garm=unet_garm,
            unet=unet_vton, # 改成unet，适配ipadapter
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
            image_encoder = self.ip_image_encoder # TODO 这个是我另加的
        ).to(self.gpu_id)
        
        self.pipe.load_ip_adapter("../IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        
        # print("=======================self.unet.encoder_hid_proj.image_embeds.weight = ", self.pipe.unet.encoder_hid_proj.image_embeds.weight)
        
        # TODO load_ip_adapter完self.encoder_hid_proj = ImageProjection, self.config.encoder_hid_dim_type =  ip_image_proj
        # TODO 不加载self.encoder_hid_proj和self.config.encoder_hid_dim_type都为None
        
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        
        self.auto_processor = AutoProcessor.from_pretrained(VIT_PATH) # 也可以用用CLIPProcessor  TODO check

        self.tokenizer = CLIPTokenizer.from_pretrained(
            MODEL_PATH,
            subfolder="tokenizer",
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            MODEL_PATH,
            subfolder="text_encoder",
        ).to(self.gpu_id)
    def tokenize_captions(self, captions, max_length):
        inputs = self.tokenizer(
            captions, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids


    def __call__(self,
                model_type='hd',
                category='upperbody',
                image_garm=None,
                image_vton=None,
                mask=None,
                image_ori=None,
                num_samples=1,
                num_steps=20,
                image_scale=1.0,
                seed=-1,
    ):
        if seed == -1:
            random.seed(time.time())
            seed = random.randint(0, 2147483647)
        print('Initial seed: ' + str(seed))
        generator = torch.manual_seed(seed)

        with torch.no_grad():
            prompt_image = self.auto_processor(images=image_garm, return_tensors="pt").to(self.gpu_id)
            prompt_image = self.image_encoder(prompt_image.data['pixel_values']).image_embeds
            prompt_image = prompt_image.unsqueeze(1)
            print("===============================================image_garm.type = ", type(image_garm)) # PIL.Image.Image
            if model_type == 'hd':
                # test_input = ["A cloth","A cloth"]
                # prompt_embeds = self.text_encoder(self.tokenize_captions([""],1).to(self.gpu_id))[0] # TODO max length不影响
                # print("#############################################prompt_embeds.shape = ", prompt_embeds.shape) # ([1, 2, 768]), batch_size = 2的话是 torch.Size([2, 4, 768])

                # TODO 这边不选择文本注入?
                prompt_embeds = self.text_encoder(self.tokenize_captions([""], 2).to(self.gpu_id))[0] # TODO 最大编码为77, 257不行
                print("------------------------------------------------prompt_embeds.shape1111 = ", prompt_embeds.shape) # ([1, 2, 768])
                # self.tokenize_captions([""], 1) 就算 max len= 1也是生成([1, 2, 768])
                prompt_embeds[:, 1:] = prompt_image[:]
            elif model_type == 'dc':
                prompt_embeds = self.text_encoder(self.tokenize_captions([category], 3).to(self.gpu_id))[0]
                prompt_embeds = torch.cat([prompt_embeds, prompt_image], dim=1)
            else:
                raise ValueError("model_type must be \'hd\' or \'dc\'!")

            cloth_img = Image.open("/media/jqzhu/941A7DD31A7DB33A/lpd/OOTDiffusion-Training/run/examples/garment/10297_00.jpg").resize((768, 1024))
            cloth_img = cloth_img.resize((384, 512), Image.NEAREST)
            images = self.pipe(prompt_embeds=prompt_embeds,
                        image_garm=image_garm,
                        image_vton=image_vton, 
                        mask=mask,
                        image_ori=image_ori,
                        num_inference_steps=num_steps,
                        image_guidance_scale=image_scale,
                        num_images_per_prompt=num_samples,
                        ip_adapter_image=cloth_img,
                        # negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality", 
                        generator=generator,
            ).images

        return images
