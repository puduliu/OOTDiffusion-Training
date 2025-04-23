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

from pipelines_vton.pipeline_ip_vton import StableDiffusionPipeline as IPVTONPipeline
from pipelines_vton.unet_garm_2d_condition import UNet2DConditionModel as UNetGarm2DConditionModel#TODO 都是要修改导入的，因为要输出特征，确认下和源码有何不同
from pipelines_vton.unet_vton_2d_condition import UNet2DConditionModel as UNetVton2DConditionModel
from diffusers import UniPCMultistepScheduler
from diffusers import AutoencoderKL

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
    
        
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(VIT_PATH).to(self.gpu_id) # TODO 是ip_adapter自带的还是VIT，查看下diffuer的源码
        #TODO image_encoder ootd有输入这个吗 
        self.pipe = IPVTONPipeline.from_pretrained(
            MODEL_PATH,
            unet_garm=unet_garm,
            unet=unet_vton, # 改成unet，适配ipadapter
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
            image_encoder = self.image_encoder # TODO 这个是我另加的
        ).to(self.gpu_id)
        
        #TODO 使用ip adapter就算是从源码修改还是会报错，看一下如何调用！！！！
        
        # self.pipe.load_ip_adapter("../IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        self.pipe.load_ip_adapter("/media/jqzhu/941A7DD31A7DB33A/lpd/OOTDiffusion-Training/IP-Adapter"
                                  , subfolder="models", weight_name="ip-adapter_sd15.bin")
        # 'OotdPipeline' object has no attribute 'unet'
        # TODO 没有继承IPAdapterMixin，手敲吧


        # image_encoder = self.get_image_encoder(repo_id="h94/IP-Adapter", subfolder="models/image_encoder")
        # pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        #     "runwayml/stable-diffusion-v1-5", image_encoder=image_encoder, safety_checker=None, torch_dtype=self.dtype
        # )
        # pipeline.to(torch_device)
        # pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        # TODO 别人是这样加载的, 有传入image_encoder? 看一下diffuer源码
        
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

            images = self.pipe(prompt_embeds=prompt_embeds,
                        image_garm=image_garm,
                        image_vton=image_vton, 
                        mask=mask,
                        image_ori=image_ori,
                        num_inference_steps=num_steps,
                        image_guidance_scale=image_scale,
                        num_images_per_prompt=num_samples,
                        ip_adapter_image=image_garm,
                        generator=generator,
            ).images

        return images
