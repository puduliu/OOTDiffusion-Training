import sys
import os
from shutil import copyfile, rmtree
import argparse
from utils.dataset import DressCodeDataLoader, DressCodeDataset, VITONDataLoader, VITONDataset
CUDIA_VISIBLE_DEVICES = 0,1
# models import
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from diffusers.optimization import get_scheduler 
from diffusers.models import AutoencoderKL
from diffusers import UniPCMultistepScheduler, PNDMScheduler
sys.path.append(r'../ip_vton')
from pipelines_vton.unet_garm_2d_condition import UNet2DConditionModel as UNetGarm2DConditionModel#TODO 都是要修改导入的，因为要输出特征，确认下和源码有何不同
from pipelines_vton.unet_vton_2d_condition import UNet2DConditionModel as UNetVton2DConditionModel
# train tools import
from tqdm import tqdm
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
import math
from pathlib import Path
import logging

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
    parser.add_argument("--unet_path", type=str, default="../checkpoints/stable-diffusion-v1-5/unet")
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


import numpy as np
import torchvision
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor
from torchvision.transforms.functional import to_pil_image

def log_local(save_dir, split, images, global_step, current_epoch, batch_idx):
    root = os.path.join(save_dir, "image_log", split)
    rescale=True
    for k in images:
        grid = torchvision.utils.make_grid(images[k], nrow=4)
        if rescale:
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
        grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
        grid = grid.numpy()
        grid = (grid * 255).astype(np.uint8)
        filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
        path = os.path.join(root, filename)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        Image.fromarray(grid).save(path)
            
def log_images(
    batch,
    vae,
    unet_garm,
    unet_vton,
    image_encoder,
    text_encoder,
    auto_processor,
    image_processor,
    model_type,
    device):
    img_name = batch['img_name']
    garm_name = batch['garm_name']
    print("========================img_name = ", img_name, "====garm_name = ", garm_name)
    # TODO 单纯打印测试使用
    
    image_garm = batch['img_garm'].to(device) # TODO 要换的衣服
    image_vton = batch['img_vton'].to(device) # TODO 对模特衣服进行mask后的目标图像
    image_ori = batch['img_ori'].to(device) # TODO 原图
    img_vton_mask = batch['img_vton_mask'].to(device)
    
    print("#########################image_vton tensor range1111",image_vton.min(), image_vton.max())
    
    prompt = batch["prompt"]
    prompt_vton = [f'Model is wearing {item}' for item in prompt] # TODO 给 vton使用的提示词
    print("---------------------------------------prompt = ", prompt)
    print("---------------------------------------prompt_vton = ", prompt_vton)
    # 保存图片以便可视化
    save_dir = "./debug_images"
    os.makedirs(save_dir, exist_ok=True)
    for i in range(image_garm.size(0)):
        to_pil_image(image_garm[i].cpu()).save(os.path.join(save_dir, f"batch_garm.jpg"))
        to_pil_image(image_vton[i].cpu()).save(os.path.join(save_dir, f"batch_vton.jpg"))
        to_pil_image(image_ori[i].cpu()).save(os.path.join(save_dir, f"batch_ori.jpg"))
        to_pil_image(img_vton_mask[i].cpu()).save(os.path.join(save_dir, f"img_vton_mask.jpg"))

    # 获取服装嵌入
    prompt_image = auto_processor(images=image_garm, return_tensors="pt").data['pixel_values'].to(device)
    prompt_image = image_encoder(prompt_image).image_embeds.unsqueeze(1)

    if model_type == 'hd':
        prompt_embeds = text_encoder(tokenize_captions(prompt).to(device))[0]
        print("-------------------------------------------------prompt_embeds.shape = ", prompt_embeds.shape) 
        # [2, 77, 768] max length是否影响训练速度? 长度和内容，推理的时候测试不会影响结果
        prompt_embeds[:, 1:] = prompt_image[:]
        
        prompt_embeds_vton = text_encoder(tokenize_captions(prompt_vton).to(device))[0]
        print("-------------------------------------------------prompt_embeds.shape = ", prompt_embeds_vton.shape) 
        prompt_embeds_vton[:, 1:] = prompt_image[:]
    elif model_type == 'dc':
        prompt_embeds = text_encoder(tokenize_captions(prompt).to(device))[0]
        prompt_embeds = torch.cat([prompt_embeds, prompt_image], dim=1)
        
        prompt_embeds_vton = text_encoder(tokenize_captions(prompt_vton).to(device))[0]
        print("-------------------------------------------------prompt_embeds.shape = ", prompt_embeds_vton.shape) 
        prompt_embeds_vton[:, 1:] = prompt_image[:]
    else:
        raise ValueError("model_type must be 'hd' or 'dc'!")

    # TODO 参数设置, 如果 num_images_per_prompt = 1 和 do_classifier_free_guidance = False，则与原来代码一样?
    num_images_per_prompt = 1 # TODO 每个prompt生成一张图像就好
    do_classifier_free_guidance = False
    image_guidance_scale = 2.0 # TODO edit
    batch_size = prompt_embeds.shape[0]


    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    prompt_embeds_vton = prompt_embeds_vton.to(dtype=text_encoder.dtype, device=device)
    
    if do_classifier_free_guidance: # TODO 有执行到
        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds])
        prompt_embeds_vton = torch.cat([prompt_embeds_vton, prompt_embeds_vton])

    image_garm = image_processor.preprocess(image_garm)
    image_vton = image_processor.preprocess(image_vton)
    image_ori = image_processor.preprocess(image_ori) # TODO check这个做了mask没

    num_inference_steps =  20
    # 4. set timesteps
    MODEL_PATH = "/home/zyserver/work/lpd/OOTDiffusion-Training/checkpoints/stable-diffusion-v1-5"
    scheduler = UniPCMultistepScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    
    image_garm = image_garm.to(device=device, dtype=prompt_embeds.dtype)
    garm_latents = vae.encode(image_garm).latent_dist.mode()
    garm_latents = torch.cat([garm_latents], dim=0)
    if do_classifier_free_guidance:
        uncond_garm_latents = torch.zeros_like(garm_latents)
        garm_latents = torch.cat([garm_latents, uncond_garm_latents], dim=0) # TODO cat uncond_image_latents

    image_vton = image_vton.to(device=device, dtype=prompt_embeds.dtype)
    image_ori = image_ori.to(device=device, dtype=prompt_embeds.dtype)
    vton_latents = vae.encode(image_vton).latent_dist.mode()
    image_ori_latents = vae.encode(image_ori).latent_dist.mode()
    vton_latents = torch.cat([vton_latents], dim=0)
    image_ori_latents = torch.cat([image_ori_latents], dim=0)

    mask_latents = torch.nn.functional.interpolate(
        img_vton_mask, size=(vton_latents.size(-2), vton_latents.size(-1))
    )
    print("============================================vton_latents.shape = ", vton_latents.shape)
    print("============================================mask_latents.shape = ", mask_latents.shape)
    if do_classifier_free_guidance:
        vton_latents = torch.cat([vton_latents] * 2, dim=0) # TODO 这个直接 * 2?
        
    log = dict() # TODO check 要返回什么, image是什么类型
    image_ori_decode = vae.decode(image_ori_latents, return_dict=False)[0] # TODO vae.decode
    log["rescontruction"] = image_ori_decode #重建看看ori图像，按道理训练的时候应该是要匹配的?

    image_vton_decode = vae.decode(garm_latents, return_dict=False)[0] # TODO vae.decode 

    log["condition"] = image_vton_decode #重建看看ori图像，按道理训练的时候应该是要匹配的? 
    
    height, width = vton_latents.shape[-2:] # height =  64 ----width =  48
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    height = height * vae_scale_factor
    width = width * vae_scale_factor # height =  512 ----width =  384

    # 6. Prepare latent variables
    num_channels_latents = vae.config.latent_channels
    seed = 1 # TODO check干啥用的, 需不需要随机 random.seed(time.time())， seed = random.randint(0, 2147483647)
    generator = torch.manual_seed(seed)
    
    shape = (batch_size * num_images_per_prompt, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)
    print ("==============================================shape = ", shape)
    latents = randn_tensor(shape, generator=generator, device=device, dtype=text_encoder.dtype) # TODO 使用torch.float16
    latents = latents * scheduler.init_noise_sigma
    print ("==============================================latents.shape = ", latents.shape)

    noise = latents.clone()
    
    with torch.cuda.amp.autocast(): # TODO 推理的时候也加上这句话，不然出现dtype问题
        _, spatial_attn_outputs = unet_garm( # TODO float32的模型
                garm_latents, 0, encoder_hidden_states=prompt_embeds, return_dict=False
            )
        for i, t in enumerate(tqdm(timesteps, desc="Sampling", total=num_inference_steps)):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            scaled_latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            latent_vton_model_input = torch.cat([scaled_latent_model_input, vton_latents], dim=1)

            spatial_attn_inputs = spatial_attn_outputs.copy()
            noise_pred = unet_vton( 
                latent_vton_model_input, # TODO 输入?
                spatial_attn_inputs,
                t,
                encoder_hidden_states=prompt_embeds_vton,
                return_dict=False,
            )[0]

            # print("-------------------------------------do_classifier_free_guidance")
            # perform guidance
            if  do_classifier_free_guidance:
                noise_pred_text_image, noise_pred_text = noise_pred.chunk(2)
                noise_pred = (
                    noise_pred_text
                    + image_guidance_scale * (noise_pred_text_image - noise_pred_text)
                )


            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            init_latents_proper = image_ori_latents * vae.config.scaling_factor

            # repainting
            if i < len(timesteps) - 1:
                noise_timestep = timesteps[i + 1]
                init_latents_proper = scheduler.add_noise(
                    init_latents_proper, noise, torch.tensor([noise_timestep])
                )

            latents_mask = (1 - mask_latents) * init_latents_proper + mask_latents * latents # TODO 这个训练的时候可要可不要
            # TODO 我没有mask, 直接latents生成看下是否可行

            # progress_bar.update()

            image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0] # TODO vae.decode
            
            image_mask = vae.decode(latents_mask / vae.config.scaling_factor, return_dict=False)[0] # TODO vae.decode

    log["images"] = image
    log["images_mask"] = image_mask
    return log

#-----prepare dataset-----
# test_dataset = DressCodeDataset(args, "test") # TODO
# test_loader = DressCodeDataLoader(args, test_dataset)
# train_dataset = DressCodeDataset(args, "train")
# train_dataloader =DressCodeDataLoader(args, train_dataset)
# train_dataloader = train_dataloader.data_loader

test_dataset = VITONDataset(args, "test") # TODO
test_loader = VITONDataLoader(args, test_dataset)
train_dataset = VITONDataset(args, "train")
train_dataloader =VITONDataLoader(args, train_dataset)
train_dataloader = train_dataloader.data_loader

#-----load models-----
vae = AutoencoderKL.from_pretrained(args.vae_path)
# unet_garm = UNetGarm2DConditionModel.from_pretrained(args.unet_path,use_safetensors=True)
# unet_vton = UNetVton2DConditionModel.from_pretrained(args.unet_path,use_safetensors=True)
unet_garm = UNetGarm2DConditionModel.from_pretrained("/home/zyserver/work/lpd/OOTDiffusion-Training/checkpoints/stable-diffusion-v1-5/unet",use_safetensors=True)
unet_vton = UNetVton2DConditionModel.from_pretrained("/home/zyserver/work/lpd/OOTDiffusion-Training/checkpoints/stable-diffusion-v1-5/unet",use_safetensors=True)
noise_scheduler = PNDMScheduler.from_pretrained(args.scheduler_path)
auto_processor = AutoProcessor.from_pretrained(args.vit_path)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.vit_path)
tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_path)
text_encoder = CLIPTextModel.from_pretrained(args.text_encoder_path)

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

vae.requires_grad_(False)
unet_garm.requires_grad_(True)
unet_vton.requires_grad_(True)
image_encoder.requires_grad_(False)
text_encoder.requires_grad_(False)

unet_garm.train()
unet_vton.train()

#-----set training environment-----
# run on gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training type
weight_dtype=torch.float32
if args.mixed_precision == "fp16":
    weight_dtype = torch.float16
elif args.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

vae.to(device,dtype=weight_dtype)
unet_garm.to(device)
unet_vton.to(device)
image_encoder.to(device,dtype=weight_dtype)
text_encoder.to(device,dtype=weight_dtype)

#-----training-----
# configs
model_type=args.model_type
batch_size = args.batch_size
train_epochs=args.train_epochs
learning_rate=args.learning_rate
gradient_accumulation_steps=args.gradient_accumulation_steps
first_epoch = args.first_epoch

# logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Training")
loss_log = []
loss_log_file = "./train/loss_log.txt"

# optimizer
all_params = list(unet_garm.parameters()) + list(unet_vton.parameters())
optimizer = torch.optim.AdamW(all_params,lr=learning_rate)

# scaler for AMP
scaler = torch.cuda.amp.GradScaler() # 使用 FP16 / mixed precision 必须使用 GradScaler

# method to tokenize captions
def tokenize_captions(captions):
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids.cuda()
    return inputs

# load checkpoint
if args.checkpoint_path:
    checkpoint = torch.load(args.checkpoint_path)
    unet_garm.load_state_dict(checkpoint['unet_garm'])
    unet_vton.load_state_dict(checkpoint['unet_vton'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # lr_scheduler.load_state_dict(checkpoint['lr_scheduler']) # TODO 学习率是固定的应该无所谓
    scaler.load_state_dict(checkpoint['scaler'])
    first_epoch = checkpoint['epoch'] + 1
    logger.info(f"Resumed training from checkpoint {args.checkpoint_path} at epoch {first_epoch}")

# training loop 
logger.info("Start training")
global_step = 0  # 初始化全局步数
for epoch in tqdm(range(first_epoch, train_epochs)):
    logger.info(f"Epoch {epoch}")
    epoch_loss = 0
    # for step, batch in enumerate(train_dataloader):
    # 给 step 循环包 tqdm，这样你就能看到每个 epoch 内部的进度条了
    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch}"):
        optimizer.zero_grad() # TODO 开头清空梯度了
        
        # 每 500 step 可视化一次
        global_step += 1
        # =========================================
        if step % 500 == 0:
            print("111111----------vae = ",vae.training,"----------unet_garm = "
                ,unet_garm.training,"----------unet_vton = ",unet_vton.training
                ,"----------image_encoder = ",image_encoder.training,"----------text_encoder = ",text_encoder.training)
            is_train = unet_vton.training
            if is_train:
                unet_garm.eval()
                unet_vton.eval()
            print("222222----------vae = ",vae.training,"----------unet_garm = "
                ,unet_garm.training,"----------unet_vton = ",unet_vton.training
                ,"----------image_encoder = ",image_encoder.training,"----------text_encoder = ",text_encoder.training)
            with torch.no_grad():
                images = log_images(        
                        batch,
                        vae=vae,
                        unet_garm=unet_garm,
                        unet_vton=unet_vton,
                        image_encoder=image_encoder,
                        text_encoder=text_encoder,
                        auto_processor=auto_processor,
                        image_processor=image_processor,
                        model_type=model_type,
                        device=device)  # 假设你是在一个类的方法中
            for k in images:
                N = min(images[k].shape[0], 4)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                        # 修复 float16 在 CPU 上不能 clamp 的问题
                    if images[k].dtype == torch.float16:
                        images[k] = images[k].to(torch.float32)
                    images[k] = torch.clamp(images[k], -1., 1.)
            log_local("train/checkpoints_hd", "train", images, global_step, epoch, step)       
            if is_train:
                unet_garm.train()
                unet_vton.train()
                print("333333----------vae = ",vae.training,"----------unet_garm = "
                    ,unet_garm.training,"----------unet_vton = ",unet_vton.training
                    ,"----------image_encoder = ",image_encoder.training,"----------text_encoder = ",text_encoder.training)
        # =========================================
                
        # get original image data
        image_garm = batch['img_garm'].to(device).to(dtype=weight_dtype)
        image_vton = batch['img_vton'].to(device).to(dtype=weight_dtype)
        image_ori = batch['img_ori'].to(device).to(dtype=weight_dtype)
        prompt = batch["prompt"]
        prompt_vton = [f'Model is wearing {item}' for item in prompt] 

        # get garment prompt embeddings
        prompt_image = auto_processor(images=image_garm, return_tensors="pt").data['pixel_values'].to(device)
        prompt_image = image_encoder(prompt_image).image_embeds
        prompt_image = prompt_image.unsqueeze(1)
        
        if model_type == 'hd':
            # for unet_garm
            prompt_embeds = text_encoder(tokenize_captions(prompt).to(device))[0] # TODO 没有设置最大长度吗 max_length
            prompt_embeds[:, 1:] = prompt_image[:]
            # for unet_vton
            prompt_embeds_vton = text_encoder(tokenize_captions(prompt_vton).to(device))[0]
            prompt_embeds_vton[:, 1:] = prompt_image[:]
        elif model_type == 'dc':
            # for unet_garm
            prompt_embeds = text_encoder(tokenize_captions(prompt).to(device))[0]
            prompt_embeds = torch.cat([prompt_embeds, prompt_image], dim=1)
            # for unet_vton
            prompt_embeds_vton = text_encoder(tokenize_captions(prompt_vton).to(device))[0]
            prompt_embeds_vton = torch.cat([prompt_vton, prompt_image], dim=1)
        else:
            raise ValueError("model_type must be 'hd' or 'dc'!")
        
        prompt_embeds = prompt_embeds.to(dtype=weight_dtype, device=device) 
        prompt_embeds_vton = prompt_embeds_vton.to(dtype=weight_dtype, device=device) 
        
        image_garm = image_processor.preprocess(image_garm)
        image_vton = image_processor.preprocess(image_vton)
        image_ori = image_processor.preprocess(image_ori)

        # get model img latents
        latents = vae.encode(image_ori).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        # add noise to latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
        timesteps = timesteps.long()
        
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # get garm and vton img latents
        image_latents_garm = vae.encode(image_garm).latent_dist.mode() # .sample() 加入随机性（训练常用）.mode 取最大可能值（更稳定）
        image_latents_garm = torch.cat([image_latents_garm], dim=0)

        image_latents_vton = vae.encode(image_vton).latent_dist.mode()
        image_latents_vton = torch.cat([image_latents_vton], dim=0)
        latent_vton_model_input = torch.cat([noisy_latents, image_latents_vton], dim=1)
        
        with torch.cuda.amp.autocast():       
            # outfitting fusion
            sample, spatial_attn_outputs = unet_garm(
                image_latents_garm,
                0,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )
            spatial_attn_inputs = spatial_attn_outputs.copy()

            # outfitting denoising
            noise_pred = unet_vton(
                latent_vton_model_input,
                spatial_attn_inputs,
                timesteps,
                encoder_hidden_states=prompt_embeds_vton,
                return_dict=False,
            )[0]  
        
            # calculate loss
            noise_loss= F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            loss = noise_loss
            
            # torch.cuda.empty_cache()

        # backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer) # 执行参数更新
        scaler.update() # 更新 AMP 的动态缩放策略
        
        logger.info(f"Loss: {loss.item()} at step {step} in epoch {epoch}") 
        epoch_loss += loss.item()

    epoch_loss /= len(train_dataloader)
    loss_log.append(epoch_loss)
    logger.info(f"Training loss: {epoch_loss} at epoch {epoch}")
    with open(loss_log_file, "a") as f:
        f.write(f"Epoch {epoch}: Loss {epoch_loss}\n")

    # save checkpoints
    if (epoch%5==0) or epoch == (args.train_epochs - 1):
        save_dir=f"./train/checkpoints_{model_type}"
        checkpoints=os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith("epoch")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[1]))
        if len(checkpoints) >= args.checkpoints_total_limit:
            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                rmtree(removing_checkpoint)
                
        state_dict_unet_vton = unet_vton.state_dict()
        for key in state_dict_unet_vton.keys():
            state_dict_unet_vton[key] = state_dict_unet_vton[key].to('cpu')
        state_dict_unet_garm = unet_garm.state_dict()
        for key in state_dict_unet_garm.keys():
            state_dict_unet_garm[key] = state_dict_unet_garm[key].to('cpu')
        
        checkpoint_dir = os.path.join(save_dir, f"epoch_{str(epoch)}")
        checkpoint_dir_vton = os.path.join(checkpoint_dir, "unet_vton")
        checkpoint_dir_garm = os.path.join(checkpoint_dir, "unet_garm")
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(checkpoint_dir_vton, exist_ok=True)
        os.makedirs(checkpoint_dir_garm, exist_ok=True)
        
        save_file(state_dict_unet_vton, os.path.join(checkpoint_dir_vton, "diffusion_pytorch_model.safetensors"))
        save_file(state_dict_unet_garm, os.path.join(checkpoint_dir_garm, "diffusion_pytorch_model.safetensors"))
        
        state = {
            'epoch': epoch,
            'unet_garm': unet_garm.state_dict(),
            'unet_vton': unet_vton.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict()
        }
        torch.save(state, os.path.join(checkpoint_dir,f"checkpoint-epoch{str(epoch)}.pt"))
        logger.info('Checkpoints successfully saved')
        


