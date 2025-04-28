import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from safetensors.torch import save_file
from datetime import datetime
from diffusers.utils.torch_utils import randn_tensor
from diffusers import UniPCMultistepScheduler
from torchvision.transforms.functional import to_pil_image
import os
from tqdm import tqdm

class VTONModel(pl.LightningModule):
    def __init__(self, unet_garm, unet_vton, vae, text_encoder, tokenizer, image_processor, image_encoder, noise_scheduler, auto_processor, 
                 train_data_loader, learning_rate=1e-4, model_type='hd'):
        super().__init__()
        self.unet_garm = unet_garm
        self.unet_vton = unet_vton
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_encoder = image_encoder
        self.noise_scheduler = noise_scheduler
        self.auto_processor = auto_processor
        self.train_data_loader = train_data_loader
        self.learning_rate = learning_rate
        self.model_type = model_type

    def tokenize_captions(self, captions):
        inputs = self.tokenizer( # TODO 这好像和推理的时候不太一样，推理的时候最大是2. 不知道有没有区别
            captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.cuda()
        return inputs
    
    def training_step(self, batch, batch_idx):
        image_garm = batch['img_garm'].to(self.device)
        image_vton = batch['img_vton'].to(self.device)
        image_ori = batch['img_ori'].to(self.device)
        prompt = batch["prompt"] # TODO ['A cloth', 'A cloth'] 有文本提示词的，但是这样训练有用，是否是因为没有冻结unet_garm
        # print("----------------------------------------prompt = ", prompt) # TODO检查下有没有内容，推理的时候没有内容
        prompt_vton = [f'A model is wearing {item}' for item in prompt]

        # 获取服装嵌入
        prompt_image = self.auto_processor(images=image_garm, return_tensors="pt").data['pixel_values'].to(self.device) 
        prompt_image = self.image_encoder(prompt_image).image_embeds.unsqueeze(1)

        if self.model_type == 'hd':
            # TODO check text_encoder
            prompt_embeds = self.text_encoder(self.tokenize_captions(prompt).to(self.device))[0] #TODO 给unet_gram
            # prompt_embeds[:, 1:] = prompt_image[:] 
            # TODO 给 unet_garm和unet_vton同时注入prompt_image效果会不会比较好呢，会不会有助于unet_vton理解 TODO
            # TODO 给 unet_garm和unet_vton同时都不注入prompt_image效果会不会比较好呢，会不会有助于unet_vton理解 TODO
            
            # text_encoder返回得是BaseModelOutputWithPooling ? [0]取得是last_hidden_state打印看看
            prompt_embeds_vton = self.text_encoder(self.tokenize_captions(prompt_vton).to(self.device))[0]
            # 'A model is wearing A cloth传给 vton，应该继续把prompt_image(garm信息传给vton)
            prompt_embeds_vton[:, 1:] = prompt_image[:] # TODO 只给vton输入prompt_image
            # 第一个 token (prompt_embeds_vton[:, 0]) 通常是 特殊的起始符号，但经过 transformer 后，它变成了总结整句话的一个向量。有一定的信息

        elif self.model_type == 'dc':
            prompt_embeds = self.text_encoder(self.tokenize_captions(prompt).to(self.device))[0]
            # TODO dc这个情况的话,提示词要改一下
            prompt_embeds_vton = self.text_encoder(self.tokenize_captions(prompt_vton).to(self.device))[0]
            prompt_embeds_vton = torch.cat([prompt_embeds_vton, prompt_image], dim=1) # TODO 只给vton输入prompt_image
        else:
            raise ValueError("model_type must be 'hd' or 'dc'!")

        prompt_embeds = prompt_embeds.to(self.device)
        prompt_embeds_vton = prompt_embeds_vton.to(self.device)

        # 预处理图片
        image_garm = self.image_processor.preprocess(image_garm)
        image_vton = self.image_processor.preprocess(image_vton)
        image_ori = self.image_processor.preprocess(image_ori)

        # 获取 VAE 潜变量
        latents = self.vae.encode(image_ori).latent_dist.sample() * self.vae.config.scaling_factor
        # TODO 这是用image_ori?,为什么不用image_vton

        # 添加噪声
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.device)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        # TODO 用随机的 t 训练模型去预测对应步长的噪声。add_noise 会按 根号αt x0 + 根号(1−αt) ϵ  的公式合成 xt

        # 服装和试穿图片的潜变量
        # print("-----------------------------------------image_garm.shape = ", image_garm.shape) # torch.Size([1, 3, 512, 384])
        # TODO Stable Diffusion 使用的 VAE 会 将原始输入图像压缩到 1/8 的大小
        image_latents_garm = self.vae.encode(image_garm).latent_dist.mode() #  vae需要输入的宽高是 8 的倍数
        # print("-----------------------------------------image_latents_garm.shape = ", image_latents_garm.shape) # torch.Size([1, 4, 64, 48])
        image_latents_vton = self.vae.encode(image_vton).latent_dist.mode()
        # latent_vton_model_input = noisy_latents + image_latents_vton
        latent_vton_model_input = torch.cat([noisy_latents, image_latents_vton], dim=1) 
        # TODO check 为什么这两个要cat起来，不应该跟纯噪声cat? noisy_latents是纯噪声吗

        with torch.cuda.amp.autocast(): 
               
            # 服装融合
            _, spatial_attn_outputs = self.unet_garm(
                image_latents_garm, 0, encoder_hidden_states=prompt_embeds, return_dict=False
            )
        
            # 试穿去噪
            noise_pred = self.unet_vton(
                latent_vton_model_input, spatial_attn_outputs.copy(), timesteps, encoder_hidden_states=prompt_embeds_vton, return_dict=False
            )[0]  

            # 计算损失
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        print("------------------------------------------configure_optimizers!!!!!!============================")
        optimizer = torch.optim.AdamW(
            # list(self.unet_garm.parameters()) + list(self.unet_vton.parameters()),  # 只优化这两个模块, 优化两个模块爆显存
            list(self.unet_vton.parameters()),  # TODO 只训练unet_vton，大概 45G, batch size = 1的情况下
            lr=self.learning_rate,
            weight_decay=1e-4  # 适当加一点 L2 正则化
        )
        return optimizer
    
    # TODO log images 看看训练的变化？

    def on_epoch_end(self):
            # 每 5 个 epoch 保存一次
            if (self.current_epoch  + 1) % 2 == 0:
                # save_dir = "checkpoints"
                save_dir = "train"
                # torch.save(self.unet_garm.state_dict(), f"{save_dir}/unet_garm.pth")
                save_file(self.unet_vton.state_dict(), f"{save_dir}/unet_vton/unet_vton.safetensors")

                # 获取当前时间
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
                # 输出信息  
                message = f"[{timestamp}]✅ 已保存 unet_garm 和 unet_vton 模型,epoch {self.current_epoch}!\n"  
                
                # 将输出写入txt文件  
                with open(f"{save_dir}/training_log.txt", "a", encoding='utf-8') as log_file:  
                    log_file.write(message) 
                print(f"✅ 已保存 unet_garm 和 unet_vton 模型,epoch {self.current_epoch}!")

    def train_dataloader(self):
        return self.train_data_loader


    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        
        image_garm = batch['img_garm'].to(self.device) # TODO 要换的衣服
        image_vton = batch['img_vton'].to(self.device) # TODO 对模特衣服进行mask后的目标图像
        image_ori = batch['img_ori'].to(self.device) # TODO 原图
        prompt = batch["prompt"]
        prompt_vton = [f'A model is wearing {item}' for item in prompt]
        print("---------------------------------------prompt = ", prompt)
        print("---------------------------------------prompt_vton = ", prompt_vton)

        # 获取服装嵌入
        prompt_image = self.auto_processor(images=image_garm, return_tensors="pt").data['pixel_values'].to(self.device)
        prompt_image = self.image_encoder(prompt_image).image_embeds.unsqueeze(1)

        if self.model_type == 'hd':
            prompt_embeds = self.text_encoder(self.tokenize_captions(prompt).to(self.device))[0]
            prompt_embeds_vton = self.text_encoder(self.tokenize_captions(prompt_vton).to(self.device))[0]
            prompt_embeds_vton[:, 1:] = prompt_image[:] 
        elif self.model_type == 'dc':
            prompt_embeds = self.text_encoder(self.tokenize_captions(prompt).to(self.device))[0]
            prompt_embeds_vton = self.text_encoder(self.tokenize_captions(prompt_vton).to(self.device))[0]
            prompt_embeds_vton = torch.cat([prompt_embeds_vton, prompt_image], dim=1)
        else:
            raise ValueError("model_type must be 'hd' or 'dc'!")

        # TODO 参数设置, 如果 num_images_per_prompt = 1 和 do_classifier_free_guidance = False，则与原来代码一样?
        num_images_per_prompt = 1 # TODO 每个prompt生成一张图像就好
        do_classifier_free_guidance = False
        image_guidance_scale = 2.0 # TODO edit
        batch_size = prompt_embeds.shape[0]


        # prompt_embeds = prompt_embeds.to(self.device)
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=self.device)
        # print("-------------------------------------------_encode_prompt prompt_embeds.shape111111 = ", prompt_embeds.shape)
        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1) # TODO num_images_per_prompt = 1的情况下，prompt_embeds.shape 不会有任何变化
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        # print("-------------------------------------------_encode_prompt prompt_embeds.shape222222 = ", prompt_embeds.shape)
        if do_classifier_free_guidance: # TODO 有执行到
            prompt_embeds = torch.cat([prompt_embeds, prompt_embeds])

        # 预处理图片
        image_garm = self.image_processor.preprocess(image_garm)
        image_vton = self.image_processor.preprocess(image_vton)
        image_ori = self.image_processor.preprocess(image_ori) # TODO check这个做了mask没
    

        #TODO start
        
        num_inference_steps =  20
        # 4. set timesteps
        MODEL_PATH = "/home/zyserver/work/lpd/OOTDiffusion-Training/checkpoints/stable-diffusion-v1-5"
        scheduler = UniPCMultistepScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = scheduler.timesteps
        
        # TODO 方式1
        # garm_latents = self.vae.encode(image_garm).latent_dist.mode() #  vae需要输入的宽高是 8 的倍数
        # vton_latents = self.vae.encode(image_vton).latent_dist.mode()
        # image_ori_latents = self.vae.encode(image_ori).latent_dist.mode() # TODO 不应该训练的时候是GT才好训练吗，如何训练, todo check

        # TODO 方式2
        image_garm = image_garm.to(device=self.device, dtype=prompt_embeds.dtype)
        garm_latents = self.vae.encode(image_garm).latent_dist.mode()
        garm_latents = torch.cat([garm_latents], dim=0)
        if do_classifier_free_guidance:
            uncond_garm_latents = torch.zeros_like(garm_latents)
            garm_latents = torch.cat([garm_latents, uncond_garm_latents], dim=0) # TODO cat uncond_image_latents

        image_vton = image_vton.to(device=self.device, dtype=prompt_embeds.dtype)
        image_ori = image_ori.to(device=self.device, dtype=prompt_embeds.dtype)
        vton_latents = self.vae.encode(image_vton).latent_dist.mode()
        image_ori_latents = self.vae.encode(image_ori).latent_dist.mode()
        vton_latents = torch.cat([vton_latents], dim=0)
        image_ori_latents = torch.cat([image_ori_latents], dim=0)
        if do_classifier_free_guidance:
            vton_latents = torch.cat([vton_latents] * 2, dim=0) # TODO 这个直接 * 2?

        # TODO check 确认下是否影响重建

        # TODO 试试重建的效果
        print("-------------------------------------------self.vae.config.scaling_factor = ", self.vae.config.scaling_factor)
        log = dict() # TODO check 要返回什么, image是什么类型
        # image_ori_decode = self.vae.decode(image_ori_latents / self.vae.config.scaling_factor, return_dict=False)[0] # TODO vae.decode
        # TODO 不用除以 self.vae.config.scaling_factor  可视化效果是好的,check下为什么要除。但是最后输出的要除
        image_ori_decode = self.vae.decode(image_ori_latents, return_dict=False)[0] # TODO vae.decode
        # TODO check为什么重建的时候效果也不好
        log["rescontruction"] = image_ori_decode #重建看看ori图像，按道理训练的时候应该是要匹配的?

        image_vton_decode = self.vae.decode(garm_latents, return_dict=False)[0] # TODO vae.decode 
        # TODO do_classifier_free_guidance = true时候，garm_latents会扩展
        # TODO check为什么重建的时候效果也不好
        log["condition"] = image_vton_decode #重建看看ori图像，按道理训练的时候应该是要匹配的? 
        
        height, width = vton_latents.shape[-2:] # height =  64 ----width =  48
        # print("-----------------------------------batch_size = ",batch_size, "----height = ", height, "----width = ", width)
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        height = height * vae_scale_factor
        width = width * vae_scale_factor # height =  512 ----width =  384
        # print("-----------------------------------batch_size = ",batch_size, "----height = ", height, "----width = ", width)
        

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        seed = 1 # TODO check干啥用的, 需不需要随机 random.seed(time.time())， seed = random.randint(0, 2147483647)
        generator = torch.manual_seed(seed)
        
        shape = (batch_size * num_images_per_prompt, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)
        print ("==============================================shape = ", shape)
        latents = randn_tensor(shape, generator=generator, device=self.device, dtype=torch.float32) # TODO 使用torch.float16
        latents = latents * scheduler.init_noise_sigma
        print ("==============================================latents.shape = ", latents.shape)

        noise = latents.clone()
        
        _, spatial_attn_outputs = self.unet_garm( # TODO float32的模型
                garm_latents, 0, encoder_hidden_states=prompt_embeds, return_dict=False
            )
        
        for i, t in enumerate(tqdm(timesteps, desc="Sampling", total=num_inference_steps)):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            # print ("-----t = ", t,"--------timesteps = ", timesteps,"----latent_model_input.shape = ", latent_model_input.shape)
            # concat latents, image_latents in the channel dimension
            scaled_latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            # print("----vton_latents.shape = ", vton_latents.shape,"--scaled_latent_model_input.shape = ", scaled_latent_model_input.shape)
            # vton_latents.shape要等于scaled_latent_model_input.shape，如果num_images_per_prompt = 1, 均为[batchsize*num, 4, 64, 48]
            latent_vton_model_input = torch.cat([scaled_latent_model_input, vton_latents], dim=1)
            # latent_vton_model_input = scaled_latent_model_input + vton_latents

            spatial_attn_inputs = spatial_attn_outputs.copy()

            noise_pred = self.unet_vton( 
                latent_vton_model_input, # TODO 输入?
                spatial_attn_inputs,
                t,
                encoder_hidden_states=prompt_embeds_vton,
                # added_cond_kwargs=added_cond_kwargs, # TODO add ip_adapter
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

            init_latents_proper = image_ori_latents * self.vae.config.scaling_factor

            # repainting
            if i < len(timesteps) - 1:
                noise_timestep = timesteps[i + 1]
                init_latents_proper = scheduler.add_noise(
                    init_latents_proper, noise, torch.tensor([noise_timestep])
                )

            # latents = (1 - mask_latents) * init_latents_proper + mask_latents * latents
            # TODO 我没有mask, 直接latents生成看下是否可行

            # progress_bar.update()

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0] # TODO vae.decode
        
        # do_denormalize = [True] * image.shape[0]
        # image = self.image_processor.postprocess(image, output_type="pil", do_denormalize=do_denormalize)
        # TODO check能否自动放到logger处理

        log["images"] = image
        return log
