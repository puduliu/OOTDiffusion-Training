import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class VTONModel(pl.LightningModule):
    # def __init__(self, unet_vton, vae, text_encoder, tokenizer, image_processor, image_encoder, noise_scheduler, auto_processor, 
    #              train_data_loader, learning_rate=1e-4, model_type='hd'):
    def __init__(self, unet_garm=None, unet_vton=None, vae=None, text_encoder=None, tokenizer=None, image_processor=None, image_encoder=None, noise_scheduler=None, auto_processor=None, 
                 train_data_loader=None, learning_rate=1e-4, model_type='hd'):
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
        inputs = self.tokenizer(
            captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.cuda()
        return inputs

    def training_step(self, batch, batch_idx):
        image_garm = batch['img_garm'].to(self.device)
        image_vton = batch['img_vton'].to(self.device)
        image_ori = batch['img_ori'].to(self.device)
        prompt = batch["prompt"]

        # 获取服装嵌入
        prompt_image = self.auto_processor(images=image_garm, return_tensors="pt").data['pixel_values'].to(self.device)
        prompt_image = self.image_encoder(prompt_image).image_embeds.unsqueeze(1)

        if self.model_type == 'hd':
            prompt_embeds = self.text_encoder(self.tokenize_captions(prompt).to(self.device))[0]
            prompt_embeds[:, 1:] = prompt_image[:]
        elif self.model_type == 'dc':
            prompt_embeds = self.text_encoder(self.tokenize_captions(prompt).to(self.device))[0]
            prompt_embeds = torch.cat([prompt_embeds, prompt_image], dim=1)
        else:
            raise ValueError("model_type must be 'hd' or 'dc'!")

        prompt_embeds = prompt_embeds.to(self.device)

        # 预处理图片
        image_garm = self.image_processor.preprocess(image_garm)
        image_vton = self.image_processor.preprocess(image_vton)
        image_ori = self.image_processor.preprocess(image_ori)

        # 获取 VAE 潜变量
        latents = self.vae.encode(image_ori).latent_dist.sample() * self.vae.config.scaling_factor

        # 添加噪声
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.device)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # 服装和试穿图片的潜变量
        # print("-----------------------------------------image_garm.shape = ", image_garm.shape) # torch.Size([1, 3, 512, 384])
        # TODO Stable Diffusion 使用的 VAE 会 将原始输入图像压缩到 1/8 的大小
        image_latents_garm = self.vae.encode(image_garm).latent_dist.mode() #  vae需要输入的宽高是 8 的倍数
        # print("-----------------------------------------image_latents_garm.shape = ", image_latents_garm.shape) # torch.Size([1, 4, 64, 48])
        image_latents_vton = self.vae.encode(image_vton).latent_dist.mode()
        latent_vton_model_input = torch.cat([noisy_latents, image_latents_vton], dim=1)

        with torch.cuda.amp.autocast(): 
            spatial_attn_outputs = self.unet_garm(image_latents_garm)     
            # 服装融合
            # _, spatial_attn_outputs = self.unet_garm(
            #     image_latents_garm, 0, encoder_hidden_states=prompt_embeds, return_dict=False
            # )
            # if isinstance(spatial_attn_outputs, list): # TODO 打印 spatial_attn_outputs,能否轻量化unet
            #     print(f"spatial_attn_outputs is a list with {len(spatial_attn_outputs)} elements.")
            #     for i, tensor in enumerate(spatial_attn_outputs):
            #             print(f"Tensor {i}: shape = {tensor.shape}")
            
            # TODO edit start
            # shapes = [
            #     (1, 3072, 320), (1, 3072, 320),
            #     (1, 768, 640), (1, 768, 640),
            #     (1, 192, 1280), (1, 192, 1280),
            #     (1, 48, 1280), (1, 192, 1280),
            #     (1, 192, 1280), (1, 192, 1280),
            #     (1, 768, 640), (1, 768, 640),
            #     (1, 768, 640), (1, 3072, 320),
            #     (1, 3072, 320), (1, 3072, 320)
            # ]

            # # 生成随机初始化的 tensor list
            # spatial_attn_outputs = [torch.randn(shape, dtype=torch.float16).to(image_latents_garm.device) for shape in shapes] # TODO 确保 spatial_attn_input 在同一设备
        
            # 试穿去噪
            noise_pred = self.unet_vton(
                latent_vton_model_input, spatial_attn_outputs.copy(), timesteps, encoder_hidden_states=prompt_embeds, return_dict=False
            )[0]  

            # 计算损失
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        print("------------------------------------------configure_optimizers!!!!!!============================")
        optimizer = torch.optim.AdamW(
            list(self.unet_garm.parameters()) + list(self.unet_vton.parameters()),  # 只优化这两个模块
            # list(self.unet_vton.parameters()),  # 只优化这两个模块
            lr=self.learning_rate,
            weight_decay=1e-4  # 适当加一点 L2 正则化
        )
        return optimizer
    
    # def on_save_checkpoint(self, checkpoint): # TODO 覆盖父类方法, 自动按照模块保存
    #     """Lightning 自动保存时，按模块保存"""
    #     save_dir = "checkpoints/"
    #     torch.save(self.unet_garm.state_dict(), f"{save_dir}/unet_garm.pth")
    #     torch.save(self.unet_vton.state_dict(), f"{save_dir}/unet_vton.pth")
    #     torch.save(self.vae.state_dict(), f"{save_dir}/vae.pth")
    #     torch.save(self.text_encoder.state_dict(), f"{save_dir}/text_encoder.pth")
    #     torch.save(self.image_encoder.state_dict(), f"{save_dir}/image_encoder.pth")
    #     print("【自动保存】所有子模块已分别保存至 `checkpoints/` 目录")

    def train_dataloader(self):
        return self.train_data_loader
    
    def get_learned_conditioning(self, c):
        #c 1,3,224,224 
        if self.cond_stage_forward is None:
            # print("---------------------------------------cond_stage_forward is NONE") # #这边有执行到
            # 如果 cond_stage_model 有 encode() 并且可调用：用 encode(c) 处理 c。
            # 如果 cond_stage_model 没有 encode()：直接调用 cond_stage_model(c) 处理输入。
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                #1,1,1024
                c = self.cond_stage_model.encode(c) # todo check FrozenDinoV2Encoder? 这边执行到了, c.shape = torch.Size([1, 257, 1024]
                # print("---------------------------------------cond_stage_model.encode(c), shape = ", c.shape) #这边有执行到
                if isinstance(c, DiagonalGaussianDistribution):
                    print("---------------------------------------DiagonalGaussianDistribution")
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            print("---------------------------------------cond_stage_forward not NONE")
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

