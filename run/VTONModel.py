import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader

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
        image_latents_garm = self.vae.encode(image_garm).latent_dist.mode()
        image_latents_vton = self.vae.encode(image_vton).latent_dist.mode()
        latent_vton_model_input = torch.cat([noisy_latents, image_latents_vton], dim=1)

        with torch.cuda.amp.autocast():       
            # 服装融合
            _, spatial_attn_outputs = self.unet_garm(
                image_latents_garm, 0, encoder_hidden_states=prompt_embeds, return_dict=False
            )

            # 试穿去噪
            noise_pred = self.unet_vton(
                latent_vton_model_input, spatial_attn_outputs.copy(), timesteps, encoder_hidden_states=prompt_embeds, return_dict=False
            )[0]  

            # 计算损失
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.unet_garm.parameters()) + list(self.unet_vton.parameters()),  # 只优化这两个模块
            lr=self.learning_rate,
            weight_decay=1e-4  # 适当加一点 L2 正则化
        )
        return optimizer

    def train_dataloader(self):
        return self.train_data_loader

