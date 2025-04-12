import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLiteUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # conv_in: 将 VAE latent (B,4,64,48) 映射到 (B,320,64,48)
        self.conv_in = nn.Conv2d(4, 320, kernel_size=3, padding=1)
        # 我们之后会把 (B,320,64,48) 展平并转置成 (B, 3072, 320)
        # 其中 64*48=3072
        
        # 下采样阶段（在 token 维度上操作，用 1D 卷积）
        self.down1_conv = nn.Conv1d(3072, 768, kernel_size=3, padding=1)   # (B,3072,320) -> (B,768,320)
        self.down2_conv = nn.Conv1d(768, 192, kernel_size=3, padding=1)     # (B,768,640) -> (B,192,?)
        self.down3_conv = nn.Conv1d(192, 48, kernel_size=3, padding=1)      # (B,192,1280) -> (B,48,1280)
        
        # 上采样阶段（镜像下采样操作）
        self.up3_conv = nn.Conv1d(48, 192, kernel_size=3, padding=1)        # (B,48,1280) -> (B,192,1280)
        self.up2_conv = nn.Conv1d(192, 768, kernel_size=3, padding=1)       # (B,192,1280) -> (B,768,?)
        self.up1_conv = nn.Conv1d(768, 3072, kernel_size=3, padding=1)       # (B,768,640) -> (B,3072,?)
        
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    def forward(self, x):
        # x: (B, 4, 64, 48)
        B = x.shape[0]
        x = self.conv_in(x)   # (B,320,64,48)
        # 将空间维度展平：64×48 = 3072, 得到 (B,320,3072)
        x = x.view(B, 320, -1)  
        # 转置为 (B,3072,320)
        x = x.transpose(1, 2)  
        out1 = x  # (B,3072,320)
        
        # 下采样 block 1:
        x = self.down1_conv(x)  # (B,768,320)
        # 为使 token 数翻倍，从320 -> 640
        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)  # (B,768,640)
        out2 = x  # (B,768,640)
        
        # 下采样 block 2:
        x = self.down2_conv(x)  # (B,192,640)
        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)  # (B,192,1280)
        out3 = x  # (B,192,1280)
        
        # 下采样 block 3:
        x = self.down3_conv(x)  # (B,48,1280)
        out4 = x  # (B,48,1280)
        
        # 上采样 block 1:
        x = self.up3_conv(x)  # (B,192,1280)
        # 加上 skip（来自下采样 block 2）:
        x = x + out3       # (B,192,1280)
        out5 = x  # (B,192,1280)
        
        # 上采样 block 2:
        x = self.up2_conv(x)  # (B,768,1280)
        # 将 token 数减半: 1280 -> 640
        x = F.interpolate(x, scale_factor=0.5, mode='linear', align_corners=False)  # (B,768,640)
        x = x + out2      # (B,768,640)
        out6 = x  # (B,768,640)
        
        # 上采样 block 3:
        x = self.up1_conv(x)  # (B,3072,640)
        x = F.interpolate(x, scale_factor=0.5, mode='linear', align_corners=False)  # (B,3072,320)
        x = x + out1      # (B,3072,320)
        out7 = x  # (B,3072,320)
        
        # 根据需要构造一个包含16个输出的列表（这里只是重复部分输出以示范）
        outputs = [
            out1, out1, 
            out2, out2, 
            out3, out3, 
            out4, 
            out5, out5, out5, 
            out6, out6, out6, 
            out7, out7, out7
        ]
        return outputs

# 测试
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = SimpleLiteUNet().to(device)
# # VAE 编码后的 latent 输入为 (1,4,64,48)
# input_tensor = torch.randn(8, 4, 64, 48).to(device)
# outs = model(input_tensor)
# for i, o in enumerate(outs):
#     print(f"Output {i}: shape = {o.shape}")

# Tensor 0: shape = torch.Size([8, 3072, 320])
# Tensor 1: shape = torch.Size([8, 3072, 320])
# Tensor 2: shape = torch.Size([8, 768, 640])
# Tensor 3: shape = torch.Size([8, 768, 640])
# Tensor 4: shape = torch.Size([8, 192, 1280])
# Tensor 5: shape = torch.Size([8, 192, 1280])
# Tensor 6: shape = torch.Size([8, 48, 1280])
# Tensor 7: shape = torch.Size([8, 192, 1280])
# Tensor 8: shape = torch.Size([8, 192, 1280])
# Tensor 9: shape = torch.Size([8, 192, 1280])
# Tensor 10: shape = torch.Size([8, 768, 640])
# Tensor 11: shape = torch.Size([8, 768, 640])
# Tensor 12: shape = torch.Size([8, 768, 640])
# Tensor 13: shape = torch.Size([8, 3072, 320])
# Tensor 14: shape = torch.Size([8, 3072, 320])
# Tensor 15: shape = torch.Size([8, 3072, 320])