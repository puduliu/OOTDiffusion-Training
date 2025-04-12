import torch
import torch.nn as nn
import torch.nn.functional as F

class LiteUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 输入卷积
        self.conv_in = nn.Conv2d(4, 320, kernel_size=3, padding=1)
        
        # 编码阶段：逐步减少通道数
        self.down1 = nn.Conv2d(320, 768, kernel_size=3, stride=2, padding=1)  # 输出: (B, 768, H/2, W/2)
        self.down2 = nn.Conv2d(768, 192, kernel_size=3, stride=2, padding=1)  # 输出: (B, 192, H/4, W/4)
        self.down3 = nn.Conv2d(192, 48, kernel_size=3, stride=2, padding=1)   # 输出: (B, 48, H/8, W/8)
        
        # Bottleneck
        self.bottleneck = nn.Conv2d(48, 48, kernel_size=3, padding=1)  # 输出: (B, 48, H/8, W/8)

        # 解码阶段：逐步恢复通道数
        self.up3 = nn.ConvTranspose2d(48, 192, kernel_size=3, stride=2, padding=1, output_padding=1)  # 输出: (B, 192, H/4, W/4)
        self.up2 = nn.ConvTranspose2d(192, 768, kernel_size=3, stride=2, padding=1, output_padding=1)  # 输出: (B, 768, H/2, W/2)
        self.up1 = nn.ConvTranspose2d(768, 3072, kernel_size=3, stride=2, padding=1, output_padding=1)  # 输出: (B, 3072, H, W)

        # 独立输出层：确保每个输出的形状符合需求
        self.out1 = nn.Conv2d(3072, 3072, kernel_size=1)  # 输出: (B, 3072, H, W)
        self.out2 = nn.Conv2d(3072, 3072, kernel_size=1)  # 输出: (B, 3072, H, W)
        self.out3 = nn.Conv2d(768, 768, kernel_size=1)   # 输出: (B, 768, H/2, W/2)
        self.out4 = nn.Conv2d(768, 768, kernel_size=1)   # 输出: (B, 768, H/2, W/2)
        self.out5 = nn.Conv2d(192, 192, kernel_size=1)   # 输出: (B, 192, H/4, W/4)
        self.out6 = nn.Conv2d(192, 192, kernel_size=1)   # 输出: (B, 192, H/4, W/4)
        self.out7 = nn.Conv2d(48, 48, kernel_size=1)     # 输出: (B, 48, H/8, W/8)
        self.out8 = nn.Conv2d(192, 192, kernel_size=1)   # 输出: (B, 192, H/4, W/4)
        self.out9 = nn.Conv2d(192, 192, kernel_size=1)   # 输出: (B, 192, H/4, W/4)
        self.out10 = nn.Conv2d(192, 192, kernel_size=1)  # 输出: (B, 192, H/4, W/4)
        self.out11 = nn.Conv2d(768, 768, kernel_size=1)  # 输出: (B, 768, H/2, W/2)
        self.out12 = nn.Conv2d(768, 768, kernel_size=1)  # 输出: (B, 768, H/2, W/2)
        self.out13 = nn.Conv2d(3072, 3072, kernel_size=1) # 输出: (B, 3072, H, W)
        self.out14 = nn.Conv2d(3072, 3072, kernel_size=1) # 输出: (B, 3072, H, W)
        self.out15 = nn.Conv2d(3072, 3072, kernel_size=1) # 输出: (B, 3072, H, W)

    def forward(self, x):
        # 编码
        x1 = self.conv_in(x)   # (B, 320, H, W)
        x2 = self.down1(x1)    # (B, 768, H/2, W/2)
        x3 = self.down2(x2)    # (B, 192, H/4, W/4)
        x4 = self.down3(x3)    # (B, 48, H/8, W/8)

        # Bottleneck
        x5 = self.bottleneck(x4)  # (B, 48, H/8, W/8)

        # 解码
        x6 = self.up3(x5)   # (B, 192, H/4, W/4)
        x7 = self.up2(x6)   # (B, 768, H/2, W/2)
        x8 = self.up1(x7)   # (B, 3072, H, W)

        # 输出
        out1 = self.out1(x8)   # (B, 3072, H, W)
        out2 = self.out2(x8)   # (B, 3072, H, W)
        out3 = self.out3(x7)   # (B, 768, H/2, W/2)
        out4 = self.out4(x7)   # (B, 768, H/2, W/2)
        out5 = self.out5(x6)   # (B, 192, H/4, W/4)
        out6 = self.out6(x6)   # (B, 192, H/4, W/4)
        out7 = self.out7(x5)   # (B, 48, H/8, W/8)
        out8 = self.out8(x6)   # (B, 192, H/4, W/4)
        out9 = self.out9(x6)   # (B, 192, H/4, W/4)
        out10 = self.out10(x6)  # (B, 192, H/4, W/4)
        out11 = self.out11(x7)  # (B, 768, H/2, W/2)
        out12 = self.out12(x7)  # (B, 768, H/2, W/2)
        out13 = self.out13(x8)  # (B, 3072, H, W)
        out14 = self.out14(x8)  # (B, 3072, H, W)
        out15 = self.out15(x8)  # (B, 3072, H, W)

        return out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15

# 创建模型
model = LiteUNet()

# 测试输出形状
input_tensor = torch.randn(1, 4, 256, 256)  # 假设输入为 (B, 4, H, W)
outputs = model(input_tensor)

for i, output in enumerate(outputs):
    print(f"Output {i}: shape = {output.shape}")
