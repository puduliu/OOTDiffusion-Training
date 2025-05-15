import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel
import torchvision.transforms as T
from PIL import Image
import sys


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):

    def encode(self, x):
        return x


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class', ucg_rate=0.1):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.ucg_rate = ucg_rate

    def forward(self, batch, key=None, disable_dropout=False):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        if self.ucg_rate > 0. and not disable_dropout:
            mask = 1. - torch.bernoulli(torch.ones_like(c) * self.ucg_rate)
            c = mask * c + (1-mask) * torch.ones_like(c)*(self.n_classes-1)
            c = c.long()
        c = self.embedding(c)
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = self.n_classes - 1  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc}
        return uc


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self



sys.path.append("../dinov2")
import hubconf
# from omegaconf import OmegaConf
# config_path = './configs/anydoor.yaml'
# config = OmegaConf.load(config_path)
# DINOv2_weight_path = config.model.params.cond_stage_config.weight # todo 加载预训练模型? 模型的输入只能是224x224大小吗?

DINOv2_weight_path = "/media/jqzhu/941A7DD31A7DB33A/lpd/download/dinov2_vitg14_pretrain.pth"
class FrozenDinoV2Encoder(AbstractEncoder):
    """
    Uses the DINOv2 encoder for image
    """
    def __init__(self, device="cuda", freeze=True):
        super().__init__()
        dinov2 = hubconf.dinov2_vitg14() 
        state_dict = torch.load(DINOv2_weight_path)
        dinov2.load_state_dict(state_dict, strict=False)
        self.model = dinov2.to(device)
        self.device = device
        if freeze:
            self.freeze()
        self.image_mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.image_std =  torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)        
        self.projector = nn.Linear(1536,1024)

    def freeze(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image): # TODO check attention map哪里可以输出
        # print("-----------------------------FrozenDinoV2Encoder encode!!") #这边有执行到
        # todo FrozenDinoV2Encoder 
        if isinstance(image,list):
            image = torch.cat(image,0)

        image = (image.to(self.device)  - self.image_mean.to(self.device)) / self.image_std.to(self.device)
        features = self.model.forward_features(image)
        # 假设 features 是一个字典，里面包含 NumPy 数组或 Tensor

        # print("------------------------------------features = ", features)
        tokens = features["x_norm_patchtokens"] #([1, 256, 1536]) [B, 256, 1536]
        # print("------------------------------------tokens.shape = ", tokens.shape)
        image_features  = features["x_norm_clstoken"] #([1, 1536]) [B, 1536]
        # print("------------------------------------image_features.shape1 = ", image_features.shape)
        image_features = image_features.unsqueeze(1) #([1, 1, 1536]) [B, 1, 1536]
        # print("------------------------------------image_features.shape2 = ", image_features.shape)
        hint = torch.cat([image_features,tokens],1) # 8,257,1024 hint.shape1 =  torch.Size([1, 257, 1536]) [B, 257, 1536]
        # print("------------------------------------hint.shape1 = ", hint.shape)
        hint = self.projector(hint) #([1, 256, 1536]) # 8,257,1024 hint.shape2 =  torch.Size([1, 257, 1024])
        # print("------------------------------------hint.shape2 = ", hint.shape)
        return hint

    def encode(self, image):
        return self(image)







