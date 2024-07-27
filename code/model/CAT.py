import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class CrossAttentionBlock(nn.Module):
    def __init__(self, channel_a=1, channel_b=1, dim=64*64, num_heads=4, mlp_ratio=4.):
        super().__init__()
        self.transformer1 = Transformer(channel_a, channel_b, dim, 1, num_heads, mlp_ratio)
        self.transformer2 = Transformer(channel_b, channel_a, dim, 1, num_heads, mlp_ratio)

    def forward(self, feats_a, feats_b):
        feats_a = self.transformer1(feats_a, feats_b)
        feats_b = self.transformer2(feats_b, feats_a)
        return feats_a, feats_b

class CrossAttention(nn.Module):
    def __init__(self, channel_a=1, channel_b=1, out_channel=1, dim=64*64, depth=2, num_heads=4, mlp_ratio=4.):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channel_a + channel_b, out_channel, 1, 1),
            nn.BatchNorm2d(out_channel, momentum=0.01),
            nn.ReLU(inplace=True))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(CrossAttentionBlock(channel_a, channel_b, dim, num_heads, mlp_ratio))

    def forward(self, FI, FP):
        ca_feats_a, ca_feats_b = FI, FP
        for i, layer in enumerate(self.layers):
            ca_feats_a, ca_feats_b = layer(ca_feats_a, ca_feats_b)
        fusion_feats = self.fusion_conv(torch.cat([ca_feats_a, ca_feats_b], dim=1))
        return fusion_feats

class Transformer(nn.Module):
    def __init__(self, q_c=1, k_c=1, dim=64*64, depth=2, num_heads=4, mlp_ratio=4.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Block(q_c=q_c, k_c=k_c, dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio))

    def forward(self, query, key):
        output = query
        for i, layer in enumerate(self.layers):
            output = layer(query=output, key=key)
        return output

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.sigmoid = nn.Sigmoid()

    def forward(self, query, key, value):
        B, Q_C, Q_N = query.shape
        B, K_C, K_N = key.shape
        query = query.reshape(B, Q_C, self.num_heads, Q_N // self.num_heads).permute(0, 2, 1, 3)
        key = key.reshape(B, K_C, self.num_heads, K_N // self.num_heads).permute(0, 2, 1, 3)
        value = value.reshape(B, K_C, self.num_heads, K_N // self.num_heads).permute(0, 2, 1, 3)
        attn = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = torch.matmul(attn, value).transpose(1, 2).reshape(B, Q_C, Q_N)
        return x

class Block(nn.Module):
    def __init__(self, q_c, k_c, dim, num_heads, mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.encode_value = nn.Conv2d(in_channels=k_c, out_channels=k_c, kernel_size=1, stride=1, padding=0)
        self.encode_query = nn.Conv2d(in_channels=q_c, out_channels=q_c, kernel_size=1, stride=1, padding=0)
        self.encode_key = nn.Conv2d(in_channels=k_c, out_channels=k_c, kernel_size=1, stride=1, padding=0)
        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.q_embedding = nn.Parameter(torch.randn(1, q_c, 64, 64))
        self.k_embedding = nn.Parameter(torch.randn(1, k_c, 64, 64))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, query, key, query_embed=None, key_embed=None):
        b, c, h, w = query.shape
        query_embed = repeat(self.q_embedding, '() n c d -> b n c d', b=b)
        key_embed = repeat(self.k_embedding, '() n c d -> b n c d', b=b)
        q_embed = self.with_pos_embed(query, query_embed)
        k_embed = self.with_pos_embed(key, key_embed)
        v = self.encode_value(key).view(b, key.shape[1], -1)
        q = self.encode_query(q_embed).view(b, c, -1)
        k = self.encode_key(k_embed).view(b, key.shape[1], -1)
        query = query.view(b, c, -1)
        query = query + self.attn(query=q, key=k, value=v)
        query = query + self.mlp(self.norm2(query))
        query = query.view(b, c, h, w)
        return query

def load_and_preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0)
    return image.cuda()


def tensor_to_image(tensor):
    tensor = tensor.squeeze(0).cpu().detach().numpy()
    tensor = np.clip(tensor, 0, 1)
    image = Image.fromarray((tensor * 255).astype(np.uint8))
    return image


if __name__ == '__main__':
    import os
    model = CrossAttention().cuda()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    feats1 = load_and_preprocess_image('20_side1.png')
    feats2 = load_and_preprocess_image('20_side2.png')
    with torch.no_grad():
        output = model(feats1, feats2)
    print(output.shape)
    # toPIL = transforms.ToPILImage()
    # # pic = toPIL(output[0])
    # # pic.save('random.jpg')
    #
    # # # 显示图像

