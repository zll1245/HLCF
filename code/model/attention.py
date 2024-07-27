import torch
import torch.nn as nn
import torch.nn.functional as F


class attention(nn.Module):
    def __init__(self, input_dim, num_regions, num_attended_regions):
        super(attention, self).__init__()
        self.patch_size = input_dim // num_regions
        self.num_regions = num_regions
        self.num_attended_regions = num_attended_regions
        self.query_key_value = nn.Linear(input_dim, input_dim)
        self.token_to_token_attention = nn.Linear(input_dim, input_dim)
        self.downsample_conv = nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1)
        self.upsample_conv = nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1)

    def forward(self, input):
        S = self.num_regions
        k = self.num_attended_regions

        # Patchify input
        x = F.unfold(input, self.patch_size, stride=self.patch_size)
        x = x.view(x.size(0), S ** 2, -1, self.patch_size ** 2)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size(0), -1, S ** 2)

        # Linear projection of query, key, value
        qkv = self.query_key_value(x)
        query, key, value = qkv.chunk(3, dim=-1)

        # Regional query and key
        query_r = query.mean(dim=1)
        key_r = key.mean(dim=1)

        # Adjacency matrix for regional graph
        A_r = torch.matmul(query_r, key_r.transpose(-1, -2))
        # print(A_r)

        # Compute index matrix of routed regions
        _, I_r = torch.topk(A_r, k, dim=-1)
        #I_r=I_r.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, key.size(-1)).contiguous()
        # I_r = I_r.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, key.size(-1)).reshape(-1, self.num_regions * self.num_attended_regions, key.size(-1))

        print(I_r)
        print("8888888888888")

        # Gather key-value pairs
        print(key)
        key_g = torch.gather(key, 1, I_r.unsqueeze(0).expand(1,2,2))
        value_g = torch.gather(value, 1, I_r.unsqueeze(0).expand(1,2,2))

        # Token-to-token attention
        A = torch.matmul(query, key_g.transpose(-2, -1))
        A = F.softmax(A, dim=-1)
        output = torch.matmul(A, value_g)

        # Upsample and add residual connection
        output = output + self.downsample_conv(value.transpose(-1, -2)).transpose(-1, -2)
        output = self.upsample_conv(output.transpose(-1, -2)).transpose(-1, -2)

        # Unpatchify output
        output = output.view(output.size(0), -1, S ** 2, self.patch_size ** 2)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(output.size(0), -1, S * self.patch_size, S * self.patch_size)

        return output


# 示例用法
# input_dim = 256
# num_regions = 16
# num_attended_regions = 4
# batch_size = 32
# H = 64
# W = 64
#
# input_data = torch.randn(batch_size, input_dim, H, W)
#
# attention_model = DoubleSparseDynamicAttention(input_dim, num_regions, num_attended_regions)
# output = attention_model(input_data)
#print(output)
