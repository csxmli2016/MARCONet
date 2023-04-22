import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange


'''
Details of our transformer encoder
'''

class TextViT(nn.Module):
    def __init__(self, num_classes, dim, max_length=16):
        super().__init__()
        image_size=(8,512)
        patch_size=8
        depth = 3 
        heads = 8 
        mlp_dim = 1024
        channels = 512
        dim_head = 64
        max_length = 16

        self.max_length = max_length
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()
        self.linear_cls = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        lengh_seq = int(32*max_length/8)

        
        self.linear_locs = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim//2), 
            nn.GELU(),
            nn.Linear(dim//2, 2), 
            nn.Sigmoid()
        )

        self.linear_w = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 512)
        )
        self.linear_w_maxlen = nn.Sequential(
            nn.LayerNorm(lengh_seq),
            nn.Linear(lengh_seq, 1), #
        )
   

    def forward(self, img):
        x0 = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x0)
        x = rearrange(x0, 'b ... d -> b (...) d') + pe 
        x1, x_loc, x_w = self.transformer(x) #
        x2 = self.to_latent(x1)
        out_cls = self.linear_cls(x2)
        x_w = self.linear_w_maxlen(x_w.permute(0,2,1)).permute(0,2,1)
        out_w = self.linear_w(x_w.view(x_w.size(0), -1))
    
        out_locs = self.linear_locs(x_loc)
        out_locs_16 = out_locs.view(x_loc.size(0), -1)
        return out_cls, out_locs_16, out_w.view(x_w.size(0), -1)



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth-1):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))

        self.layers_cls = nn.ModuleList([])
        self.layers_cls.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
        self.layers_locs = nn.ModuleList([])
        self.layers_locs.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim//2),
            ]))
        
        self.layers_w = nn.ModuleList([])
        self.layers_w.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim//2),
            ]))
        self.linear_seq_maxlen = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 16), #
        )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        for attn_cls, ff_cls in self.layers_cls:
            x_cls = attn_cls(x) + x
            x_cls = ff_cls(x_cls) + x_cls

        x_16 = self.linear_seq_maxlen(x.permute(0,2,1)).permute(0,2,1)
        for attn_loc, ff_loc in self.layers_locs:
            x_loc = attn_loc(x_16) + x_16
            x_loc = ff_loc(x_loc) + x_loc
        
        for attn_w, ff_w in self.layers_w:
            x_w = attn_w(x) + x
            x_w = ff_w(x_w) + x_w

        return x_cls, x_loc, x_w


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)
