import torch
from vit_pytorch import ViT
import torch.nn as nn
import timm
v = ViT(
    image_size = (224,224),
    patch_size = 16,
    num_classes = 4,
    dim = 768,
    depth = 12,
    heads = 12,
    mlp_dim = 512,

)
# timm加载 mae
timm_v = timm.create_model('vit_base_patch16_224',pretrained=False)
timm_v1 = nn.Sequential(
    timm_v.patch_embed,
    timm_v.pos_drop,
    timm_v.norm_pre,
    timm_v.blocks,
)
# print(timm_v1)
# exit()
img = torch.randn(8, 98, 768)
img1 = torch.randn(1,3,224,224)
preds = v.transformer(img) # (1, 1000)
preds1 = timm_v1(img1)
print(preds.shape)
print(preds1.shape)
#
# exit()
# print(v)
# print(v.state_dict().keys())
# print(timm_v.state_dict().keys())
# print(v)
# print(timm_v)
for name in timm_v.state_dict():
    print(name)
