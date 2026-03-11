from torchvision.models import vgg16, VGG16_Weights
from torch import nn, optim
import torch.nn.functional as F

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        # use first 3 blocks for low+mid level features
        self.slice1 = vgg[:4].eval()    # edges
        self.slice2 = vgg[4:9].eval()   # textures
        self.slice3 = vgg[9:16].eval()  # patterns
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, x_recon):
        # denormalize from [-1,1] to [0,1] for VGG
        x = x * 0.5 + 0.5
        x_recon = x_recon * 0.5 + 0.5
        loss = 0
        for slice in [self.slice1, self.slice2, self.slice3]:
            x = slice(x)
            x_recon = slice(x_recon)
            loss += F.mse_loss(x_recon, x)
        return loss