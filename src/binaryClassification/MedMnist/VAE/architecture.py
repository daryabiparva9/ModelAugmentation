import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
import wandb



class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, ksize=(3,3)):
        """

        Args:
            in_channels (int) 
            out_channels (int)
            ksize (tuple, optional): kernel shape (3,3).
        
        Description:
            A Res layer with two blocks of convolutions, batch norm, and relu
        """
        super().__init__()
        
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.ksize = ksize
        # Start with shortcut so that F(x) and x
        # have the same dim for residual part
        # A projection layer that adjusts the input
        # so it can be added to the main path.
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel,
                      out_channels=self.out_channel,
                      kernel_size=self.ksize,
                      padding="same"),
            nn.BatchNorm2d(num_features=out_channels)
        )
        
        # Main Mapping begings here
        self.conv1 = nn.Conv2d(in_channels=self.in_channel,
                      out_channels=self.out_channel,
                      kernel_size=self.ksize,
                      padding="same")
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.act1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=self.out_channel,
                               out_channels=self.out_channel,
                               kernel_size=self.ksize,
                               padding="same")
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.act2 = nn.LeakyReLU(negative_slope=0.2)
        
    
    def forward(self, x):
        
        if self.in_channel != self.out_channel:
            residual = self.shortcut(x)
        else:
            residual = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        
        # Residual connection
        x = residual + x
        x = self.act2(x)
        return x
        
        
        
        
        
class ConvResBlock(nn.Module):
    """ Conv layers followed by resnet blocks
    """
    
    def __init__(self,n_out_conv_filters, n_res_bloc, n_filters):
        """

        Args:
            n_out_conv_filters (int): number of convolutional filters (out_channels)
            n_res_bloc (int): number of residual layers
            n_filters (int): number of conv filters inside the res layer
            inside the resnet block
        """
        super().__init__()
        
        self.n_outconv_filters = n_out_conv_filters
        self.n_res_block = n_res_bloc
        self.n_inconv_filters = n_filters
        
        
        # Layer starts here
        self.conv = nn.LazyConv2d(out_channels=self.n_outconv_filters,
                                  stride=2, kernel_size=4, padding=1)
        self.act = nn.PReLU()
        # We need to add res blocks depending on n_res_bloc
        self.res_blocks_list = nn.ModuleList()
        for i in range(self.n_res_block):
            if i == 0:
                # First block needs to take self.conv as input
                self.res_blocks_list.append(ResidualBlock(self.n_outconv_filters,
                                                     out_channels=self.n_inconv_filters))
            else:
                self.res_blocks_list.append(ResidualBlock(in_channels=self.n_inconv_filters,
                                                     out_channels=self.n_inconv_filters))
        
        self.residual_blocks = nn.Sequential(*self.res_blocks_list)
    def forward(self, x):
            
        x = self.conv(x)
        #x = self.act(x)
        x = self.residual_blocks(x)
        return x
    
    
    
    
    
    
    
class Encoder(nn.Module):
    
    """ VAE Encoder"""
    
    def __init__(self, latent_dim, layer_list, version=None):
        
        """
        latent_dim: Dimension of the latent space
        version: None, or "vq": if the resnet + conv doesn't work,
        I might try doing Vector Quantized VAE
        layer_list = list of tuples in the form:
        (n_res_block, n_res_filters)
        
        
        """
        
        super().__init__()
        self.latent_dim = latent_dim
        self.layer_list = layer_list
        feature_layers = []
        
        for layer in self.layer_list:
            n_res_block, n_filters = layer
            feature_layers.append(
                ConvResBlock(n_out_conv_filters=n_filters, n_res_bloc=n_res_block,
                             n_filters=n_filters)
            )
        self.features = nn.Sequential(*feature_layers)
        
        self.flatten = nn.Flatten()
        self.mean = nn.LazyLinear(latent_dim)
        self.logvar = nn.LazyLinear(latent_dim)
        
        
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        mean = self.mean(x)
        logvar = self.logvar(x)
        return mean, logvar
    
    
    
    
    
    
    
class ConvTResBlock(nn.Module):
    
    def __init__(self, n_out_filters, n_res_block, n_in_filters, ksize=(4,4)):
        """

        Args:
            n_out_filters (int): number of conv filters
            n_res_block (int): number of res blocks
            n_in_filters (int): number of conv filters in res block
            ksize (tuple, optional): kernel size Defaults to (3,3).
        """
        super().__init__()
        self.n_out_filters = n_out_filters
        self.n_res_block = n_res_block
        self.n_in_filters = n_in_filters
        
        self.convT = nn.LazyConvTranspose2d(out_channels=n_out_filters,
                                            kernel_size=ksize, stride=2, padding=1)
        self.act1 = nn.PReLU()
        
        
        self.res_list = nn.ModuleList()
        for i in range(self.n_res_block):
            if i == 0:
                self.res_list.append(ResidualBlock(in_channels=n_out_filters,
                                                   out_channels=n_in_filters))
            else:
                self.res_list.append(ResidualBlock(in_channels=n_in_filters,
                                                   out_channels=n_in_filters))
                
                
                
        self.res_block = nn.Sequential(*self.res_list)
        
    def forward(self, x):
        
        x = self.convT(x)
        x = self.act1(x)
        x = self.res_block(x)
        return x
    
    
    
    
    
class Decoder(nn.Module):
    
    def __init__(self, target_shape, latend_dim, layer_list):
        """

        Args:
            target_shape : The shape of the final image
            latend_dim (int): 
            layer_list (list of tuples): layer lists 
            
            target shape could be (3, 64, 64)
            We have stride = 2
            We have len(layer_list) as #conv layers
            thus, spatial reduction is 64/(2 ^ len(layer_list))
            which is the spatial size of encoders final feature map
            So we start by basically unflattening
        """
        super().__init__()
        target_shape_side = target_shape[-1]
        self.startDim = target_shape_side // 2 ** len(layer_list)
        self.first_channels = layer_list[0][-1]
        
        # Unflatten
        self.predecoder = nn.Linear(latend_dim,
                                    self.first_channels * self.startDim * self.startDim)
        
        feature_layers = []
        for layer in layer_list:
            n_res_block, n_filters = layer
            feature_layers.append(ConvTResBlock(n_out_filters=n_filters,
                                                n_res_block=n_res_block,
                                                n_in_filters=n_filters))
        self.features = nn.Sequential(*feature_layers)
        self.decoder_output = nn.LazyConvTranspose2d(target_shape[0], 3, padding=1)
        self.act = nn.Sigmoid()
    
    def forward(self, x):
        x = self.predecoder(x)
        # Unflatten (reshape)
        x = x.view(x.shape[0],
                   self.first_channels,
                   self.startDim,
                   self.startDim)
        x = self.features(x)
        x = self.decoder_output(x)
        x = self.act(x)
        
        return x
    
    




class VAE(nn.Module):
    
    def __init__(self, encoder, decoder, laten_dim,
                 input_dim, n_basis=10, wandb_log=False,
                 block=False):
        
        super().__init__()
        # self.basis = None 
        self.wandb_log = wandb_log
        self.block = block
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = laten_dim
        self.input_dim = input_dim
        self.n_basis = n_basis
        self.register_buffer('basis', None)
        
    def save_model(self, path):
        torch.save(self.state_dict(),os.path.join(path,"VAE_model.pt"))
        torch.save(self.basis,os.path.join(path,"vae_basis.pt"))
        
    
    def init_basis(self,z,visualize=False):
        """Init the basis, used in the first forward pass
        
        The columns of the B (basis) matrix are updated as random latent representations
        
        @param z: Tensor, latent vectors
        @param visualize: bool, if True visualize tSNE representations of centroids
        
        """
        
        
        #pick n_basis random idx and set their rapresentations as columns
        
        idxs=torch.randint(0,z.shape[0],(self.n_basis,))
        
        self.basis = torch.zeros(self.latent_dim, self.n_basis)
        for i in range(self.n_basis):
            self.basis[:,i]=z[idxs[i]]
        #self.basis=torch.rand((self.latent_dim,self.n_basis))
        if visualize:
            
            
            fig,ax=plt.subplots()
            print(f"Computing t-SNE to visualize from {self.latent_dim} to 2 dim - this could take a while..")
            tsne=TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(self.basis.T)
            ax.scatter(tsne[:,0],tsne[:,1])
            
            if self.wandb_log:
                wandb.log({"init": wandb.Image(ax)})
            
            plt.show(block=self.block)
    
    
    def compute_similarity(self, z):
        """
        Compute the similarity between the latent representations z and the prototypes matrix B
        compute the similarity between each sample and the mean vector of the cluster
        
        @param z: Tensor, latent reprs
        
        return similarity, Tensor with shape BS,K
        """

        if self.basis is None:
            self.init_basis(z.detach())
        sim= z@self.basis.type_as(z)

        return sim

    
    
    
    def forward(self,x):
        
        """forward pass implementing the idea of this VAE
        
        @param: x data images
        
        return x_recon, z, soft_sim: Tuple of (Tensor,Tensor,Tensor)
        where x_recon are the reconstructed imgs, 
        z are the latent reprs and 
        soft_sim the softmax of the similarities
        
        
        """
        
        
        z_mean, z_log_var = self.encoder(x)
        
        self.q_z = torch.distributions.normal.Normal(z_mean,
                                                   torch.exp(0.5 * z_log_var))

        device = z_mean.device

        # sample z from it
        z = self.q_z.rsample().to(device)
        
        self.z = z
        
        #
        
        # reference N(0,1)
        #ref_mu = torch.zeros(z.shape[0], z.shape[-1])
        #ref_sigma = torch.ones(z.shape[0], z.shape[-1])

        #self.p_z = D.normal.Normal(ref_mu.to(device), ref_sigma.to(device))
        

        sim = self.compute_similarity(z)
        soft_sim = F.softmax(sim, dim=1)
        
        x = self.decoder(z)
        
        
        ## generate the posterior
        
        
        #self.p_z=self.generate_ref(z,soft_sim)
        
        self.soft_sim = soft_sim
        
        return x, z, soft_sim
                
          