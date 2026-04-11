
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from VAE.architecture import VAE, Encoder, Decoder
from VAE.loss import PerceptualLoss
import os
from sklearn.manifold import TSNE
import wandb



# def vae_loss_p(x, x_recon, z_mean, z_log_var, beta=1.0, lam=0.1):
#         recon_loss = F.mse_loss(x_recon, x, reduction="sum")
#         perc_loss = perceptual(x, x_recon)
#         kl_loss = -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
#         return recon_loss + lam * perc_loss + beta * kl_loss, recon_loss, kl_loss

def vae_loss(x, x_recon, z_mean, z_log_var, beta=1.0):
        recon_loss = F.mse_loss(x_recon, x, reduction="sum")
        # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        return recon_loss + beta * kl_loss, recon_loss, kl_loss


def run_epoch(model, loader, optimizer, device, train=True, beta=1.0):
        model.train() if train else model.eval()
        total_loss, total_recon, total_kl = 0, 0, 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                imgs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)

                x_recon, z, _, z_mean, z_log_var = model(imgs)
                #z_mean, z_log_var = model.encoder(imgs)  
                loss, recon, kl = vae_loss(imgs, x_recon, z_mean, z_log_var, beta)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                    optimizer.step()

                total_loss += loss.item()
                total_recon += recon.item()
                total_kl += kl.item()

        n = len(loader)
        return total_loss / n, total_recon / n, total_kl / n


def VAE_trainer(latent_dim,
                encoder_layer_list,
                decoder_layer_list,
                target_shape,
                train_loader,
                val_loader,
                epochs,
                lr,
                DEVICE,
                input_shape=None,
                SAVE=True
                ):
    
    encoder = Encoder(latent_dim=latent_dim, layer_list=encoder_layer_list)
    decoder = Decoder(target_shape=target_shape, latend_dim=latent_dim, layer_list=decoder_layer_list)
    model = VAE(encoder=encoder, decoder=decoder, laten_dim=latent_dim, input_dim=target_shape).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # perceptual = PerceptualLoss().to(DEVICE)

    train_losses, val_losses = [], []
    best_loss = float("inf")
    for epoch in range(epochs):
        beta = min(1.0, epoch / 40)
        train_loss, train_recon, train_kl = run_epoch(model, train_loader, optimizer, DEVICE, train=True, beta=beta)
        val_loss, val_recon, val_kl = run_epoch(model, val_loader, optimizer, DEVICE, train=False)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        l = val_loss + train_loss
        if l < best_loss and SAVE:
            best_loss = l
            model.save_model("val512_100epoch")
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")

        print(f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} (recon={train_recon:.4f}, kl={train_kl:.4f}) | "
            f"Val Loss: {val_loss:.4f} (recon={val_recon:.4f}, kl={val_kl:.4f})")
    #model.save_model("")

    # --- Plot ---
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.show()
    return model, train_losses, val_losses