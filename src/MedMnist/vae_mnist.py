import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
# Data loaders
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

batch_size = 256
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(SimpleVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 14x14 -> 7x7
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 7x7 -> 7x7
            nn.LeakyReLU(0.2)
        )

        # Mean and variance for the latent space
        self.mu = nn.Linear(128 * 7 * 7, latent_dim)
        self.log_var = nn.Linear(128 * 7 * 7, latent_dim)

        # Decoder input layer
        self.decoder_input = nn.Linear(latent_dim, 128 * 7 * 7)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),  # 7x7 -> 7x7
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7x7 -> 14x14
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
            nn.Tanh()  # Output normalized between -1 and 1
        )

    def encode(self, x):
        # Encode the input
        h = self.encoder(x)
        h = h.view(h.size(0), -1)

        # Get mean and log variance
        mu = self.mu(h)
        log_var = self.log_var(h)

        return mu, log_var

    def reparameterize(self, mu, log_var):
        # Reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        # Decode the latent vector
        h = self.decoder_input(z)
        h = h.view(h.size(0), 128, 7, 7)
        return self.decoder(h)

    def forward(self, x):
        # Full forward pass
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mu, log_var, z


def mnist_vae_trainer(trainloader):
    '''
    This function will train a VAE on mnist data set. 
    Latent dim: 32
    Training epochs: 20
    '''
    latent_dim = 32  # Size of the latent space
    vae = SimpleVAE(latent_dim=latent_dim).to(device)

    # VAE training parameters
    vae_optimizer = optim.Adam(vae.parameters(), lr=0.001)
    vae_epochs = 20  # Just a few epochs to get a decent VAE

    def vae_loss_function(recon_x, x, mu, log_var):
        # Reconstruction loss (mean squared error)
        MSE = F.mse_loss(recon_x, x, reduction='sum')

        # KL divergence
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return MSE + KLD

    # Train the VAE
    print("Training the VAE...")
    for epoch in range(vae_epochs):
        vae.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(tqdm(trainloader, desc=f"VAE Epoch {epoch+1}/{vae_epochs}")):
            data = data.to(device)
            vae_optimizer.zero_grad()
            recon_batch, mu, log_var, _ = vae(data)
            loss = vae_loss_function(recon_batch, data, mu, log_var)
            loss.backward()
            vae_optimizer.step()
            train_loss += loss.item()

        print(f'VAE Epoch: {epoch+1}, Loss: {train_loss / len(trainloader.dataset):.6f}')
        torch.save(vae.state_dict(), 'models/mnist_vae.pth')


if __name__ == "__main__":
    mnist_vae_trainer(trainloader)
