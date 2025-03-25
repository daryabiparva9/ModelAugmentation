import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def visualize_vae_reconstructions(vae, images, labels):
    """
    Visualize VAE reconstructions for 5 random examples from the test set
    Shows original images and their VAE reconstructions side by side
    """
    # Set VAE to evaluation mode
    vae.eval()

    # Select 5 random images from the batch
    random_indices = torch.randperm(len(images))[:5]
    selected_images = images[random_indices]
    selected_labels = labels[random_indices]

    # Get VAE reconstructions
    with torch.no_grad():
        reconstructed, mu, log_var, z = vae(selected_images)

    # Move tensors to CPU for visualization
    selected_images = selected_images.cpu()
    reconstructed = reconstructed.cpu()

    # Plot original and reconstructed images side by side
    fig, axes = plt.subplots(5, 2, figsize=(8, 10))

    for i in range(5):
        # Original image
        axes[i, 0].imshow(selected_images[i].squeeze().numpy(), cmap='gray')
        axes[i, 0].set_title(f"Original: Digit {selected_labels[i].item()}")
        axes[i, 0].axis('off')

        # Reconstructed image
        axes[i, 1].imshow(reconstructed[i].squeeze().numpy(), cmap='gray')
        axes[i, 1].set_title("VAE Reconstruction")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig('vae_reconstructions.png')
    plt.show()


def visualize_latent_space(vae, images, labels):
    """
    Create a 2D visualization of the latent space using PCA
    Color points by digit class
    """

    # Set VAE to evaluation mode
    vae.eval()

    # Encode images to get latent representations
    with torch.no_grad():
        mu, _ = vae.encode(images)
        latent_vectors = mu.cpu().numpy()

    # Apply PCA to reduce to 2 dimensions for visualization
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_vectors)

    # Plot the 2D latent space
    plt.figure(figsize=(10, 8))

    # Color by digit class
    for digit in range(10):
        idx = labels.numpy() == digit
        plt.scatter(latent_2d[idx, 0], latent_2d[idx, 1], label=f'Digit {digit}', alpha=0.6)

    plt.title('VAE Latent Space (2D PCA projection)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('vae_latent_space.png')
    plt.show()


def visualize_latent_space_interpolation(vae, images, labels, device, num_steps=10):
    """
    Generate interpolations between two digits in latent space
    """
    vae.eval()

    # Find indices of two different digits (e.g., 0 and 1)
    # We'll look for specific digits to make the interpolation more interesting
    digit_indices = {}
    for i, label in enumerate(labels):
        digit = label.item()
        if digit not in digit_indices:
            digit_indices[digit] = i

        # If we found at least two different digits, we can stop
        if len(digit_indices) >= 2:
            break

    # Select two different digits
    digits = list(digit_indices.keys())[:2]
    idx1, idx2 = digit_indices[digits[0]], digit_indices[digits[1]]

    # Get latent representations
    with torch.no_grad():
        mu1, _ = vae.encode(images[idx1].unsqueeze(0))
        mu2, _ = vae.encode(images[idx2].unsqueeze(0))

    interpolations = torch.zeros(num_steps, 1, 28, 28).to(device)

    for i in range(num_steps):
        # Linear interpolation between mu1 and mu2
        alpha = i / (num_steps - 1)
        mu_interp = (1 - alpha) * mu1 + alpha * mu2

        # Decode the interpolated latent vector
        interpolated_image = vae.decode(mu_interp)
        interpolations[i] = interpolated_image

    # Plot the interpolation
    plt.figure(figsize=(15, 3))

    # Plot original start image
    plt.subplot(1, num_steps + 2, 1)
    plt.imshow(images[idx1].cpu().squeeze().numpy(), cmap='gray')
    plt.title(f'Start: {digits[0]}')
    plt.axis('off')

    # Plot interpolations
    for i in range(num_steps):
        plt.subplot(1, num_steps + 2, i + 2)
        plt.imshow(interpolations[i].cpu().squeeze().numpy(), cmap='gray')
        plt.axis('off')

    # Plot original end image
    plt.subplot(1, num_steps + 2, num_steps + 2)
    plt.imshow(images[idx2].cpu().squeeze().numpy(), cmap='gray')
    plt.title(f'End: {digits[1]}')
    plt.axis('off')

    plt.suptitle(f'Latent Space Interpolation: Digit {digits[0]} to Digit {digits[1]}')
    plt.tight_layout()
    plt.savefig('vae_interpolation.png')
    plt.show()


def visualize_vae_generation(vae, device, latent_dim=32, grid_size=5):
    """
    Generate new digit images by sampling from the latent space
    """
    vae.eval()

    # Create a grid of random latent vectors
    num_samples = grid_size * grid_size

    # Sample random points from the latent space (standard normal distribution)
    with torch.no_grad():
        # Create random latent vectors
        random_latent = torch.randn(num_samples, latent_dim).to(device)

        # Decode the random latent vectors
        generated_images = vae.decode(random_latent)

    # Plot the generated images
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            axes[i, j].imshow(generated_images[idx].cpu().squeeze().numpy(), cmap='gray')
            axes[i, j].axis('off')

    plt.suptitle('Generated Digits from Random Latent Vectors')
    plt.tight_layout()
    plt.savefig('vae_generated_digits.png')
    plt.show()
