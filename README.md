## Variational Autoencoder (VAE) on MNIST

This project implements a simple Variational Autoencoder (VAE) in PyTorch, as described in [`cody-vae.ipynb`](cody-vae.ipynb), to reconstruct and generate MNIST digit images.

### Model Architecture

- **Encoder:**  
  The encoder consists of convolutional layers that process the input image and output two vectors: the mean (`mu`) and log-variance (`logvar`) of the latent distribution for each input. These vectors parameterize a Gaussian distribution in the latent space.

- **Reparameterization Trick:**  
  To allow backpropagation through the stochastic sampling, the model uses the reparameterization trick:  
  `z = mu + std * eps`, where `eps` is sampled from a standard normal distribution.

- **Decoder:**  
  The decoder is a series of transposed convolutional layers that map the latent vector `z` back to the image space, reconstructing the input.

### Training Method

- **Loss Function:**  
  The VAE is trained with a combination of:
  - **Reconstruction Loss:** Binary cross-entropy between the input and reconstructed image.
  - **KL Divergence:** Regularizes the latent space to follow a standard normal distribution.

- **KL Warmup:**  
  The KL term is gradually increased from 0 to 1 over several epochs (KL warmup), allowing the model to first focus on reconstruction before regularizing the latent space.

- **Optimization:**  
  - Adam optimizer is used.
  - Learning rate scheduling (`ReduceLROnPlateau`) and early stopping are implemented for robust convergence.
  - Model checkpoints are saved during training.

### Latent Space Interpolation Animations

The following GIFs demonstrate the VAE's ability to interpolate in latent space:

- **1 to 9 Animation:**  
  ![@anim_1_to_9.gif](anim_1_to_9.gif)  
  Interpolates between the latent representations of the digits 1 and 9, showing smooth transitions in generated images.

- **Pi Digits Animation:**  
  ![@anim_pi_digits.gif](anim_pi_digits.gif)  
  Traverses the latent space following a sequence of digits from the number π, visualizing the diversity and structure learned by the VAE.

For more details, see the code and comments in [`cody-vae.ipynb`](cody-vae.ipynb).
