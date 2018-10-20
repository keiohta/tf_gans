# TF-GANs
This repository supports following GAN implementations
- DCGAN: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- WGAN: [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
- WGAN-GP: [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
- LSGAN: [Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076)
- SNGAN: [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957)

# Dependencies
- python libraries
```bash
$ pip install requirements.txt
```

- others
  - [ffmpeg](https://www.ffmpeg.org/) to generate animation

# Examples
## MNIST
```bash
$ cd tf_gans
$ python run_mnist.py
```

# Visualization
This code automatically generates `results` directory and it contains a log for each experiment and includes TensorBoard log. So only you have to do is just call the `results` directory like following.

```bash
$ cd /path/to/results
$ tensorboard --logdir=results
```