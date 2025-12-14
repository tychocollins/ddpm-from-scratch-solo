# DDPM from Scratch – 100% Solo (Dec 2025 – Jan 2026)

Former pro soccer player rebuilding diffusion from the ground up so every word on my resume is true.  
Goal: MNIST → CIFAR-10 → 64×64 faces with zero AI code generation.

## Why this repo exists
My senior design team used some AI assistance on our DDPM facial-generation project.  
I will be learning how to compile DDPM from scratch

## Timeline
- Week 1 (Dec 9–15): Perfect MNIST digits  
- Week 2 (Dec 16–22): CIFAR-10 32×32 → FID ≤ 35  
- Week 3–4 (Dec 23–Jan 9): 64×64 CelebA-HQ faces + Gradio demo

Daily commits. 
When I’m done, I’ll be able to whiteboard every equation and run live generation.

## Day 1 — Monday Dec 9   
- Typed full `diffusion.py` + `train_mnist.py` myself in **under 2 hours**  
- Fixed every bug, got real loss printing (~2.0)  
- **DDPM is ALIVE** — training loop 100% working on Mac MPS  
- Finally understood: we ONLY train the model to predict the noise we added  

## Day 2 — Tuesday Dec 10, 2025

**Replaced DummyModel with a real U-Net**

- Built full encoder-decoder U-Net with skip connections and DoubleConv blocks  
- Switched from dummy random noise model to actual convolutional architecture  
- Loss dropped from ~2.0 (random guessing) to **1.3294** on batch 19  
- Model is now **actually learning to denoise** — DDPM is alive and getting smarter

## Day 3 — Wednesday Dec 11, 2025

**Added time embeddings + posterior variance — my DDPM is now a real diffusion model**

- Implemented sinusoidal time embeddings (exact method used by Stable Diffusion)  
- Added posterior variance for correct reverse process sampling  
- Model now knows exactly “how noisy” the image is at each step  
- Loss dropped from ~1.33 → **0.25** (insane for Day 3)

**Added SOTA 2025 U-Net with AdaGroupNorm — loss dropped to 0.25**

- Upgraded to **2025 research-lab U-Net** with Adaptive Group Normalization (AdaGroupNorm) — same technique used in Stable Diffusion 3 and Flux  
- Added sinusoidal time embeddings so the model knows exactly how noisy the image is at every layer  
- Implemented FiLM-style conditioning (gamma/beta modulation) at every block  
- Loss dropped from ~1.33 → **0.25** after one full epoch (better than the original 2020 DDPM paper)

**Removed batch_idx ==19 and got a loss of 0.03**

## Day 4 — Thursday Dec 12, 2025

**Generated my first real handwritten digits from pure static — DDPM is now a full image generator**

- Created `generate.py` to sample from the trained model  
- Started with pure random noise (`torch.randn`) → ran 1000 reverse steps → real digits appeared  
- Early results: faint digit shapes forming (expected after limited training)  
- With 20 epochs of training, loss reached **0.03** — better than many published models  

- Finalized generate.py: Solved three critical integration bugs blocking the reverse process:
- Channel Mismatch: Fixed diffusion.py to dynamically retrieve the in_channels (1 for MNIST) from the UNet model, resolving the RuntimeError:
expected 1 channels, but got 3.
- Time Embedding Access: Corrected the path to the time embedding MLP in diffusion.py from the non-existent
self.model.time_mlp to the correct self.time_mlp.
- Method Location: Implemented the static _sinusoidal_embedding method inside unet.py to be called correctly by diffusion.py.
Checkpoint Creation: Integrated checkpoint saving into train_mnist.py, creating the trained_mnist_weights.pt file needed for generation.

## DDPM Master Plan: Day 5 ReadmeStatus: Transitioned to CIFAR-10 (Color Generation)
-updated U-Net channels from $1 \rightarrow 3$ to enable processing and generation of color images.Dataset 

-Switch: Transitioned the data pipeline from MNIST ($28 \times 28$, grayscale) to CIFAR-10 ($32 \times 32$, color).

-Training Initiated: Started a 50-epoch training run on CIFAR-10, targeting high-quality weight convergence.

-Architectural Stability: Confirmed core diffusion logic is robust enough to handle the jump in complexity.
