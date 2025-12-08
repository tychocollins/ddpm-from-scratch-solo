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

**Day 1 — Monday Dec 9**  
- Typed full `diffusion.py` + `train_mnist.py` myself in **under 2 hours**  
- Fixed every bug, got real loss printing (~2.0)  
- **DDPM is ALIVE** — training loop 100% working on Mac MPS  
- Finally understood: we ONLY train the model to predict the noise we added  
- Commit: https://github.com/tychocollins/ddpm-from-scratch-solo/commit/...

Tomorrow: replace DummyModel with real U-Net → loss drops → actual digits appear
