# SemGes: Semantics-aware Co-Speech Gesture Generation using Semantic Coherence and Relevance Learning[ICCV 2025]

[webpage]( https://semgesture.github.io/.)
[arxiv](https://www.arxiv.org/abs/2507.19359)

## 🧾 Release Plans

- [x] Inference Code  
- [x] A web demo  
- [x] Training Code  
- [ ] Pretrained Models  


---

## ⚙️ Installation

### Build Environment

```bash
# Step 1: Create and activate conda environment
conda create -n semges python=3.8
conda activate semges

# Step 2: Install Python dependencies
pip install -r requirements.txt


### 🏋️‍♂️ Train VQVAE（First Stage）


# Train the shortcut RVQVAE model
python train.py -c configs/.yaml



### 🏋️‍♂️ Train SemGes（Second Stage）

# Train the diffusion model
python train.py -c configs/.yaml
