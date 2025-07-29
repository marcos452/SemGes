# SemGes: Semantics-aware Co-Speech Gesture Generation using Semantic Coherence and Relevance Learning[ICCV 2025]
> 
> This official GitHub repository contains PyTorch implementation of the work presented above. 
> SemGes generates gesture animations based on raw audio input of speech sequences, text and speaker ID.

> We reccomend visiting the project [webpage]( https://semgesture.github.io/.) and watching the supplementary video.

> Paper [arxiv](https://www.arxiv.org/abs/2507.19359)

## ⚙️ Build Environment

```bash
# Step 1: Create and activate conda environment
conda create -n semges python=3.8
conda activate semges

# Step 2: Install Python dependencies
pip install -r requirements.txt
```

## 📁 Dataset Download


> [BEAT Datasets](https://data.vision.ee.ethz.ch/cvl/datasets/b3dac2.en.html) is available upon request for research or academic purposes and Beat Data Preparation and Data Pre-process Please follow this [link](https://github.com/Doubiiu/CodeTalker/tree/main/BIWI)

> [TED Datasets](https://data.vision.ee.ethz.ch/cvl/datasets/b3dac2.en.html) is available upon request for research or academic purposes and TED Datasets Preparation and Data Pre-process Please follow this [link](https://github.com/Doubiiu/CodeTalker/tree/main/BIWI)




### 🏋️‍♂️ Train VQVAE（First Stage）
python train.py -c configs/vqvae_hand.yaml
python train.py -c configs/vqvae_body.yaml



### 🏋️‍♂️ Train SemGes（Second Stage）
python train.py -c configs/latent_transformer.yaml

