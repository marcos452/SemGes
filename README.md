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


> [BEAT Datasets](https://pantomatrix.github.io/BEAT-Dataset/) is available upon request for research or academic purposes and Beat Data Preparation and Data Pre-process Please follow this [link](https://github.com/PantoMatrix/PantoMatrix/blob/main/datasets/process_testdata.py)

> [TED Datasets Expressive](https://mycuhk-my.sharepoint.com/personal/1155165198_link_cuhk_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F1155165198%5Flink%5Fcuhk%5Fedu%5Fhk%2FDocuments%2Fted%5Fexpressive%5Fdataset%2Ezip&parent=%2Fpersonal%2F1155165198%5Flink%5Fcuhk%5Fedu%5Fhk%2FDocuments&ga=1) is available upon request for research or academic purposes and TED Datasets Expressive Preparation and Data Pre-process Please follow this [link](https://github.com/alvinliu0/HA2G?tab=readme-ov-file)



## 🏋️‍♂️ Model Training


###  Train VQVAE（First Stage）
```commandline
python train.py -c configs/vqvae_hand.yaml
python train.py -c configs/vqvae_body.yaml
```


### Train SemGes（Second Stage）
```commandline
python train.py -c ./configs/latent_transformer.yaml
```

## 🤖 Inference
```commandline
python test_demo.py -c ./configs/latent_transformer_test.yaml
```


## 🙏 Acknowledgements
Thanks to [CAMN](https://pantomatrix.github.io/BEAT/), [EMAGE](https://pantomatrix.github.io/EMAGE/), our code is partially borrowing from them. Please check these useful repos.


## 📖 Citation

If you use this work in your research, please cite the following paper:

```bibtex
@inproceedings{liu2025semges,
  title     = {SemGes: Semantics-aware Co-Speech Gesture Generation using Semantic Coherence and Relevance Learning},
  author    = {Lanmiao Liu and Esam Ghaleb and Aslı Özyürek and Zerrin Yumak},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025},
  month     = oct,
  address   = {Honolulu, Hawai‘i, USA},
  publisher = {IEEE},
  url       = {https://semgesture.github.io/},
  note      = {To appear},
  series    = {ICCV '25'}
}



