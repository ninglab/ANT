# Multi-modality Meets Re-learning: Mitigating Negative Transfer in Sequential Recommendation

This is the implementation of the $\mathtt{ANT}$ model in our paper (will be available in arxiv soon).

## Dependency

- python==3.9.13
- torch==1.10.2
- pytorch\_lightning==1.7.7 
- recbole==1.1.1

We encourage users to install the dependencies using `pip install`

## Processed Dataset
Our data processing scripts are built upon the scripts of [UniSRec](https://github.com/RUCAIBox/UniSRec). 
Due to the copyright consideration, we realse the processed dataset instead of the processing scripts.
If you are interested in the scripts, please send a request to Bo Peng (peng.707@osu.edu).

For the datasets used in evaluation, due to the space limit in GitHub, 
we release the processed data in [Google Drive](https://drive.google.com/drive/folders/1UH7b2EkjthqLJrXdEyzHX-9O2hCbw71G?usp=sharing).
Please download the processed data and put it under the "data/SWIN\_BERT/" folder to run the training scripts.

The data used for pre-training is too large even for Google Drive. Please contact Bo Peng directly if you are interested.

## Pre-trained Recommendation Model
We release our pre-trained recommendation model in the folder "best\_model/pretrain/".

## Adapting the Pre-trained Model
Please refer to the following example on how to adapt the pre-trained recommendation model on a target task (e.g., Scientific).

`python finetune.py --datasets=Scientific --fusion_lam=0.1 --max_epochs=100 --pretrained_model=Clothing_Home_Movies_Food_0.1_0_0`
<code>fusion\_lam</code> specifies the coefficien for modality fusion $\beta$.

## Model Pre-training
Please refer to the following example on how to pretrain the recommendation model

`python pretrain.py --datasets=Clothing_Home_Movies_Food --training_stage=pretrain --max_epochs=300`

## Acknowledgements
Our data processing scripts are built on the scripts of [UniSRec](https://github.com/RUCAIBox/UniSRec). Thanks for the great work!
