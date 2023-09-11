# ANT

## Dependency
torch=1.10.2

pytorch\_lightning=1.7.7 

recbole=1.1.1

## Dataset and Pre-trained model
For the data used for fine-tuning, due to the space limit, 
we uploaded the processed data in [Google Drive](https://drive.google.com/drive/folders/1UH7b2EkjthqLJrXdEyzHX-9O2hCbw71G?usp=sharing). 
Please download the processed data and put it under the "data/SWIN\_BERT/" folder to run scripts.
The data used for pre-training is too large even for Google Drive. Please contact me directly if you are interested in these data.
We uploaded the pre-trained recommendation model in "best\_model/pretrain/".

## Example
Please refer to the following example on how to fine tune the pre-trained recommendation model on a specific dataset (i.e., Scientific)

`python finetune.py --datasets=Scientific --fusion_lam=0.1 --max_epochs=100 --fixed_enc=True --reinitialize_MoE=True --pretrained_model=Clothing_Home_Movies_Food_0.1_0_0`

Please refer to the following example on how to pretrain the recommendation model on the 

`python pretrain.py --datasets=Clothing_Home_Movies_Food --training_stage=pretrain --fusion_lam=0.1 --max_epochs=300`
