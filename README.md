# ANT

## Dependency
torch=1.10.2
pytorch\_lightning=1.7.7 

## Dataset
Due to the space limit, we uploaded the processed dataset in [Google Drive](https://drive.google.com/drive/folders/1UH7b2EkjthqLJrXdEyzHX-9O2hCbw71G?usp=sharing)

## Example
Please refer to the following example on how to train and evaluate the model\\
`python finetune.py --datasets=Scientific --fusion_lam=0.1 --max_epochs=100 --fixed_enc=True --reinitialize_MoE=True --pretrained_model=Clothing_Home_Movies_Food_0.1_0_0`
