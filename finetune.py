import os
import sys
import argparse
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from dataset.Pretrain import SequentialRecDataset, PretrainEmbeddingTable
import yaml
import pandas as pd
from models.ANT import ANT
from callback.MetricCallBack import ValidationMetricCallBack
from pytorch_lightning.callbacks import ModelCheckpoint
import logging
import pdb

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, required=True)
    parser.add_argument('--path', type=str, default='data')
    parser.add_argument('--config', type=str, default='props/ANT.yaml')
    parser.add_argument('--model', type=str, default='ANT')
    parser.add_argument('--image_model', type=str, default='SWIN')
    parser.add_argument('--fusion_lam', type=float, default=0.1)
    parser.add_argument('--image_dim', type=int, default=1024)
    parser.add_argument('--last_n_MoE', type=int, default=0)
    parser.add_argument('--n_exps_model', type=int, default=0)
    parser.add_argument('--mode', type=str, default='finetune_inductive')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--num_devices', type=int, default=1)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--strategy', type=str, default=None)
    parser.add_argument('--pretrained_model', type=str, required=True)
    parser.add_argument('--fixed_enc', type=str, required=True)
    parser.add_argument('--reinitialize_MoE', type=str, required=True)
    parser.add_argument('--n_layers_rec', type=int, default=2)
    parser.add_argument('--n_heads_rec', type=int, default=2)

    args = parser.parse_args()
    if args.num_nodes > 1 or args.num_devices > 1:
        args.strategy = 'ddp'

    args.num_sanity_val_steps = 2

    dataset = args.datasets
    args.datasets = args.datasets.split('_')

    if args.pretrained_model != '':
        args.last_n_MoE, args.n_exps_model = int(args.last_n_MoE), int(args.n_exps_model)
        if len(args.pretrained_model.split('_')) == 11:
            pretrained_datasets = '_'.join(args.pretrained_model.split('_')[:6])
        else:
            pretrained_datasets = '_'.join(args.pretrained_model.split('_')[:4])
        args.saveRoot = f'{pretrained_datasets}_{dataset}_{args.fusion_lam}_{args.last_n_MoE}_{args.n_exps_model}_{args.fixed_enc}_{args.reinitialize_MoE}_{args.n_layers_rec}_{args.n_heads_rec}'
    else:
        args.saveRoot = f'{dataset}_{args.fusion_lam}_{args.last_n_MoE}_{args.n_exps_model}_{args.fixed_enc}_{args.reinitialize_MoE}_{args.n_layers_rec}_{args.n_heads_rec}'

    args.path = f'{args.path}/{args.image_model}_BERT'
    if args.image_model != 'SWIN':
        args.image_dim = 768

    logName = os.path.join('results', args.mode, args.saveRoot)
    if os.path.exists(logName):
       os.remove(logName)
    logging.basicConfig(filename=logName, level=logging.INFO)
    ##logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    
    ftrain = os.path.join(args.path, dataset, f"{dataset}.train.inter")
    fvalid = os.path.join(args.path, dataset, f"{dataset}.valid.inter")
    ftest  = os.path.join(args.path, dataset, f"{dataset}.test.inter")

    train_data = SequentialRecDataset(args, [ftrain])
    valid_data = SequentialRecDataset(args, [fvalid])
    test_data  = SequentialRecDataset(args, [ftest])

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    config['fusion_lam'] = args.fusion_lam
    config['image_dim']  = args.image_dim
    config['last_n_MoE'] = args.last_n_MoE
    config['n_exps_model'] = args.n_exps_model
    config['training_stage'] = 'finetune'
    config['n_layers_rec'] = args.n_layers_rec
    config['n_heads_rec'] = args.n_heads_rec
    if args.mode == 'finetune_inductive' or args.mode == 'single_inductive':
        config['mode'] = 'inductive'
    elif args.mode == 'finetune_transductive' or args.mode == 'single_transductive':
        config['mode'] = 'transductive'
    else:
        config['mode'] = args.mode

    train_dataloader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, num_workers=1)
    valid_dataloader = DataLoader(valid_data, batch_size=config["batch_size"], shuffle=False, num_workers=1)
    test_dataloader  = DataLoader(test_data,  batch_size=config["batch_size"], shuffle=False, num_workers=1)

    if args.model == 'ANT':
        pm_embeddings = PretrainEmbeddingTable(args)
        model = ANT(config, pm_embeddings)
        if args.pretrained_model != '':
            model = model.load_from_checkpoint(f'best_model/pretrain/{args.pretrained_model}.ckpt', 
                    config=config, EmbeddingTable=pm_embeddings, strict=False)
        model.dataset_split = list(zip(args.datasets, test_data.dataset_split[1:]))

    with torch.no_grad():

        if args.pretrained_model != '':
            if args.fixed_enc == 'True':
                for x in model.trm_encoder.parameters():
                    x.requires_grad = False
                model.position_embedding.weight.data.normal_(mean=0.0, std=0.02)

            if args.reinitialize_MoE == 'True':

                for x in model.img2rec.proj:
                    x.bias.data.fill_(0)
                    x._init_weights(x.lin)

                for x in model.text2rec.proj:
                    x.bias.data.fill_(0)
                    x._init_weights(x.lin)

                for x in model.meta2rec.proj:
                    x._init_weights(x.lin)

                model.img_pooling.pooling.w_gate.fill_(0)
                model.img_pooling.pooling.w_noise.fill_(0)

                model.text_pooling.pooling.w_gate.fill_(0)
                model.text_pooling.pooling.w_noise.fill_(0)

                model.meta_pooling.pooling.w_gate.fill_(0)
                model.meta_pooling.pooling.w_noise.fill_(0)

    metric = ValidationMetricCallBack(args, logger)
    checkpoint = ModelCheckpoint(dirpath=f'best_model/{args.mode}', 
            filename=args.saveRoot,
            monitor='total_validation_r10', 
            mode='max')

    trainer = pl.Trainer(devices=args.num_devices, accelerator="gpu", 
            max_epochs=args.max_epochs, precision=16, num_sanity_val_steps=args.num_sanity_val_steps, 
            callbacks=[metric, checkpoint], 
            num_nodes=args.num_nodes,
            strategy=args.strategy)

    trainer.fit(model, train_dataloader, valid_dataloader)
    logged_metrics = trainer.test(model, test_dataloader, ckpt_path='best')[0]
    
    for dataset in args.datasets:
        r10, r50 = logged_metrics[f'{dataset}_test_r10'], logged_metrics[f'{dataset}_test_r50']
        n10, n50 = logged_metrics[f'{dataset}_test_n10'], logged_metrics[f'{dataset}_test_n50']
        logger.info("TEST, Dataset: %s, R10: %.4f, R50: %.4f, N10: %.4f, N50: %.4f" % (dataset, r10, r50, n10, n50))
    logger.info("TEST, Dataset: %s, R10: %.4f, R50: %.4f, N10: %.4f, N50: %.4f" % (dataset, r10, r50, n10, n50))

    print('Done')
