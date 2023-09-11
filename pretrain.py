import os
import sys
import argparse
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from dataset.Pretrain import PretrainSequentialRecDataset, PretrainEmbeddingTable, SequentialRecDataset
import yaml
import pandas as pd
from models.ANT import ANT
from callback.MetricCallBack import ValidationMetricCallBack, TrainMetricCallBack
from pytorch_lightning.callbacks import ModelCheckpoint
import logging
import pdb

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, required=True)
    parser.add_argument('--path', type=str, default='/fs/ess/PCON0041/Bo/Multi_Modality_recommendation/pretrained')
    parser.add_argument('--config', type=str, default='props/ANT.yaml')
    parser.add_argument('--model', type=str, default='ANT')
    parser.add_argument('--image_model', type=str, default='SWIN')
    parser.add_argument('--fusion_lam', type=float, default=0.1)
    parser.add_argument('--image_dim', type=int, default=1024)
    parser.add_argument('--last_n_MoE', type=int, default=0)
    parser.add_argument('--n_exps_model', type=int, default=0)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--num_devices', type=int, default=1)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--strategy', type=str, default=None)
    parser.add_argument('--training_stage', type=str, default='inductive_ft')
    parser.add_argument('--n_layers_rec', type=int, default=2)
    parser.add_argument('--n_heads_rec', type=int, default=2)

    args = parser.parse_args()
    if args.num_nodes > 1 or args.num_devices > 1:
        args.strategy = 'ddp'

    args.datasets = args.datasets.split('_')
    datasets = '_'.join(args.datasets)

    args.num_sanity_val_steps = 0 if len(args.datasets) > 1 else 2

    if args.last_n_MoE == 0:
        args.n_exps_model = 0

    args.saveRoot = f'{datasets}_{args.fusion_lam}_{args.last_n_MoE}_{args.n_exps_model}_{args.n_layers_rec}_{args.n_heads_rec}'

    args.path = f'{args.path}/{args.image_model}'
    if args.image_model != 'SWIN':
        args.image_dim = 768

    logName = os.path.join('results', args.training_stage, args.saveRoot)
    if os.path.exists(logName):
        os.remove(logName)

    logging.basicConfig(filename=logName, level=logging.INFO)
    ##logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()

    ftrains, fvalids, ftests = [], [], []

    for dataset in args.datasets:
        ftrain = os.path.join(args.path, dataset, f"{dataset}.train.inter")
        fvalid = os.path.join(args.path, dataset, f"{dataset}.valid.inter")
        ftest  = os.path.join(args.path, dataset, f"{dataset}.test.inter")

        ftrains.append(ftrain)
        fvalids.append(fvalid)
        ftests.append(ftest)

    if args.training_stage == 'pretrain':
        train_data = PretrainSequentialRecDataset(args, ftrains)
    else:
        train_data = SequentialRecDataset(args, ftrains)
        valid_data = SequentialRecDataset(args, fvalids)
        test_data = SequentialRecDataset(args, ftests)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    config['fusion_lam'] = args.fusion_lam
    config['image_dim']  = args.image_dim
    config['last_n_MoE'] = args.last_n_MoE
    config['n_exps_model'] = args.n_exps_model
    config['training_stage'] = args.training_stage
    config['n_layers_rec'] = args.n_layers_rec
    config['n_heads_rec'] = args.n_heads_rec
    config['mode'] = 'pretrain'

    if args.model == 'ANT':
        pm_embeddings = PretrainEmbeddingTable(args)
        if args.training_stage == 'pretrain':
            train_data.EmbeddingTable = pm_embeddings
        model = ANT(config, pm_embeddings)
        model.dataset_split = list(zip(args.datasets, train_data.dataset_split[1:]))

    train_dataloader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    if args.training_stage != 'pretrain':
        valid_dataloader = DataLoader(valid_data, batch_size=config["batch_size"], shuffle=False, num_workers=4)
        test_dataloader  = DataLoader(test_data,  batch_size=config["batch_size"], shuffle=False, num_workers=4)

    if args.training_stage != 'pretrain':
        metric = ValidationMetricCallBack(args, logger)
        checkpoint = ModelCheckpoint(dirpath=f'best_model/{args.training_stage}', 
                filename=args.saveRoot,
                monitor='total_validation_r10', 
                mode='max')
    else:
        metric = TrainMetricCallBack(args, logger)
        checkpoint = ModelCheckpoint(dirpath=f'best_model/{args.training_stage}',
                filename=args.saveRoot,
                monitor='train_loss',
                mode='min')

    trainer = pl.Trainer(devices=args.num_devices, accelerator="gpu", 
            max_epochs=args.max_epochs, precision=16, num_sanity_val_steps=args.num_sanity_val_steps, 
            callbacks=[metric, checkpoint], 
            num_nodes=args.num_nodes,
            strategy=args.strategy)
    
    if args.training_stage != 'pretrain':
        trainer.fit(model, train_dataloader, valid_dataloader)
    else:
        trainer.fit(model, train_dataloader)

    if args.training_stage != 'pretrain':
        logged_metrics = trainer.test(model, test_dataloader, ckpt_path='best')[0]

        for dataset in args.datasets:
            r10, r50 = logged_metrics[f'{dataset}_test_r10'], logged_metrics[f'{dataset}_test_r50']
            n10, n50 = logged_metrics[f'{dataset}_test_n10'], logged_metrics[f'{dataset}_test_n50']
            logger.info("TEST, Dataset: %s, R10: %.4f, R50: %.4f, N10: %.4f, N50: %.4f" % (dataset, r10, r50, n10, n50))

    print('Done')
