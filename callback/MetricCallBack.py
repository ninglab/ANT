import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import os
import pdb

class ClassifierValidationMetricCallBack(Callback):
    def __init__(self, args, logger):
        self.datasets = args.datasets
        self.logger = logger
        self.counter = 0

    def on_validation_epoch_end(self, trainer, module):
        elogs = trainer.logged_metrics

        acc = elogs[f'Acc']
        self.logger.info("Epoch: %d, Acc: %.4f" % (self.counter, acc)) 

        self.logger.info("------------------------------------------------------------------")
        self.counter += 1

class ValidationMetricCallBack(Callback):
    def __init__(self, args, logger):
        self.datasets = args.datasets
        self.logger = logger
        self.counter = 0
    
    def on_validation_epoch_end(self, trainer, module):
        elogs = trainer.logged_metrics

        for dataset in self.datasets:
            r10, r50 = elogs[f'{dataset}_validation_r10'].item(), elogs[f'{dataset}_validation_r50'].item()
            n10, n50 = elogs[f'{dataset}_validation_n10'].item(), elogs[f'{dataset}_validation_n50'].item() 

            self.logger.info("Dataset: %s, Epoch: %d, R10: %.4f, R50: %.4f, N10: %.4f, N50: %.4f" % (dataset, self.counter, r10, r50, n10, n50))

        self.logger.info("------------------------------------------------------------------")
        self.counter += 1

class TrainMetricCallBack(Callback):
    def __init__(self, args, logger):
        self.datasets = args.datasets
        self.logger = logger
        self.counter = 0

    def on_train_epoch_end(self, trainer, module):
        elogs = trainer.logged_metrics
        self.logger.info("Epoch: %d, training loss: %.4f" % (self.counter, elogs["train_loss"]))
        self.counter += 1
