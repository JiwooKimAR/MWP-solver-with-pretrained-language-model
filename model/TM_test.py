import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from time import time

from base.BaseModel import BaseModel
from dataloader.DataBatcher import DataBatcher
from tqdm import tqdm
from time import time
from IPython import embed

from transformers import BertConfig, AdamW
from scipy.sparse import csc_matrix

class TM_test(BaseModel):
    def __init__(self, dataset, model_conf, device):
        super(TM_test, self).__init__(dataset, model_conf)
        self.dataset = dataset
        self.model_conf = model_conf
        self.device = device

        self.batch_size = model_conf['batch_size']
        self.test_batch_size = model_conf['test_batch_size']
        self.lr = model_conf['lr']
        self.reg = model_conf['reg']
        self.demo = model_conf['demo']
        self.build_model()

    def build_model(self):

        self.to(self.device)

    def forward(self, bert_output, batch_target, compute_loss=False, avg_loss=True):
        
        return 0, 0

    def loss(self, recon, batch_target, avg_loss=True):
        pass
    def train_model_per_batch(self, bert_output, batch_target):

        self.optimizer.zero_grad()

        # model forward
        _, loss = self.forward(bert_output, batch_target, compute_loss=True, avg_loss=True)

        # backward
        loss.backward()

        # step
        self.optimizer.step()
        return loss
    
    def train_model(self, dataset, evaluator, early_stop, logger, config):
        exp_config = config['Experiment']

        num_epochs = exp_config['num_epochs']
        print_step = exp_config['print_step']
        test_step = exp_config['test_step']
        test_from = exp_config['test_from']
        verbose = exp_config['verbose']
        self.log_dir = logger.log_dir
        start = time()
        
        
        for epoch in range(1, num_epochs + 1):
            self.train()
            epoch_loss = 0.0
            batch_loader = DataBatcher(np.arange(len(dataset.train_ids)), batch_size=self.batch_size)
            num_batches = len(batch_loader)
            # ======================== Train
            epoch_train_start = time()
            for b, batch_idx in enumerate(tqdm(batch_loader, desc="train_model_per_batch")):
                pass
            epoch_train_time = time() - epoch_train_start

            epoch_info = ['epoch=%3d' % epoch, 'loss=%.3f' % (epoch_loss/num_batches), 'train time=%.2f' % epoch_train_time]

            # ======================== Valid
            if (epoch >= test_from and epoch % test_step == 0) or epoch == num_epochs:
                self.eval()
                # evaluate model
                epoch_eval_start = time()

                valid_score = evaluator.evaluate(self, dataset)
                valid_score_str = ['%s=%.4f' % (k, valid_score[k]) for k in valid_score]

                updated, should_stop = early_stop.step(valid_score, epoch)

                if should_stop:
                    logger.info('Early stop triggered.')
                    break
                else:
                    # save best parameters
                    if updated:
                        torch.save(self.state_dict(), os.path.join(self.log_dir, 'best_model.p'))

                epoch_eval_time = time() - epoch_eval_start
                epoch_time = epoch_train_time + epoch_eval_time

                epoch_info += ['epoch time=%.2f (%.2f + %.2f)' % (epoch_time, epoch_train_time, epoch_eval_time)]
                epoch_info += valid_score_str
            else:
                epoch_info += ['epoch time=%.2f (%.2f + 0.00)' % (epoch_train_time, epoch_train_time)]

            if epoch % print_step == 0:
                logger.info(', '.join(epoch_info))

        total_train_time = time() - start

        return early_stop.best_score, total_train_time


    def predict(self, mode='valid'):
        with torch.no_grad():
            if mode == 'valid':
                input_ids = self.dataset.valid_ids
            elif mode == 'test':
                input_ids = self.dataset.test_ids
        
        eval_equation = []
        for idx in input_ids:
            # for test purpose
            eval_equation.append(self.dataset.idx2postfix[idx])

        return None, eval_equation, np.array([0])

    def restore(self, log_dir):
        self.log_dir = log_dir
        with open(os.path.join(self.log_dir, 'best_model.p'), 'rb') as f:
            state_dict = torch.load(f)
        self.load_state_dict(state_dict)