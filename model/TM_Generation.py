import math
import os
import random
import re
import pickle
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.BaseModel import BaseModel
from dataloader.DataBatcher import DataBatcher
from IPython import embed
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


class TM_Generation(BaseModel):
    def __init__(self, dataset, model_conf, device):
        super(TM_Generation, self).__init__(dataset, model_conf)
        self.dataset = dataset
        self.model_conf = model_conf
        self.device = device

        self.batch_size = model_conf['batch_size']
        self.test_batch_size = model_conf['test_batch_size']
        self.lr = model_conf['lr']
        self.decoder_lr = model_conf['decoder_lr']

        if self.decoder_lr == -1:
            self.decoder_lr = self.lr

        self.reg = model_conf['reg']
        self.demo = model_conf['demo']
        self.beam_size = model_conf['beam_size']

        self.decoder_num_layers = model_conf['decoder_num_layers']
        self.decoder_num_heads = model_conf['decoder_num_heads']
        self.num_net_vocab = len(dataset.netvocab2netidx)
        self.num_op_vocab = len(self.dataset.operator2idx)

        self.pretrained_path = model_conf['pretrained_path']
        self.model_weight_path = model_conf['model_weight_path']
        self.bert_conf = AutoConfig.from_pretrained(self.pretrained_path)
        self.bert_max_len = model_conf['bert_max_len']

        self.lr_schedule = model_conf['lr_schedule']
        self.warmup_rate = model_conf['warmup_rate']
        self.grad_scaler = GradScaler(enabled=model_conf.get('mp_enabled', False))
        self.accumulation_steps = model_conf.get('accumulation_steps', 1)
        self.max_grad_norm = model_conf.get('max_grad_norm', None)
        self.swa_warmup = model_conf.get('swa_warmup', -1)
        self.swa_state = {}

        self.build_model()

        if os.path.exists(self.model_weight_path):
            self.load_model_parameters(self.model_weight_path)

        if 'google/' in self.pretrained_path:
            #self.OP_tokenid = self.tokenizer.get_added_vocab()['[OP]'] # 30549
            self.OP_tokenid = self.tokenizer('[OP]')['input_ids'][2] # [1031, 6728, 1033]
        else:
            self.OP_tokenid = self.tokenizer('[OP]')['input_ids'][1]


        assert self.test_batch_size == 1 or self.beam_size == 1, "test_batch_size must be 1 if beam_size is bigger than 1"

    def build_model(self):
        # BERT Encoder
        self.bert = AutoModel.from_pretrained(self.pretrained_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_path)
        # Add New Tokens (장소, 명칭, 인물)
        new_tokens = '(가) (나) (다) (라) (마) (바) (사) (아) (자) (차) (카) (타) (파) (하) 리터 l 밀리리터 ml 킬로미터 km 미터 m 센티미터 cm kg 제곱센티미터 ㎠ 세제곱센티미터 제곱미터 세제곱미터 ㎡ ㎤ ㎥'.split(' ')
        # Add New Tokens for Encoding IMQ+NET using BERT
        new_tokens += list(self.dataset.netvocab2netidx.keys())
        # new_tokens = list(set(new_tokens))
        num_added_toks = self.tokenizer.add_tokens(new_tokens)
        self.bert.resize_token_embeddings(len(self.tokenizer))
        print(f'{num_added_toks} tokens are added!; {new_tokens}')

        # Transformer Decoder for Generating NET
        self.embedding = nn.Embedding(self.num_net_vocab, self.bert_conf.hidden_size)
        self.pos_encoder = PositionalEncoding(self.bert_conf.hidden_size)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.bert_conf.hidden_size, nhead=self.decoder_num_heads)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.decoder_num_layers)

        # Linear Layers for Generating NET and Classifying OP
        self.generate_net_layer = nn.Linear(self.bert_conf.hidden_size, self.num_net_vocab)
        self.identify_op_layer = nn.Linear(self.bert_conf.hidden_size, self.num_op_vocab)

        self.CE_loss = nn.CrossEntropyLoss(ignore_index=0)  # ignore_index [PAD]

        ############################## Optimizer parameter 그룹 설정 ######################################
        no_decay = ["bias", "LayerNorm.weight"]
        decoder_list = ['embedding', 'pos_encoder', 'decoder_layer', 'transformer_decoder']
        param_groups = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and not any(re.match(f"^{nd}\.", n) for nd in decoder_list)
                ],
                "weight_decay": self.reg,
                "lr": self.lr,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and not any(re.match(f"^{nd}\.", n) for nd in decoder_list)
                ],
                "weight_decay": 0.0,
                "lr": self.lr,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and any(re.match(f"^{nd}\.", n) for nd in decoder_list)
                ],
                "weight_decay": self.reg,
                "lr": self.decoder_lr,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and any(re.match(f"^{nd}\.", n) for nd in decoder_list)
                ],
                "weight_decay": 0.0,
                "lr": self.decoder_lr,
            },
        ]
        self.optimizer = AdamW(param_groups)
        ###########################################################################################

        self.to(self.device)

    # INPUT: IMQ // OUTPUT: NET
    def forward_NET(self, batch_question_id, batch_question_attention, batch_net_id, batch_net_attention):
        '''
        batch_question_id: (batch, seq_len)
        batch_question_attention: (batch, seq_len)
        batch_net_id: (batch, seq_len)
        batch_net_attention: (batch, seq_len)
        '''
        # Generate NET
        batch_imq = self.bert(batch_question_id, attention_mask=batch_question_attention)[0]  # (batch, seq_len, hidden_size)
        batch_imq = batch_imq.permute(1, 0, 2)  # (seq_len, batch, hidden_size)
        batch_net_id = batch_net_id.permute(1, 0)  # (seq_len, batch)

        tgt_mask = self.generate_square_subsequent_mask(batch_net_id.size(0)).to(self.device)  # Padding mask for NET
        tgt = self.embedding(batch_net_id)
        tgt = self.pos_encoder(tgt)

        decoder_output = self.transformer_decoder(
            tgt=tgt,         # (tgt_seq_len, batch, hidden_size) // Target NET to generate
            memory=batch_imq,         # (seq_len, batch, hidden_size) // Input IMQ
            tgt_mask=tgt_mask,        # (tgt_seq_len, tgt_seq_len) // to avoid looking at the future tokens (the ones on the right)
            tgt_key_padding_mask=~batch_net_attention.bool(),     # (batch, tgt_seq_len) // to avoid working on padding, Padding mask for NET (!!!!!!!! 1 for masking !!!!!!!!)
            memory_key_padding_mask=~batch_question_attention.bool()   # (batch, src_seq_len) // avoid looking on padding of the src, Padding mask for IMQ (!!!!!!!! 1 for masking !!!!!!!!)
        )
        decoder_output = decoder_output.permute(1, 0, 2)
        logits = self.generate_net_layer(decoder_output)  # (batch, seq_len, |V|)

        return logits

    # INPUT: IMQ+NET // OUTPUT: OP
    def forward_OP(self, batch_question_id, batch_question_attention, batch_question_type):
        '''
        batch_question_id: (batch, seq_len)
        batch_question_attention: (batch, seq_len)
        '''
        # Identify operators
        if 'roberta' in self.pretrained_path:
            batch_imq_net = self.bert(batch_question_id, attention_mask=batch_question_attention)[0]
        else:
            batch_imq_net = self.bert(batch_question_id, attention_mask=batch_question_attention, token_type_ids=batch_question_type)[0]
        logits = self.identify_op_layer(batch_imq_net)  # (batch, seq_len, |V|)

        return logits

    def train_model_per_batch(self, batch_imq_id, batch_imq_attention, batch_imq_net_id, batch_imq_net_attention, batch_imq_net_type, batch_op_id, batch_op_attention, batch_net_id, batch_net_attention, scheduler=None):

        # self.optimizer.zero_grad()

        # Numpy To Tensor
        batch_imq_id = torch.LongTensor(batch_imq_id).to(self.device)
        batch_imq_attention = torch.LongTensor(batch_imq_attention).to(self.device)
        batch_imq_net_id = torch.LongTensor(batch_imq_net_id).to(self.device)
        batch_imq_net_attention = torch.LongTensor(batch_imq_net_attention).to(self.device)
        batch_imq_net_type = torch.LongTensor(batch_imq_net_type).to(self.device)
        batch_net_id = torch.LongTensor(batch_net_id).to(self.device)
        batch_net_attention = torch.LongTensor(batch_net_attention).to(self.device)
        batch_op_id = torch.LongTensor(batch_op_id).to(self.device)
        batch_op_attention = torch.LongTensor(batch_op_attention).to(self.device)

        with torch.cuda.amp.autocast(self.grad_scaler.is_enabled()):
            # Model Forward
            # (batch, seq_len, |V|)
            logit_net = self.forward_NET(batch_imq_id, batch_imq_attention, batch_net_id, batch_net_attention)
            # (batch, seq_len, |V|)
            logit_op = self.forward_OP(batch_imq_net_id, batch_imq_net_attention, batch_imq_net_type)

            # Calculate Loss
            # Only for not [PAD]
            batch_net_id = batch_net_id[:, 1:]  # remove target [BOS]
            batch_net_attention = batch_net_attention[:, 1:]  # remove target [BOS]
            logit_net = logit_net[:, :-1, :]  # remove input [EOS]
            active_loss = batch_net_attention.reshape(-1) == 1
            active_batch_net_id = batch_net_id.reshape(-1)[active_loss]
            active_logit_net = logit_net.reshape(-1, logit_net.shape[-1])[active_loss]
            loss_net = self.CE_loss(active_logit_net, active_batch_net_id)

            # Only for op token
            active_loss = batch_op_attention.view(-1) == 1
            active_batch_op_id = batch_op_id.view(-1)[active_loss]
            active_logit_op = logit_op.view(-1, logit_op.shape[-1])[active_loss]
            if len(active_batch_op_id) == 0:
                loss_op = 0
            else:
                loss_op = self.CE_loss(active_logit_op, active_batch_op_id)

            loss = loss_net + loss_op

        loss = loss / self.accumulation_steps
        loss = self.grad_scaler.scale(loss)

        # Backward
        loss.backward()

        if (self.global_step + 1) % self.accumulation_steps == 0:
            if self.max_grad_norm is not None:
                self.grad_scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)

            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.optimizer.zero_grad()

        # Step
        # self.optimizer.step()
        if scheduler:
            scheduler.step()

        return loss

    def train_model(self, dataset, evaluator, early_stop, logger, config):
        exp_config = config['Experiment']
        # Set experimental configuration
        num_epochs = exp_config['num_epochs']
        print_step = exp_config['print_step']
        test_step = exp_config['test_step']
        test_from = exp_config['test_from']
        verbose = exp_config['verbose']
        self.log_dir = logger.log_dir
        self.global_step = 0
        start = time()

        # get data from dataset class
        full_imq_text = dataset.idx2IMQ
        full_net = dataset.idx2NET
        full_postfix = dataset.idx2postfix

        # linear learning rate scheduler
        scheduler = None
        if self.lr_schedule:
            if len(dataset.train_ids) % self.batch_size == 0:
                steps_per_epoch = len(dataset.train_ids) // self.batch_size
            else:
                steps_per_epoch = len(dataset.train_ids) // self.batch_size + 1
            scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                        num_warmup_steps=int(steps_per_epoch*self.warmup_rate*num_epochs),
                                                        num_training_steps=steps_per_epoch * num_epochs)
            print(f">>> Linear scheduling at {self.warmup_rate} : warm up {self.warmup_rate*num_epochs} epochs over {num_epochs} epochs")

        for epoch in range(1, num_epochs + 1):
            epoch_train_start = time()
            epoch_loss = 0.0
            batch_loader = DataBatcher(np.arange(len(dataset.train_ids)), batch_size=self.batch_size, shuffle=True)
            num_batches = len(batch_loader)
            # ======================== Train
            self.train()

            if epoch - 1 == self.swa_warmup:
                self.swa_init()

            for b, batch_idx in enumerate(tqdm(batch_loader, desc="train_model_per_batch", dynamic_ncols=True)):
                # get batch data
                batch_indices = dataset.train_ids[batch_idx]
                batch_imq_text = [full_imq_text[i] for i in batch_indices]
                batch_net_text = [full_net[i] for i in batch_indices]
                batch_postfix_text = [full_postfix[i] for i in batch_indices]

                # Tokenize IMQ
                batch_imq_token = self.tokenizer(batch_imq_text, padding=True, truncation=True, max_length=self.bert_max_len, return_tensors='np')
                batch_imq_id, batch_imq_attention = batch_imq_token['input_ids'], batch_imq_token['attention_mask']

                # Tokenize IMQ+NET
                # 민진: token_type_ids 추가! (CC 데이터 기준 Valid는 0.9833으로 같으나 Test는 0.9750 -> 0.9500으로 하락함;)
                batch_imq_net_token = self.tokenizer(batch_imq_text, batch_net_text, padding=True, truncation=True, max_length=self.bert_max_len, return_tensors='np')
                batch_imq_net_id, batch_imq_net_attention, batch_imq_net_type = batch_imq_net_token['input_ids'], batch_imq_net_token['attention_mask'], batch_imq_net_token['token_type_ids']
                batch_num_op = [net.split(' ').count('[OP]') for net in batch_net_text]

                # Get operator id, attention, token index in IMQ+NET
                batch_op = np.where(batch_imq_net_id == self.OP_tokenid)
                batch_op_token = [[] for _ in range(len(batch_idx))]
                for idx, i in enumerate(batch_op[0]):
                    batch_op_token[i].append(batch_op[1][idx])

                # Get only OP in NET, not IMQ
                batch_op_id = np.zeros(batch_imq_net_id.shape)
                for i, toks in enumerate(batch_op_token):
                    batch_op_token[i] = toks[-batch_num_op[i]:]
                    for j, tok in enumerate(toks):
                        raw_op = [op for op in batch_postfix_text[i].split(' ') if op.startswith('[OP')]
                        batch_op_id[i][tok] = dataset.operator2idx[raw_op[j]]

                batch_op_attention = np.zeros(batch_imq_net_id.shape)
                batch_op_attention[batch_op] = 1

                # Get NET id, attention
                batch_net_id = np.zeros(batch_imq_id.shape)
                batch_net_attention = np.zeros(batch_imq_id.shape)
                for i, net in enumerate(batch_net_text):
                    batch_net_id[i][0] = dataset.netvocab2netidx['[BOS]']  # Add [BOS]
                    batch_net_attention[i][0] = 1
                    for j, v in enumerate(net.split(' ')):
                        batch_net_id[i][j+1] = dataset.netvocab2netidx[v]
                        batch_net_attention[i][j+1] = 1
                    batch_net_id[i][j+2] = dataset.netvocab2netidx['[EOS]']  # Add [EOS] to the end of the target sent
                    batch_net_attention[i][j+2] = 1

                batch_loss = self.train_model_per_batch(batch_imq_id, batch_imq_attention, batch_imq_net_id, batch_imq_net_attention,
                                                        batch_imq_net_type, batch_op_id, batch_op_attention, batch_net_id, batch_net_attention,
                                                        scheduler)
                epoch_loss += batch_loss
                self.global_step += 1

                if verbose and (b + 1) % verbose == 0:
                    print('batch %d / %d loss = %.4f' % (b + 1, num_batches, batch_loss))

            epoch_train_time = time() - epoch_train_start

            epoch_info = ['epoch=%3d' % epoch, 'loss=%.3f' % (epoch_loss/num_batches), 'train time=%.2f' % epoch_train_time]

            # ======================== Valid
            if (epoch >= test_from and epoch % test_step == 0) or epoch == num_epochs:
                self.eval()
                epoch_eval_start = time()

                self.swa_step()
                self.swap_swa_params()

                valid_score = evaluator.evaluate(self, dataset)
                valid_score['train_loss'] = epoch_loss.item()
                valid_score_str = ['%s=%.4f' % (k, valid_score[k]) for k in valid_score]

                updated, should_stop = early_stop.step(valid_score, epoch)

                if should_stop:
                    logger.info('Early stop triggered.')
                    break
                else:
                    # save best parameters
                    if updated:
                        torch.save(self.state_dict(), os.path.join(self.log_dir, 'best_model.p'))
                        torch.save({
                            "epoch": epoch,
                            'swa_state': self.swa_state,
                            'global_step': self.global_step,
                            'rng_states': (torch.get_rng_state(), np.random.get_state(), random.getstate()),
                            'optim': self.optimizer.state_dict(),
                            'scaler': self.grad_scaler.state_dict(),
                            'scheduler': scheduler.state_dict() if scheduler else None,
                        }, os.path.join(self.log_dir, 'state.p'))

                    if not os.path.exists(os.path.join(self.log_dir, 'tokenizer')):
                        self.tokenizer.save_pretrained(os.path.join(self.log_dir, 'tokenizer'))
                    if not os.path.exists(os.path.join(self.log_dir, 'dataset.pkl')):
                        with open(os.path.join(self.log_dir, 'dataset.pkl'), 'wb') as f:
                            pickle.dump((dataset.netvocab2netidx, dataset.netidx2netvocab, dataset.operator2idx, dataset.idx2operator, dataset.templatetoken2idx, dataset.idx2templatetoken), f, protocol=4)

                self.swap_swa_params()

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

    # Genrate NET and Predict OP
    def predict(self, mode='valid', pf_converter=None):
        self.eval()
        with torch.no_grad():
            if mode == 'valid':
                input_ids = self.dataset.valid_ids
            elif 'test' in mode or mode == 'problemsheet':
                test_num = self.dataset.testsets.index(mode)
                input_ids = self.dataset.test_ids[test_num]
            elif mode == 'submit':
                input_ids = self.dataset.test_ids
            # get data from dataset class
            full_imq_text = self.dataset.idx2IMQ

            eval_answer = None
            eval_equation = []
            eval_loss = torch.zeros(len(input_ids))

            batch_size = self.test_batch_size
            batch_loader = DataBatcher(np.arange(len(input_ids)), batch_size=batch_size, drop_remain=False, shuffle=False)
            for b, (batch_idx) in enumerate(tqdm(batch_loader, desc=f'{mode}..', dynamic_ncols=True)):
                # get batch data
                batch_indices = input_ids[batch_idx]
                batch_imq_text = [full_imq_text[i] for i in batch_indices]

                # Tokenizer IMQ
                batch_imq_token = self.tokenizer(batch_imq_text, padding=True, truncation=True, max_length=self.bert_max_len, return_tensors='np')
                batch_imq_id, batch_imq_attention = batch_imq_token['input_ids'], batch_imq_token['attention_mask']
                # Numpy to Tensor
                batch_imq_id = torch.LongTensor(batch_imq_id).to(self.device)
                batch_imq_attention = torch.LongTensor(batch_imq_attention).to(self.device)

                # Get greedy input = Generate NET
                if self.beam_size == 1:
                    batch_pred_net_id = self.generate_net_greedy(batch_imq_id, batch_imq_attention)
                else:
                    batch_pred_net_id = self.generate_net_beam(batch_imq_id, batch_imq_attention)
                    batch_indices = np.array(batch_indices.tolist() * len(batch_pred_net_id))
                    batch_imq_text *= len(batch_pred_net_id)

                # Convert predicted NET index to NET vocab
                batch_pred_eq = [[] for _ in range(len(batch_indices))]
                for b, net_text in enumerate(batch_pred_net_id):
                    for v in net_text:
                        batch_pred_eq[b].append(self.dataset.netidx2netvocab[v])

                # Tokenize IMQ+Predicted_NET
                batch_pred_net_text = [' '.join(net) for net in batch_pred_eq]
                batch_imq_net_token = self.tokenizer(batch_imq_text, batch_pred_net_text, padding=True, truncation=True, max_length=self.bert_max_len, return_tensors='np')
                batch_pred_imq_net_id, batch_pred_imq_net_attention, batch_pred_imq_net_type = batch_imq_net_token[
                    'input_ids'], batch_imq_net_token['attention_mask'], batch_imq_net_token['token_type_ids']
                batch_pred_op_attention = np.zeros(batch_pred_imq_net_id.shape)
                batch_pred_op_attention[np.where(batch_pred_imq_net_id == self.OP_tokenid)] = 1
                # Numpy to Tensor
                batch_pred_imq_net_id = torch.LongTensor(batch_pred_imq_net_id).to(self.device)
                batch_pred_imq_net_attention = torch.LongTensor(batch_pred_imq_net_attention).to(self.device)
                batch_pred_imq_net_type = torch.LongTensor(batch_pred_imq_net_type).to(self.device)
                batch_pred_op_attention = torch.LongTensor(batch_pred_op_attention).to(self.device)
                # Predict OP
                logit_op = self.forward_OP(batch_pred_imq_net_id, batch_pred_imq_net_attention, batch_pred_imq_net_type)  # (batch, seq_len, # of op)

                # Convert predicted OP idx to OP vocab in the equation
                outputs_op = logit_op.argmax(axis=2).flatten()[batch_pred_op_attention.view(-1) == 1]
                idx = 0
                for b, toks in enumerate(batch_pred_eq):
                    for i, tok in enumerate(toks):
                        if tok == '[OP]':
                            batch_pred_eq[b][i] = self.dataset.idx2operator[outputs_op[idx].item()]
                            idx += 1
                        elif tok.startswith('[N') and self.dataset.idx2INC[batch_indices[b]].get(tok) is not None:
                            batch_pred_eq[b][i] = self.dataset.idx2INC[batch_indices[b]][tok]
                        elif tok.startswith('[C'):
                            batch_pred_eq[b][i] = tok[2:-1]  # [C999] -> 999
                        elif tok.startswith('[X') and self.dataset.idx2IXC[batch_indices[b]].get(tok) is not None:
                            batch_pred_eq[b][i] = self.dataset.idx2IXC[batch_indices[b]][tok]
                        elif tok.startswith('[E') and self.dataset.idx2IEC[batch_indices[b]].get(tok) is not None:
                            batch_pred_eq[b][i] = self.dataset.idx2IEC[batch_indices[b]][tok]
                        elif tok.startswith('[S') and self.dataset.idx2ISC[batch_indices[b]].get(tok) is not None:
                            batch_pred_eq[b][i] = self.dataset.idx2ISC[batch_indices[b]][tok]
                batch_pred_eq = [' '.join(eq) for eq in batch_pred_eq]

                # Check net integrity
                if self.beam_size > 1:
                    batch_pred_eq_checked = []     
                    for pred_eq in batch_pred_eq:
                        try:
                            pf_converter.convert(pred_eq)
                            batch_pred_eq_checked.append(pred_eq)
                            break
                        except:
                            continue                 
                    if batch_pred_eq_checked:
                        batch_pred_eq = [batch_pred_eq_checked[0]]
                    else:
                        batch_pred_eq = [batch_pred_eq[0]]

                # Concatenate
                eval_equation += batch_pred_eq
                # print("True:" ,[self.dataset.idx2postfix[input_ids[idx]] for idx in batch_idx])
                # print("Pred:" ,batch_pred_eq)
        return eval_answer, eval_equation, eval_loss.numpy()

    def generate_net_beam(self, batch_question_id, batch_question_attention, isEmbed=False):
        '''
        batch_question_id (batch, max_seq_len)
        batch_question_attention (batch, max_seq_len)
        '''
        bos_idx = self.dataset.netvocab2netidx['[BOS]']
        eos_idx = self.dataset.netvocab2netidx['[EOS]']

        pred_net = [[bos_idx] * self.beam_size for _ in range(batch_question_id.size(0))]
        batch_size, _ = batch_question_id.size()
        max_seq_len = 50
        for b in range(batch_size):
            # Get BERT hidden state
            self.eval()
            beam_size = self.beam_size
            batch_imq = self.bert(batch_question_id[b].unsqueeze(0), attention_mask=batch_question_attention[b].unsqueeze(0))[0]  # (1, seq_len, hidden_size)
            batch_imq = batch_imq.permute(1, 0, 2)  # (seq_len, 1, hidden_size)
            cumulative_probs = torch.Tensor([0] * beam_size)
            pred_net_beam = np.array([1]).repeat(beam_size*beam_size).reshape(-1, 1)
            candidate_net = []
            for i in range(max_seq_len):
                tgt = torch.LongTensor([pred_net[b]]).view(-1, beam_size).to(self.device)  # (cur_seq_len, beam)
                tgt_mask = self.generate_square_subsequent_mask(i+1).to(self.device)  # (cur_seq_len, cur_seq_len)
                tgt = self.embedding(tgt)  # (cur_seq_len, hidden) -> (cur_seq_len, beam, hidden_size)
                tgt = self.pos_encoder(tgt)  # (cur_seq_len, 1, hidden)
                decoder_output = self.transformer_decoder(
                    tgt=tgt,
                    memory=batch_imq.repeat(1, beam_size, 1),
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=~batch_question_attention[b].unsqueeze(0).repeat(beam_size, 1).bool()   # (1, src_seq_len) // avoid looking on padding of the src, Padding mask for IMQ
                )

                tmp_logit_net = self.generate_net_layer(decoder_output)  # (cur_seq_len, beam, hidden_size) -> (cur_seq_len, beam, |V|)
                logits = tmp_logit_net[-1, :, :]  # the last timestep # (1, beam, |V|)

                if i == 0:
                    logits[:, 2] = -np.inf
                values, indices = F.softmax(logits, dim=-1).topk(beam_size, dim=-1, sorted=False)
                indices = indices.cpu().reshape(-1, 1)
                if i == max_seq_len-1:
                    indices[:] = 2  # 2 * beam_size * beam_size
                    values[:] = 1

                pred_net_beam = np.hstack([pred_net_beam, indices])
                cumulative_probs = (cumulative_probs.view(-1, 1).repeat(1, beam_size) - (-np.log(values.cpu()))).flatten()

                if i == 0: ### [BOS] token
                    _, beam_search_idx = cumulative_probs[:beam_size].topk(beam_size)
                else:
                    _, beam_search_idx = cumulative_probs.topk(beam_size)

                if len(torch.where(indices == eos_idx)[0]) > 0:
                    top = 0
                    while top < beam_size:
                        top_idx = beam_search_idx[top]
                        if indices[top_idx].item() == eos_idx:
                            candidate_idx = beam_search_idx[top]
                            net = pred_net_beam[candidate_idx]
                            final_val = cumulative_probs[candidate_idx] / self._get_length_penalty(net.shape[0])
                            ## append top1
                            candidate_net.append((net, final_val))
                            top += 1
                        else:
                            break
                    temp_indices, _ = torch.where(indices == eos_idx)
                    cumulative_probs[temp_indices] = -np.inf
                    _, beam_search_idx = cumulative_probs.topk(beam_size)

                ## 끝나는 지점
                if len(candidate_net) > beam_size * 3 or i == max_seq_len-1:
                    if isEmbed:
                        embed()
                    pred_net_beam = [net_score[0].tolist()[1:-1] for net_score in sorted(candidate_net, key=(lambda x:x[1]), reverse=True)]
                    if pred_net_beam:
                        pred_net = pred_net_beam
                    else:
                        pred_net = []
                    return pred_net

                pred_net[b] = pred_net_beam[beam_search_idx].reshape(beam_size, -1).T ## (cur_seq_len + 1, beam_size)
                pred_net_beam = pred_net[b].repeat(beam_size, axis=-1).T
                cumulative_probs = cumulative_probs[beam_search_idx]

        return []

    def _get_length_penalty(self, length, alpha=0.5, min_length=5):
        return ((min_length + length) / (min_length+1)) ** alpha

    # INPUT: IMQ // OUTPUT: NET
    def generate_net_greedy(self, batch_question_id, batch_question_attention):
        '''
        batch_question_id (batch, max_seq_len)
        batch_question_attention (batch, max_seq_len)
        '''
        bos_idx = self.dataset.netvocab2netidx['[BOS]']
        eos_idx = self.dataset.netvocab2netidx['[EOS]']

        pred_net = [[bos_idx] for _ in range(batch_question_id.size(0))]
        batch_size, _ = batch_question_id.size()
        max_seq_len = 50

        for b in range(batch_size):
            # Get BERT hidden state
            self.eval()
            batch_imq = self.bert(batch_question_id[b].unsqueeze(0), attention_mask=batch_question_attention[b].unsqueeze(0))[0]  # (1, seq_len, hidden_size)
            batch_imq = batch_imq.permute(1, 0, 2)  # (seq_len, 1, hidden_size)

            for i in range(max_seq_len):
                tgt = torch.LongTensor([pred_net[b]]).view(-1, 1).to(self.device)  # (cur_seq_len, 1)
                tgt_mask = self.generate_square_subsequent_mask(i+1).to(self.device)  # (cur_seq_len, cur_seq_len)
                tgt = self.embedding(tgt)  # (cur_seq_len, 1, hidden)
                tgt = self.pos_encoder(tgt)  # (cur_seq_len, 1, hidden)

                decoder_output = self.transformer_decoder(
                    tgt=tgt,
                    memory=batch_imq,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=~batch_question_attention[b].unsqueeze(0).bool()   # (1, src_seq_len) // avoid looking on padding of the src, Padding mask for IMQ
                )

                tmp_logit_net = self.generate_net_layer(decoder_output)  # (cur_seq_len, 1, |V|)
                logits = tmp_logit_net[-1, :, :]  # the last timestep # (1, 1, |V|)

                indices = logits.argmax(dim=-1)  # (1, 1, 1)
                pred_net[b].append(indices.item())

                if indices.item() == eos_idx:  # break if end token appears
                    break
            pred_net[b] = pred_net[b][1:-1]  # append without [BOS], [EOS] token

        return pred_net

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def load_model_parameters(self, log_dir):
        # model
        with open(os.path.join(log_dir, 'best_model.p'), 'rb') as f:
            state_dict = torch.load(f)
        try:
            self.load_state_dict(state_dict, strict=False)
            print(f"Model loaded from {log_dir}.")
        except RuntimeError:
            del state_dict['bert.embeddings.word_embeddings.weight']
            del state_dict['embedding.weight']
            del state_dict['generate_net_layer.weight']
            del state_dict['generate_net_layer.bias']
            del state_dict['identify_op_layer.weight']
            del state_dict['identify_op_layer.bias']
            self.load_state_dict(state_dict, strict=False)
            print(f"Model loaded from {log_dir}. But bert.embeddings.word_embeddings, embedding, generate_net_layer, and identify_op_layer are not loaded!")

    def restore(self, log_dir):
        self.log_dir = log_dir
        # model
        with open(os.path.join(self.log_dir, 'best_model.p'), 'rb') as f:
            state_dict = torch.load(f)
        # Get NET vocab size and op vocab size and rebuild model
        self.num_net_vocab = state_dict['generate_net_layer.weight'].shape[0]
        self.num_op_vocab = state_dict['identify_op_layer.weight'].shape[0]
        self.build_model()
        # tokenizer
        self.tokenizer = self.tokenizer.from_pretrained(os.path.join(self.log_dir, 'tokenizer'))
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.load_state_dict(state_dict)
        if 'google/' in self.pretrained_path:
            #self.OP_tokenid = self.tokenizer.get_added_vocab()['[OP]'] # 30549
            self.OP_tokenid = self.tokenizer('[OP]')['input_ids'][2] # [1031, 6728, 1033]
        else:
            self.OP_tokenid = self.tokenizer('[OP]')['input_ids'][1]
        # Dataset
        with open(os.path.join(self.log_dir, 'dataset.pkl'), 'rb') as f:
            self.dataset.netvocab2netidx, self.dataset.netidx2netvocab, self.dataset.operator2idx, self.dataset.idx2operator, self.dataset.templatetoken2idx, self.dataset.idx2templatetoken = pickle.load(f)

    def swa_init(self) -> None:
        self.swa_state["models_num"] = 1
        for n, p in self.named_parameters():
            self.swa_state[n] = p.data.clone().detach()

    def swa_step(self) -> None:
        if not self.swa_state:
            return

        self.swa_state["models_num"] += 1
        beta = 1.0 / self.swa_state["models_num"]
        with torch.no_grad():
            for n, p in self.named_parameters():
                self.swa_state[n].mul_(1.0 - beta).add_(p.data, alpha=beta)

    def swap_swa_params(self) -> None:
        if not self.swa_state:
            return

        for n, p in self.named_parameters():
            p.data, self.swa_state[n] = self.swa_state[n], p.data


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
