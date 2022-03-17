import statistics
import timeit
import os
import logging
import pdb
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl
from sklearn import metrics


class Trainer():
    def __init__(self, params, graph_classifier, train, valid_evaluator=None):
        self.graph_classifier = graph_classifier
        self.valid_evaluator = valid_evaluator
        self.params = params
        self.train_data = train

        self.updates_counter = 0

        model_params = list(self.graph_classifier.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(model_params, lr=params.lr, momentum=params.momentum, weight_decay=self.params.l2)
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(model_params, lr=params.lr, weight_decay=self.params.l2)

        self.criterion = nn.MarginRankingLoss(self.params.margin, reduction='mean')
        self.b_xent = nn.BCEWithLogitsLoss()
        self.reset_training_state()
    
    def corrupt_graph(self, g):
        # Copy this graph firstly
        g_cor = dgl.DGLGraph(g)

        # nodes
        labels = g.node_attr_schemes()
        for l in labels.keys():
            g_cor.ndata[l] = g.ndata[l].clone().detach()

        # edges
        labels = g.edge_attr_schemes()
        for l in labels.keys():
            g_cor.edata[l] = g.edata[l].clone().detach()
        
        # Corruption
        # Try different corruption method:
        # 1. Replace one neighbor relation for each node.
        # graph_pos = data_pos[0]
        # rels_len = graph_pos.ndata['nei_rels'].shape[1]
        # num_nodes = graph_pos.ndata['nei_rels'].shape[0]
        # num_rels = self.params.num_rels + 1     # Include the "padding" relation.
        # rows = np.arange(num_nodes)
        # corrupted_cols = np.random.randint(rels_len, size=num_nodes)
        # corrupted_rels = torch.LongTensor(np.random.randint(num_rels, size=num_nodes)).to(self.params.device)
        # nei_rels = graph_pos.ndata['nei_rels'].clone()
        # nei_rels[rows, corrupted_cols] = corrupted_rels
        # graph_pos.ndata['nei_rels_cor'] = nei_rels

        # 2. Reshuffle all neighbor relations
        num_nodes = g_cor.ndata['nei_rels'].shape[0]
        g_cor.ndata['nei_rels'] = g_cor.ndata['nei_rels'][torch.randperm(num_nodes)]
        
        return g_cor


    def NCE_loss(self, ht_embs_pos, ht_embs_cor, r_label_embs):
        tmp = torch.cat([(ht_embs_pos * r_label_embs.repeat(1, self.params.num_gcn_layers)).sum(1).unsqueeze(1), (ht_embs_cor * r_label_embs.repeat(1, self.params.num_gcn_layers)).sum(1).unsqueeze(1)], dim=1)
        loss = nn.functional.log_softmax(tmp, dim=1)[:, 0]  # Only consider the pos data
        return  - (loss).sum()
        
    # def DGI_loss(self, g_pos, g_cor, r_label_embs):
    #     g_out = dgl.mean_nodes(g_pos, 'repr')
    #     g_out = g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim)

    #     return 
    def reset_training_state(self):
        self.best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0

    def train_epoch(self):
        total_loss = 0
        total_MI_loss = 0
        all_preds = []
        all_labels = []
        all_scores = []
        
        
        dataloader = DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)
        self.graph_classifier.train()
        model_params = list(self.graph_classifier.parameters())

        # Move the ent2rels to device
        # ent2rels = {k: torch.LongTensor(v).to(device=self.params.device) for k,v in self.train_data.ent2rels.items()}
        # self.graph_classifier.ent2rels = ent2rels
        
        for b_idx, batch in enumerate(dataloader):
            # print("batch:", b_idx)
            ts = time.time()
            # Input positive and negative graph 
            data_pos, targets_pos, data_neg, targets_neg, data_cor = self.params.move_batch_to_device(batch, self.params.device)
            self.optimizer.zero_grad()
            self.graph_classifier.train()
            score_pos, ht_embs_pos, s_G_pos, s_g_pos = self.graph_classifier(data_pos, is_return_emb=True)
            score_neg = self.graph_classifier(data_neg)

            loss = self.criterion(score_pos, score_neg.view(len(score_pos), -1).mean(dim=1), torch.Tensor([1]).to(device=self.params.device))
            print(f"loss: {loss}")
            # print("Pos time: ", time.time() - ts)

            dgi_loss = 0
            if self.params.coef_nce_loss:
                _, ht_embs_cor, _, s_g_cor = self.graph_classifier(data_cor, is_return_emb=True, cor_graph=True)

                # Calculate the NCE loss
                # Get the output relation embedding
                # r_emb_out = self.graph_classifier.r_emb_out
                # rel_labels = data_pos[1]
                # r_label_embs = r_emb_out[rel_labels]
                # nce_loss = self.NCE_loss(ht_embs_pos, ht_embs_cor, r_label_embs)
                
                # Calculate the DGI loss           
                lbl_1 = torch.ones(data_pos[0].batch_size)
                lbl_2 = torch.zeros(data_pos[0].batch_size)
                lbl = torch.cat((lbl_1, lbl_2)).to(self.params.device)
                logits = self.graph_classifier.get_logits(s_G_pos, s_g_pos, s_g_cor)
                dgi_loss = self.b_xent(logits, lbl)

                print(f'supervised loss: {loss}, NCE loss: {dgi_loss}')
                loss = loss + self.params.coef_nce_loss * dgi_loss
                # print("MI time: ", time.time() - ts)
                # print(f'Loss: {loss}')

            loss.backward()
            self.optimizer.step()
            self.updates_counter += 1
            # print("iter time:", time.time() - ts)
            with torch.no_grad():
                all_scores += score_pos.squeeze().detach().cpu().tolist() + score_neg.squeeze().detach().cpu().tolist()
                all_labels += targets_pos.tolist() + targets_neg.tolist()
                total_loss += loss
                total_MI_loss += dgi_loss

            if self.valid_evaluator and self.params.eval_every_iter and self.updates_counter % self.params.eval_every_iter == 0:
                tic = time.time()
                result = self.valid_evaluator.eval()
                logging.info('\nPerformance:' + str(result) + 'in ' + str(time.time() - tic))

                if result['auc'] >= self.best_metric:
                    self.save_classifier()
                    self.best_metric = result['auc']
                    self.not_improved_count = 0

                else:
                    self.not_improved_count += 1
                    if self.not_improved_count > self.params.early_stop:
                        logging.info(f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
                        break
                self.last_metric = result['auc']

        auc = metrics.roc_auc_score(all_labels, all_scores)
        auc_pr = metrics.average_precision_score(all_labels, all_scores)

        weight_norm = sum(map(lambda x: torch.norm(x), model_params))

        return total_loss, total_MI_loss, auc, auc_pr, weight_norm

    def train(self):
        self.reset_training_state()

        for epoch in range(1, self.params.num_epochs + 1):
            time_start = time.time()
            loss, MI_loss, auc, auc_pr, weight_norm = self.train_epoch()
            time_elapsed = time.time() - time_start
            logging.info(f'Epoch {epoch} with loss: {loss}, MI loss: {MI_loss}, training auc: {auc}, training auc_pr: {auc_pr}, best validation AUC: {self.best_metric}, weight_norm: {weight_norm} in {time_elapsed}')

            # if self.valid_evaluator and epoch % self.params.eval_every == 0:
            #     result = self.valid_evaluator.eval()
            #     logging.info('\nPerformance:' + str(result))
            
            #     if result['auc'] >= self.best_metric:
            #         self.save_classifier()
            #         self.best_metric = result['auc']
            #         self.not_improved_count = 0

            #     else:
            #         self.not_improved_count += 1
            #         if self.not_improved_count > self.params.early_stop:
            #             logging.info(f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
            #             break
            #     self.last_metric = result['auc']

            if epoch % self.params.save_every == 0:
                torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'graph_classifier_chk.pth'))

    def save_classifier(self):
        torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'best_graph_classifier.pth'))  # Does it overwrite or fuck with the existing file?
        logging.info('Better models found w.r.t accuracy. Saved it!')
