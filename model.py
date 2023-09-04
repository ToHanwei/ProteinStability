import argparse
import datetime
import os
from argparse import ArgumentParser
from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
# 这里的Proteinbb这个类必须要导入，尽管没有显式调用
# 否则torch.load的时候pickle的反序列化操作会因为找不到Proteinbb这个class的定义而报错
from dataset import Proteinbb, ProteinDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor


class AMPNNLayer(nn.Module):
    def __init__(
        self,
        embed_dim = 128,
        n_heads = 8,
        dropout = 0.2,
        n_neighbors = 32, 
    ):
        super().__init__()
        # multihead message modules
        self.attn_message = nn.MultiheadAttention(
            embed_dim = embed_dim,
            num_heads = n_heads,
            dropout = dropout,
        )
        # feed forword message modules
        self.ffn_message = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        # multihead update modules
        self.attn_update = nn.MultiheadAttention(
            embed_dim = embed_dim,
            num_heads = n_heads,
            dropout = dropout,
        )
        # feed forword update modules
        self.ffn_update = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        # linear transition modules
        # Concat(node_feats, edge_feats), so input dim is embed_dim * 2
        self.message_transition = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.layer_norms = nn.ModuleList(
            nn.LayerNorm(embed_dim) 
            for _ in range(4)
        )
        self.n_neighbors = n_neighbors
        
    def forward(self, h_0: Tensor, e_0: Tensor) -> Tensor:
        """
            Args:
                h_0 (Tensor): The output of embd(node_feats)
                e_0 (Tensor): The output of embd(edge_feats)
        """
        h_1, _ = self.attn_message(
            h_0[0, :, :][None, ...].repeat(self.n_neighbors, 1, 1),  # central node
            h_0,
            h_0
        )
        h_1 = self.ffn_message(self.layer_norms[0](h_1))
        h_2 = self.layer_norms[1](h_1 + h_0)
        # concat & transition
        mess_t = self.message_transition(
            torch.cat([h_2, e_0], dim=-1)
        )
        
        h_3, _ = self.attn_update(mess_t, h_2, h_2)
        h_3 = self.ffn_update(self.layer_norms[2](h_3))
        h_4 = self.layer_norms[3](h_3 + h_2)
        
        return h_4


class AMPNN(nn.Module): 
    def __init__(
        self,
        embed_dim = 128,
        edge_dim = 27,
        node_dim = 28,  # 源码上这个默认参数是38，实际应是28，我们认为这是作者的小笔误，这里给修改成28
        n_heads = 8,
        n_layers = 3,
        n_tokens = 33,
        dropout = 0.2,
        n_neighbors = 32
    ):  
        super().__init__()
        # basic attributes
        self.embed_dim = embed_dim
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.n_layers = n_layers
        self.n_tokens = n_tokens
        self.dropout = dropout
        self.n_neighbors = n_neighbors
        # parameters
        self.init_node_embed = nn.Linear(node_dim, embed_dim, bias=False)
        self.init_edge_embed = nn.Linear(edge_dim, embed_dim, bias=False)
        self.layers = nn.ModuleList([
            AMPNNLayer(
                embed_dim = embed_dim, 
                n_heads = n_heads, 
                dropout = dropout, 
                n_neighbors = self.n_neighbors,
            ) for _ in range(n_layers)
        ])
        self.lm_heads = nn.Linear(embed_dim, n_tokens)
        self.layer_norms = nn.LayerNorm(embed_dim)
    
    def forward(
        self, 
        node_feats: Tensor, 
        edge_feats: Tensor
    ) -> Tuple[Tensor, Tensor]:
        h_0 = self.init_node_embed(node_feats)
        e_0 = self.init_edge_embed(edge_feats)
        
        for layer in self.layers:
            h_0 = layer(h_0, e_0)
        
        return self.lm_heads(h_0.sum(dim=0)), h_0


class LiteAMPNN(pl.LightningModule):
    def __init__(
        self,
        args,
        embed_dim = 128,
        edge_dim = 27,
        node_dim = 28,
        dropout = 0.2,
        n_layers = 3,
        n_tokens = 21,
        lr = 1e-3,
        n_neighbors = 32,
    ):
        super().__init__()
        
        self.ampnn = AMPNN(
            embed_dim = embed_dim,
            edge_dim = edge_dim,
            node_dim = node_dim,
            n_tokens = n_tokens,
            n_layers = n_layers,
            dropout = dropout,
            n_neighbors = n_neighbors
        )
        self.lr = lr
        self.args = args

        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=21, top_k=1)
        self.valid_acc = torchmetrics.Accuracy(task='multiclass', num_classes=21, top_k=1)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=21, top_k=1)
    
    def training_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        node, edge, y = batch
        
        y_hat, _ = self.ampnn(node, edge)
        # compute loss
        loss = F.cross_entropy(y_hat, y.squeeze(0))
        
        self.train_acc(y_hat, y.squeeze(0))
        self.log(
            'train_loss',
            loss,
            on_step = True,
            on_epoch = False,
            rank_zero_only = True,
            batch_size = self.args.train_batch_size,
        )
        self.log(
            'train_acc_step',
            self.train_acc,
            on_step = True,
            on_epoch = False,
            rank_zero_only = True,
            batch_size = self.args.train_batch_size,
        )
        
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        node, edge, y = batch
        
        y_hat, _ = self.ampnn(node, edge)
        # compute loss
        loss = F.cross_entropy(y_hat, y.squeeze(0))
        
        self.valid_acc(y_hat, y.squeeze(0))
        self.log(
            'valid_loss',
            loss,
            on_step = False,
            on_epoch = True,
            rank_zero_only = True,
            sync_dist = True,
            batch_size = self.args.valid_batch_size,
        )
        self.log(
            'valid_acc_step',
            self.valid_acc,
            on_step = False,
            on_epoch = True,
            rank_zero_only = True,
            sync_dist = True,
            batch_size = self.args.valid_batch_size,
        )
    
    def test_step(self, batch, batch_idx):
        node, edge, y = batch
        
        y_hat, _ = self.ampnn(node, edge)
        
        self.test_acc(y_hat, y.squeeze(0))
        self.log(
            'test_acc_step',
            self.test_acc,
            rank_zero_only = True,
            sync_dist = True,
            batch_size = self.args.valid_batch_size,
        )
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr = self.lr,
            weight_decay = 1e-4,
        )
        
        return [optimizer]


def main(args):
    for _ in range(5):
        torch.cuda.empty_cache()

    if torch.cuda.is_available():
        # if the CUDA device is A100 80GB
        # Set the matrix multiplication precision `high` rather than `highest`
        cuda_device_name = torch.cuda.get_device_properties(0).name
        if cuda_device_name == 'NVIDIA A100-SXM4-80GB':
            torch.set_float32_matmul_precision('high')
    
    model = LiteAMPNN(
        args,
        embed_dim = args.embed_dim,
        edge_dim = args.edge_dim,
        node_dim = args.node_dim,
        dropout = args.dropout,
        n_layers = args.n_layers,
        n_tokens = args.n_tokens,
    )
    
    logger = TensorBoardLogger(
        save_dir = os.getcwd(),
        name = 'logs',
        version = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'),
    )
    
    trainer = pl.Trainer(
        accelerator = 'gpu',
        devices = [1],
        precision = '16-mixed',
        max_epochs = 30,
        logger = logger,
        log_every_n_steps = 1,
        enable_model_summary = True,
    )
    
    # 这里用于训练的数据是处理后的结果，直接导入即可
    # 第一次训练请先执行：python dataset.py处理数据；
    # 然后，执行python model.py训练模型
    processed_data_path = os.path.join(os.getcwd(), 'data', 's669_AF_PDBs.pt')
    data_module = ProteinDataModule(
        args,
        processed_data = processed_data_path,
        noise = 0.0,
        n_neighbors = args.n_neighbors,
    )
    
    # start training process
    trainer.fit(
        model = model,
        datamodule = data_module,
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--edge_dim', type=int, default=27)
    parser.add_argument('--node_dim', type=int, default=28)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--n_tokens', type=int, default=21)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_neighbors', type=int, default=32)
    parser.add_argument('--train-batch-size', type=int, default=8)
    parser.add_argument('--valid-batch-size', type=int, default=32)
    
    args = parser.parse_args()
    
    main(args)
