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


class AttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim = 128,
        n_heads = 8,
        dropout = 0.2,
        n_neighbors = 32,
    ):
        super().__init__()
        # multihead layer
        self.multi_attn_layer = nn.MultiheadAttention(
            embed_dim = embed_dim,
            num_heads = n_heads,
            dropout = dropout,
        )
        # feed forword layer
        self.ffn_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        # layerNorm layer
        self.layer_norms = nn.ModuleList(
            nn.LayerNorm(embed_dim) 
            for _ in range(2)
        )
        self.n_neighbors = n_neighbors
        
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        x_0 = v
        x_1, _ = self.multi_attn_layer(q, k, v)
        x_2 = self.layer_norms[0](x_1 + x_0)
        x_3 = self.ffn_layer(x_2)
        x_4 = self.layer_norms[1](x_3 + x_2)
        
        return x_4


class AmpnnBlock(nn.Module): 
    def __init__(
        self,
        embed_dim = 128,
        n_heads = 8,
        n_layers = 6,
        dropout = 0.2,
        n_neighbors = 32
    ):
        super().__init__()
        # node attention layers
        self.node_attn_layers = nn.ModuleList([
            AttentionBlock(
                embed_dim = embed_dim, 
                n_heads = n_heads, 
                dropout = dropout, 
                n_neighbors = n_neighbors,
            ) for _ in range(n_layers)
        ])
        # edge attention layers
        self.edge_attn_layers = nn.ModuleList([
            AttentionBlock(
                embed_dim = embed_dim, 
                n_heads = n_heads, 
                dropout = dropout, 
                n_neighbors = n_neighbors,
            ) for _ in range(n_layers)
        ])
        
        # linear layer for cat(node, edge)
        # Concat(node_ebedding, edge_ebedding), so input dim is embed_dim * 2
        self.cat_layer_embed = nn.Linear(embed_dim * 2, embed_dim)

        self.n_neighbors = n_neighbors
    
    def forward(
        self, 
        h: Tensor, 
        e: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Node embedding are processed by AttentionBlocks
        for node_attn_layer in self.node_attn_layers:
            # repeat the embedding of the central atom
            q_h = h[0, :, :][None, ...].repeat(self.n_neighbors, 1, 1)
            k_h, v_h = h, h
            h = node_attn_layer(q_h, k_h, v_h)
        # edge embedding are processed by AttentionBlocks
        for edge_attn_layer in self.edge_attn_layers:
            q_e = e[0, :, :][None, ...].repeat(self.n_neighbors, 1, 1)
            k_e, v_e = e, e
            e = edge_attn_layer(q_e, k_e, v_e)
        
        # cat node&edge embedding
        m = self.cat_layer_embed(
            torch.cat([h, e], dim=-1)
        )
        
        return m, h, h


class CollectModel(nn.Module):
    def __init__(
        self,
        embed_dim = 128,
        edge_dim = 27,
        node_dim = 28,
        dropout = 0.2,
        n_layers = 6,
        n_tokens = 21,
        n_neighbors = 32,
        n_heads = 8,
    ):
        super().__init__()
        # linear layer for node
        self.init_node_embed = nn.Linear(node_dim, embed_dim, bias=False)
        # linear layer for edge
        self.init_edge_embed = nn.Linear(edge_dim, embed_dim, bias=False)
        # ampnn layers
        self.ampnns = nn.ModuleList(
            AmpnnBlock(
                embed_dim = embed_dim,
                n_heads = n_heads,
                n_layers = n_layers,
                dropout = dropout,
                n_neighbors = n_neighbors
            ) for _ in range(n_layers)
        )
        
        # attention layer
        self.attn_layer = AttentionBlock(
            embed_dim = embed_dim, 
            n_heads = n_heads, 
            dropout = dropout, 
            n_neighbors = n_neighbors,
        )
        
        self.lm_heads = nn.Linear(embed_dim, n_tokens)
        
    def forward(
        self, 
        node_feats: Tensor,
        edge_feats: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # node features embedding
        h = self.init_node_embed(node_feats)
        # edge features embedding
        e = self.init_edge_embed(edge_feats)
        
        x, y = h, e
        for ampnn in self.ampnns:
            # y already does not mean edge
            x, y, _ = ampnn(x, y)
        
        y_1 = self.attn_layer(y)
        y_2 = self.lm_heads(y_1.sum(dim=0))
        
        return y_2


class LiteModel(pl.LightningModule):
    def __init__(
        self,
        args,
        embed_dim = 128,
        edge_dim = 27,
        node_dim = 28,
        dropout = 0.2,
        n_layers = 6,
        n_tokens = 21,
        n_heads = 8,
        n_neighbors = 32,
        lr = 1e-3,
    ):
        super().__init__()
        self.lr = lr
        self.args = args

        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=21, top_k=1)
        self.valid_acc = torchmetrics.Accuracy(task='multiclass', num_classes=21, top_k=1)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=21, top_k=1)
        
        # ampnn layers
        self.collec_model = CollectModel(
            embed_dim = embed_dim,
            edge_dim = edge_dim,
            node_dim = node_dim,
            dropout = dropout,
            n_layers = n_layers,
            n_tokens = n_tokens,
            n_neighbors = n_neighbors,
            n_heads = n_heads,
        )
        
    
    def training_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        node, edge, y = batch
        
        y_hat = self.collec_model(node, edge)
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
        
        y_hat = self.collec_model(node, edge)
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
        
        y_hat = self.collec_model(node, edge)
        
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
    
    model = LiteModel(
        args,
        embed_dim = args.embed_dim,
        edge_dim = args.edge_dim,
        node_dim = args.node_dim,
        dropout = args.dropout,
        n_layers = args.n_layers,
        n_tokens = args.n_tokens,
        n_heads= args.n_heads,
        n_neighbors = args.n_neighbors,
        lr = args.lr,
    )
    
    logger = TensorBoardLogger(
        save_dir = os.getcwd(),
        name = 'logs',
        version = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'),
    )
    
    trainer = pl.Trainer(
        accelerator = 'gpu',
        devices = 2,
        precision = '16-mixed',
        max_epochs = 1,
        logger = logger,
        log_every_n_steps = 1,
        enable_model_summary = True,
    )
    
    # 这里用于训练的数据是处理后的结果，直接导入即可
    # 第一次训练请先执行：python dataset.py处理数据；
    # 然后，执行python model.py训练模型
    processed_data_path = args.train_data
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
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_tokens', type=int, default=21)
    parser.add_argument('--n_heads', type=float, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_neighbors', type=int, default=32)
    parser.add_argument('--train-batch-size', type=int, default=8)
    parser.add_argument('--valid-batch-size', type=int, default=32)
    parser.add_argument(
        '--train_data',
        type=str,
        help='training data path.default[./data/s669_AF_PDBs.pt]',
        default=os.path.join(os.getcwd(), 'data', 's669_AF_PDBs.pt')
    )
    
    args = parser.parse_args()
    
    main(args)
