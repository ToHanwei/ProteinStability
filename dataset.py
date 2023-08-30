import dataclasses
import gzip
import pickle
from functools import partial
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from Bio.PDB import FastMMCIFParser, PDBParser
from Bio.PDB.Polypeptide import three_to_index
from torch import Tensor
from torch.utils.data import Dataset


@dataclasses.dataclass
class Proteinbb:
    ca: Tensor
    cb: Tensor
    c: Tensor
    n: Tensor
    o: Tensor
    seq: Tensor
    resseq: Tensor
    chain_id: Tensor
    bb_ang: Tensor
    sasa: Tensor
    
    def __post_init__(self):
        self.ca = self._preprocess(self.ca)
        self.cb = self._preprocess(self.cb)
        self.c = self._preprocess(self.c)
        self.n = self._preprocess(self.n)
        self.o = self._preprocess(self.o)
        
        self.seq = self._preprocess(self.seq, is_atom=False)
        self.resseq = self._preprocess(self.resseq, is_atom=False)
        self.chain_id = self._preprocess(self.chain_id, is_atom=False)
        self.bb_ang = self._preprocess(self.bb_ang, is_atom=False)
        self.sasa = self._preprocess(self.sasa, is_atom=False)
    
    def _preprocess(self, x: List, is_atom: bool = True):
        x = Tensor(np.array(x))[:, None, ...]
        
        return x[:, None, ...] if is_atom else x


def init_empty_protein(num_res):
    init_atom = np.zeros((num_res, 3))
    
    return Proteinbb(
        ca = init_atom,
        cb = np.zer
    )
