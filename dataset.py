import dataclasses
import glob
import gzip
import os
import pickle
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from Bio.PDB import FastMMCIFParser, PDBParser
from Bio.PDB.Polypeptide import three_to_index
from Bio.PDB.Structure import Structure
from joblib import Parallel, delayed
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


@dataclasses.dataclass
class Proteinbb:
    ca: Tensor  # (NUM_RES, 1, 3)
    cb: Tensor  # (NUM_RES, 1, 3)
    c: Tensor  # (NUM_RES, 1, 3)
    n: Tensor  # (NUM_RES, 1, 3)
    o: Tensor  # (NUM_RES, 1, 3)
    
    seq: Tensor  # (NUM_RES, 1)
    resseq: Tensor  # (NUM_RES, 1)
    chain_id: Tensor  # (NUM_RES, 1)
    
    bb_ang: Tensor  # (NUM_RES, 6)
    sasa: Tensor  # (NUM_RES, 5)
    
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
        x = Tensor(np.array(x))
        
        return x[:, None, ...] if is_atom else x


def init_empty_protein(n_res: int = 32):
    """Initializes the backbone data
    Args:
        n_res (int): number of neighbor residues

    Returns:
        Proteinbb (Proteinbb): protein backbone dataset
    """
    init_atom = np.zeros((n_res, 3))
    init_others = torch.zeros((n_res, 1), dtype=torch.long)
    
    return Proteinbb(
        ca = init_atom,
        cb = init_atom,
        c = init_atom,
        n = init_atom,
        o = init_atom,
        resseq = init_others,
        seq = init_others,
        chain_id = init_others,
        bb_ang = torch.zeros((n_res, 6)),
        sasa = torch.zeros((n_res, 5))
    )


def read_pdb(pdb_file: str):
    """get data from PDB file

    Args:
        pdb_file (str): input file path (pdb, cif or pdb.gz format)

    Returns:
        Proteinbb (Proteinbb): protein backbone dataset
    """
    ca_list, cb_list, c_list, o_list, n_list = [], [], [], [], []
    resseq_list, seq_list, chain_id_list, bb_ang_list, sasa_list = [], [], [], [], []
    
    try:
        if pdb_file.endswith('.pdb'):
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure(None, pdb_file)[0]
        elif pdb_file.endswith('.cif'):
            parser = FastMMCIFParser(QUIET=True)
            structure = parser.get_structure(None, pdb_file)[0]
        elif pdb_file.endswith('.pdb.gz'):
            with gzip.open(pdb_file, 'rt') as f:
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure(None, f)[0]
    except:
        print(pdb_file)
    
    structure.atom_to_internal_coordinates()
    chain_dict = {}
    # backbone heavy atoms
    heavy_atoms = ['C', 'N', 'O', 'CA']
    
    for chain in structure.get_chains():
        if chain.id not in chain_dict:
            chain_dict[chain.id] = len(chain_dict)
        # structure chain index
        chain_id = chain_dict[chain.id]
        
        for residue in chain.get_residues():
            # residue.id[0] == ' ' mean "Classical residue"
            if all(atom in residue for atom in heavy_atoms) and residue.id[0] == ' ':
                try:
                    cb = residue['CB'].coord
                except:
                    # Handling Gly situations
                    b = residue['CA'].coord - residue['N'].coord
                    c_ = residue['C'].coord - residue['CA'].coord
                    a = np.cross(b, c_)
                    # Virtual CB coordinates
                    cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c_ + residue['CA'].coord
                
                ca_list.append(residue['CA'].coord)
                o_list.append(residue['O'].coord)
                c_list.append(residue['C'].coord)
                n_list.append(residue['N'].coord)
                cb_list.append(cb)
                
                chain_id_list.append([chain_id])
                resseq_list.append([residue.full_id[3][1]])
                
                try:
                    # residue to index
                    token_id = three_to_index(residue.get_resname())
                except:
                    # uncanonical amino acid
                    token_id = 20
                seq_list.append([token_id])
                
                # Get the backbone dihedral angle
                ric = residue.internal_coord
                phi = ric.get_angle('phi')
                psi = ric.get_angle('psi')
                omg = ric.get_angle('omg')
                
                phi = 0 if phi == None else phi
                psi = 0 if psi == None else psi
                omg = 0 if omg == None else omg
                
                bb_ang_list.append(np.concatenate([
                    np.sin(np.deg2rad([phi, psi, omg])),
                    np.cos(np.deg2rad([phi, psi, omg])),
                ]))
    
    return Proteinbb(
        ca = ca_list, 
        cb = cb_list, 
        c = c_list, 
        o = o_list, 
        n = n_list, 
        seq = seq_list, 
        resseq = resseq_list, 
        chain_id = chain_id_list, 
        bb_ang = bb_ang_list, 
        sasa = sasa_list,
    )
                

def get_neighbors(
    protbb: Proteinbb, 
    n_neighbors: int = 32,
    noise_level: float = 0.0,
    train: bool = False,
) -> Tuple[Tensor, Tensor, torch.LongTensor]:
    """get the features of the neighbor residues

    Args:
        protbb (Proteinbb): protein backbone dataset
        n_neighbors (int, optional): number of neighbors. Defaults to 32.
        noise_level (float, optional): backbone atomic coordinate noise. Defaults to 0.0.
        train (bool, optional): Defaults to False.
    """
    # number of residues
    n_res = len(protbb.ca)
    
    if n_res < n_neighbors:
        init_prot = init_empty_protein(n_neighbors)
        
        init_prot.ca[:n_res, :, :] = protbb.ca
        init_prot.cb[:n_res, :, :] = protbb.cb
        init_prot.c[:n_res, :, :] = protbb.c
        init_prot.o[:n_res, :, :] = protbb.o
        init_prot.n[:n_res, :, :] = protbb.n
        
        init_prot.seq[:n_res, :] = protbb.seq
        init_prot.resseq[:n_res, :] = protbb.resseq
        init_prot.chain_id[:n_res, :] = protbb.chain_id
        init_prot.bb_ang[:n_res, :] = protbb.bb_ang
        
        protbb = init_prot
        n_res = n_neighbors
    
    assert len(protbb.ca) == len(protbb.resseq)
    assert len(protbb.resseq) == len(protbb.seq)
    assert len(protbb.seq) == len(protbb.chain_id)
    
    # Backbone CA atomic interatomic distance
    dist = torch.sqrt(torch.sum(
        protbb.ca - protbb.ca.reshape(1, n_res, 3) ** 2,
        dim = -1,
    ))  # Pytorch broadcasting mechanism
    _, indices = torch.topk(dist, n_neighbors, largest=False)
    rel_pos = torch.clamp((protbb.resseq.T - protbb.resseq), min=-32, max=32)  # (NUM_RES, NUM_RES)
    rel_chains = (protbb.chain_id.T - protbb.chain_id).bool()  # (NUM_RES, NUM_RES)
    rel_chains = (~rel_chains).int()
    
    pos = torch.gather(rel_pos, dim=1, index=indices)  # (NUM_RES, NUM_NEIGHBORS)
    chain_id = torch.gather(rel_chains, dim=1, index=indices)
    seq_tokens = torch.gather(protbb.seq.T.repeat(n_res, 1), dim=1, index=indices)  # (NUM_RES, NUM_NEIGHBORS)
    
    if train:
        mask_prob = torch.rand(n_res)
        seq_tokens[:, 0] = torch.tensor([21] * n_res) * (mask_prob < 0.85).int() + \
                           torch.randint(0, 20, (n_res, )) * (mask_prob >= 0.85).int()
    else:
        seq_tokens[:, 0] = 21  # mask all for inference
    
    bb_ang = torch.gather(
        protbb.bb_ang[:, None, ...].repeat(1, n_res, 1),
        dim = 1,
        index = indices[..., None].repeat(1, 1, 6),
    )  # (NUM_RES, NUM_NEIGHBORS, 6)
    bb_coords = torch.cat([
        protbb.ca, protbb.cb, protbb.c, protbb.n, protbb.o
    ], dim = -2)  # (NUM_RES, 5, 3)
    
    if noise_level > 0:
        bb_coords += torch.rand_like(bb_coords) * noise_level  # noise disturbance
    
    dist_x = bb_coords[None, :, None, ...] - bb_coords[:, None, :, None, ...]  # (NUM_RES, NUM_RES, 5, 5, 3)
    dist = torch.sqrt(torch.sum(dist_x ** 2, dim=-1)).reshape(n_res, n_res, 25)  # (NUM_RES, NUM_RES, 5, 5) -> (NUM_RES, NUM_RES, 25)
    dist = torch.gather(dist, dim=1, index=indices[..., None].repeat(1, 1, 25))  # (NUM_RES, NUM_NEIGHBORS, 25)
    
    nodes = torch.cat([
        F.one_hot(seq_tokens.long(), num_classes=22),
        bb_ang
    ], dim = -1).transpose(1, 0)  # (NUM_RES, NUM_NEIGHBORS, 22 + 6)
    edge = torch.cat([
        dist,
        pos[..., None],
        chain_id[..., None],
    ], dim = -1).transpose(1, 0)  # (NUM_RES, NUM_NEIGHBORS, 25 + 1 + 1)
    
    return nodes, edge, protbb.seq.squeeze(-1).long()


def parallel_converter(pdb: str) -> Proteinbb:
    return read_pdb(pdb_file=pdb)


def save_all(all_pdbs: List[str]) -> None:
    # all_protbb = Parallel(n_jobs=-5)(
    #     delayed(parallel_converter)(pdb)
    #     for pdb in tqdm(all_pdbs)
    # )
    all_protbb = []
    
    for pdb in tqdm(all_pdbs):
        all_protbb.append(read_pdb(pdb))
    
    return all_protbb
    
    # for pdb, protbb in zip(all_pdbs, all_protbb):
    #     with open(os.path.join('./data', f'{pdb}.pkl'), 'wb') as f:
    #         pickle.dump(protbb, f)


class ProteinDataset(Dataset):
    def __init__(
        self,
        # data: List[List[Tensor, Tensor, torch.LongTensor]],
        protbbs: List[Proteinbb],
        # meta_batch_size = 2000,
        noise = 0.0,
        n_neighbors = 32,
    ):
        super().__init__()
        
        self.protbbs = protbbs
        self.noise = noise
        self.n_neighbors = n_neighbors
        
        # self.data = data
        
    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, torch.LongTensor]:
        # batch = self.protbbs[idx]
        # nodes, edges, targets = [], [], []
        
        # for protbb in batch:
        #     node, edge, target = get_neighbors(
        #         protbb,
        #         n_neighbors = self.n_neighbors,
        #         noise_level = self.noise,
        #     )
        #     map(lambda l, elem: l.append(elem), [nodes, edges, targets], [node, edge, target])
            
        # return (
        #     torch.cat(nodes, dim=1),
        #     torch.cat(edges, dim=1),
        #     torch.cat(targets).long(),
        # )
        protbb = self.protbbs[idx]
        
        node, edge, target = get_neighbors(
            protbb,
            n_neighbors = self.n_neighbors,
            noise_level = self.noise,
        )
        
        return node, edge, target
    
    def __len__(self):
        return len(self.protbbs)


def collat_fn(batch):
    nodes = torch.cat(
        [item[0] for item in batch],
        dim = 1,
    )
    edges = torch.cat(
        [item[1] for item in batch],
        dim = 1,
    )
    targets = torch.cat(
        [item[2] for item in batch],
    ).long()
    
    return nodes, edges, targets

            
class ProteinDataModule(pl.LightningDataModule):
    def __init__(
        self,
        args,
        data_dir: str,
        noise = 0.0,
        n_neighbors = 32,
    ):
        super().__init__()
        
        self.data_dir = data_dir
        self.args = args
        
        self.n_neighbors = n_neighbors
        self.noise = noise
    
    def setup(self, stage: Optional[str] = None) -> None:
        print(self.data_dir)
        self.data = torch.load(self.data_dir, map_location='cpu')
        
        dataset = ProteinDataset(
            self.data,
            n_neighbors = self.n_neighbors,
            noise = self.noise,
        )
        
        self.train_set, self.val_set, self.test_set = random_split(
            dataset, 
            lengths = [0.7, 0.2, 0.1],
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size = self.args.train_batch_size,
            shuffle = True,
            num_workers = 8,
            collate_fn = collat_fn, 
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size = self.args.valid_batch_size,
            num_workers = 8,
            collate_fn = collat_fn,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size = self.args.valid_batch_size,
            num_workers = 8,
            collate_fn = collat_fn,
        )


if __name__ == '__main__':
    # data_dir = '../Pythia_note/s669_AF_PDBs/'
    
    # pdb_filenames = glob.glob(data_dir + '*.pdb')
    # all_protbb = save_all(pdb_filenames)
    # torch.save(all_protbb, 's669_AF_PDBs.pt')
    
    data = []
    all_protbb = torch.load('s669_AF_PDBs.pt')
    print(type(all_protbb[0]))
    
    for protbb in tqdm(all_protbb):
        node, edge, target = get_neighbors(protbb, train=True)
        data.append([node, edge, target])
    
    torch.save(data, 'processed_data.pt')

    