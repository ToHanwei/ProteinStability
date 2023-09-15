import argparse
import dataclasses
# import glob
import gzip
import os
# import pickle
import warnings
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from Bio.PDB import FastMMCIFParser, PDBParser
from Bio.PDB.Polypeptide import three_to_index
# from Bio.PDB.Structure import Structure
# from joblib import Parallel, delayed
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from Bio.PDB.DSSP import DSSP
from Bio.SeqUtils import seq1
from config import *


@dataclasses.dataclass
class Proteinbb:
    # Atoms
    ca: Tensor  # (NUM_RES, 1, 3)
    cb: Tensor  # (NUM_RES, 1, 3)
    c: Tensor  # (NUM_RES, 1, 3)
    n: Tensor  # (NUM_RES, 1, 3)
    o: Tensor  # (NUM_RES, 1, 3)
    
    # residue & chain
    seq: Tensor  # (NUM_RES, 1)
    resseq: Tensor  # (NUM_RES, 1)
    chain_id: Tensor  # (NUM_RES, 1)
    
    # residue properties
    dihe_ang: Tensor  # (NUM_RES, 8)
    charge: Tensor # (NUM_RES, 1)
    aroma: Tensor # (NUM_RES, 1)
    hydro: Tensor # (NUM_RES, 1)
    polar: Tensor # (NUM_RES, 1)
    size: Tensor # (NUM_RES, 1)
    rsa: Tensor # (NUM_RES, 1)
    ss: Tensor # (NUM_RES, 1)
    
    def __post_init__(self):
        self.ca = self._preprocess(self.ca, is_atom=True)
        self.cb = self._preprocess(self.cb, is_atom=True)
        self.c = self._preprocess(self.c, is_atom=True)
        self.n = self._preprocess(self.n, is_atom=True)
        self.o = self._preprocess(self.o, is_atom=True)
        
        self.seq = self._preprocess(self.seq)
        self.resseq = self._preprocess(self.resseq)
        self.chain_id = self._preprocess(self.chain_id)

        self.dihe_ang = self._preprocess(self.dihe_ang)
        self.charge = self._preprocess(self.charge)
        self.aroma = self._preprocess(self.aroma)
        self.hydro = self._preprocess(self.hydro)
        self.polar = self._preprocess(self.polar)
        self.size = self._preprocess(self.size)
        self.rsa = self._preprocess(self.rsa)
        self.ss = self._preprocess(self.ss)

    
    def _preprocess(self, x: List, is_atom: bool = False):
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
        dihe_ang = torch.zeros((n_res, 8)),
        charge = init_others,
        aroma = init_others,
        hydro = init_others,
        polar = init_others,
        size = init_others,
        rsa = init_others,
        ss = init_others,
    )


def get_rsa_ss(model, pdb_file, aads):
    """get RAS(relative solvent accessibility) and secondary structure from structure
    Args:
        model (Structure): structure object
        pdb_file (str): path of pdb file
        aads (list): amino acids of structure
    Returns:
        ras_list (list): three-state RAS. 0->Buried, 1->Intermediate, 2->Exposed
        ss_list (list): secondary structure index, 0-> helix, 1-sheet, 2->other
    """
    offset = 0
    ras_list, ss_list = [], []
    dssp = DSSP(model, pdb_file, dssp='mkdssp')
    for i, d in enumerate(dssp):
        _, aad, ss, rsa = d[:4]
        # three-state RSA
        state = rsa_to_three_state(rsa)
        # secondary structure index
        ss_index = ss_to_index.get(ss, 2)
        while aad != aads[i+offset]:
            # absence or abnormality of residues
            ras_list.append([0])
            ss_list.append([2])
            offset += 1
        ras_list.append([state])
        ss_list.append([ss_index])
    assert len(ras_list) == len(ss_list)
    return ras_list, ss_list


def read_pdb(pdb_file: str):
    """read PDB format data

    Args:
        pdb_file (str): input file path (pdb, cif or pdb.gz format).

    Returns:
        structure : protein structure object.
    """
    
    structure = None
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
        else:
            extension = os.path.splitext(pdb_file)[-1]
            warnings.warn(
                message=f"{extension} format is not supported.",
                category=None,
            )
    except:
        warnings.warn(
                    message=f"Failed to read structure {pdb_file}.",
                    category=None,
        )
    return structure


def calculator(pdb_file: str):
    """get data from PDB file

    Args:
        pdb_file (str): input file path (pdb, cif or pdb.gz format)

    Returns:
        Proteinbb (Proteinbb): protein backbone dataset
    """
    
    # list of side-chain atom
    cb_list = []
    # list of backbone atom
    ca_list, c_list, o_list, n_list = [], [], [], []
    # list of resudice
    resseq_list, seq_list = [], []
    # list of chain identity
    chain_id_list = []
    # dihedral angle
    ang_list = []
    # properties of residue
    charge_list, aroma_list, hydro_list, polar_list, size_list = [], [], [], [], []
    
    one_letters = []

    structure = read_pdb(pdb_file)
    if structure:
        structure.atom_to_internal_coordinates()
        chain_dict = {}
        # backbone heavy atoms
        heavy_atoms = ['C', 'N', 'O', 'CA']
        
        for chain in structure.get_chains():
            if chain.id not in chain_dict:
                # chain name to chain index
                chain_dict[chain.id] = len(chain_dict)
            # structure chain index
            chain_id = chain_dict[chain.id]
            
            for residue in chain.get_residues():
                res_name = residue.get_resname()
                res_charge = residue_to_charge.get(res_name, 0)
                res_aroma = residue_to_aromatic.get(res_name, 0)
                res_hydro = residue_to_hydrophobic.get(res_name, 0)
                res_polar = residue_to_polarity.get(res_name, 0)
                res_size = residue_to_size.get(res_name, 0)
                charge_list.append([res_charge])
                aroma_list.append([res_aroma])
                hydro_list.append([res_hydro])
                polar_list.append([res_polar])
                size_list.append([res_size])
                one_letters.append(seq1(res_name))
                # determine the integrity of the backbone atoms
                isBBcomplete = all(atom in residue for atom in heavy_atoms)
                # residue.id[0] == ' ' mean "Classical residue"
                if isBBcomplete and residue.id[0] == ' ':
                    ca = residue['CA'].coord
                    c = residue['C'].coord
                    n = residue['N'].coord
                    o = residue['O'].coord
                    try:
                        cb = residue['CB'].coord
                    except:
                        # Handling Gly situations
                        m, n = ca - n, c - ca
                        q = np.cross(m, n)
                        # Virtual CB coordinates
                        cb = (
                            -0.58273431 * q
                            + 0.56802827 * m 
                            - 0.54067466 * n 
                            + residue['CA'].coord
                            )
                    
                    ca_list.append(ca)
                    cb_list.append(cb)
                    o_list.append(o)
                    c_list.append(c)
                    n_list.append(n)
                    
                    chain_id_list.append([chain_id])
                    resseq_list.append([residue.full_id[3][1]])
                    
                    try:
                        # residue to index
                        token_id = three_to_index(residue.get_resname())
                    except:
                        # uncanonical amino acid
                        token_id = 20
                    seq_list.append([token_id])
                    
                    # Get the dihedral angle
                    ric = residue.internal_coord
                    phi = ric.get_angle('phi')
                    psi = ric.get_angle('psi')
                    omg = ric.get_angle('omg')
                    chi1 = ric.get_angle('chi1')
                    
                    phi = 0 if phi == None else phi
                    psi = 0 if psi == None else psi
                    omg = 0 if omg == None else omg
                    chi1 = 0 if chi1 == None else chi1
                    
                    ang_list.append(np.concatenate([
                        np.sin(np.deg2rad([phi, psi, omg, chi1])),
                        np.cos(np.deg2rad([phi, psi, omg, chi1])),
                    ]))
        # get three-state RSA, and secondary structure index
        rsa_list, ss_list = get_rsa_ss(structure, pdb_file, one_letters)
        assert len(rsa_list) == len(ca_list)
    else:
        warnings.warn(
                    message=f"Failed to calculate structure {pdb_file}.",
                    category=None,
        )
    
    return Proteinbb(
        ca = ca_list, 
        cb = cb_list, 
        c = c_list, 
        o = o_list, 
        n = n_list, 
        seq = seq_list, 
        resseq = resseq_list, 
        chain_id = chain_id_list, 
        dihe_ang = ang_list, 
        charge = charge_list,
        aroma = aroma_list,
        hydro = hydro_list,
        polar = polar_list,
        size = size_list,
        rsa = rsa_list,
        ss = ss_list,
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
        init_prot.dihe_ang[:n_res, :] = protbb.dihe_ang
        init_prot.charge[:n_res, :] = protbb.charge
        init_prot.aroma[:n_res, :] = protbb.aroma
        init_prot.hydro[:n_res, :] = protbb.hydro
        init_prot.polar[:n_res, :] = protbb.polar
        init_prot.size[:n_res, :] = protbb.size
        init_prot.rsa[:n_res, :] = protbb.rsa
        init_prot.ss[:n_res, :] = protbb.ss
        
        protbb = init_prot
        n_res = n_neighbors
    
    assert len(protbb.ca) == len(protbb.resseq)
    assert len(protbb.resseq) == len(protbb.seq)
    assert len(protbb.seq) == len(protbb.chain_id)
    assert len(protbb.chain_id) == len(protbb.charge)
    assert len(protbb.charge) == len(protbb.rsa)
    
    # Backbone CA atomic interatomic distance
    # ca_dist = torch.sqrt(torch.sum(
    #     input=protbb.ca - protbb.ca.reshape(1, n_res, 3) ** 2,
    #     dim=-1,
    # ))  # Pytorch broadcasting mechanism
    
    # Backbone CA atomic interatomic distance
    ca_dist = torch.cdist(
        x1=protbb.ca.squeeze(1), 
        x2=protbb.ca.squeeze(1), 
        p=2
    )
    # index of the neighbor residue
    _, indices=torch.topk(
        input=ca_dist, 
        k=n_neighbors, 
        largest=False
    ) # (NUM_RES, NUM_NEIGHBOR)
    
    ### node features ###

    # amino acid type torken
    seq_tokens = torch.gather(
        input=protbb.seq.T.repeat(n_res, 1),
        dim=1, 
        index=indices
    )  # (NUM_RES, NUM_NEIGHBORS)
    
    if train:
        mask_prob = torch.rand(n_res)
        # 中心残基85%概率被MASK，15%概率被突变
        seq_tokens[:, 0] = torch.tensor([21] * n_res) * (mask_prob < 0.85).int() + \
                           torch.randint(0, 20, (n_res, )) * (mask_prob >= 0.85).int()
    else:
        # mask all for inference
        seq_tokens[:, 0] = 21
    
    # 这里的维度扩张似乎没有意义，只是为了确保shape上的统一
    dihe_ang = torch.gather(
        input=protbb.dihe_ang[:, None, ...].repeat(1, n_res, 1),
        dim=1,
        index=indices[..., None].repeat(1, 1, 8),
    )  # (NUM_RES, NUM_NEIGHBORS, 8)
    # residue charge propertie
    charges = torch.gather(
        input=protbb.charge.repeat(1, n_res),
        dim=1,
        index=indices,
    )[..., None] # (NUM_RES, NUM_NEIGHBORS, 1)
    # residue aromatic propertie
    aromas = torch.gather(
        input=protbb.aroma.repeat(1, n_res),
        dim=1,
        index=indices,
    )[..., None] # (NUM_RES, NUM_NEIGHBORS, 1)
    # residue hydrophobicity propertie
    hydros = torch.gather(
        input=protbb.hydro.repeat(1, n_res),
        dim=1,
        index=indices,
    )[..., None] # (NUM_RES, NUM_NEIGHBORS, 1)
    # residue polarity propertie
    polars = torch.gather(
        input=protbb.polar.repeat(1, n_res),
        dim=1,
        index=indices,
    )[..., None] # (NUM_RES, NUM_NEIGHBORS, 1)
    # residue size propertie
    sizes = torch.gather(
        input=protbb.size.repeat(1, n_res),
        dim=1,
        index=indices,
    )[..., None] # (NUM_RES, NUM_NEIGHBORS, 1)
    # residue RSA(relative solvent accessibility)
    rsas = torch.gather(
        input=protbb.rsa.repeat(1, n_res),
        dim=1,
        index=indices,
    )[..., None] # (NUM_RES, NUM_NEIGHBORS, 1)
    # residue secondary structure
    sss = torch.gather(
        input=protbb.ss.repeat(1, n_res),
        dim=1,
        index=indices,
    )[..., None] # (NUM_RES, NUM_NEIGHBORS, 1)
    
    ### edge features ###

    bb_coords = torch.cat(
        tensors=[protbb.ca, protbb.cb, protbb.c, protbb.n, protbb.o],
        dim = -2
    )  # (NUM_RES, 5, 3)
    
    if noise_level > 0:
        # noise disturbance
        bb_coords += torch.rand_like(bb_coords) * noise_level  
    
    dist_x = bb_coords[None, :, None, ...] - bb_coords[:, None, :, None, ...]  # (NUM_RES, NUM_RES, 5, 5, 3)
    dist = torch.sqrt(torch.sum(dist_x ** 2, dim=-1)).reshape(n_res, n_res, 25)  # (NUM_RES, NUM_RES, 5, 5) -> (NUM_RES, NUM_RES, 25)
    dist = torch.gather(dist, dim=1, index=indices[..., None].repeat(1, 1, 25))  # (NUM_RES, NUM_NEIGHBORS, 25)
    dist = torch.clamp(dist, min=0, max=15)
    
    rel_pos = torch.clamp(
        input=(protbb.resseq.T - protbb.resseq), 
        min=-32, 
        max=32
    )  # (NUM_RES, NUM_RES)
    rel_chains = (protbb.chain_id.T - protbb.chain_id).bool()
    rel_chains = (~rel_chains).int() # (NUM_RES, NUM_RES)
    # 在蛋白序列上相对于中心残基的偏移量
    pos = torch.gather(
        input=rel_pos, 
        dim=1, 
        index=indices
    )  # (NUM_RES, NUM_NEIGHBORS)
    # chain identity
    chain_id = torch.gather(
        input=rel_chains, 
        dim=1, 
        index=indices
    )  # (NUM_RES, NUM_NEIGHBORS)
    
    ### cat features ###

    nodes = torch.cat(
        tensors=[
            F.one_hot(seq_tokens.long(), num_classes=22), # amino acid type
            dihe_ang, # dihedral angle
            charges, aromas, hydros, polars, sizes, rsas, sss, # properties
            ], 
        dim = -1
    ).transpose(1, 0)  # (NUM_RES, NUM_NEIGHBORS, 22 + 8 + 7)
    edges = torch.cat(
        tensors=[dist, pos[..., None], chain_id[..., None],],
        dim = -1
    ).transpose(1, 0)  # (NUM_RES, NUM_NEIGHBORS, 25 + 1 + 1)
    targets = protbb.seq.squeeze(-1).long()

    return nodes, edges, targets


def parallel_converter(pdb: str) -> Proteinbb:
    return calculator(pdb_file=pdb)


def Process_stru_dir(all_pdbs: List[str]) -> None:
    """Process structure directories iteratively

    Args:
        all_pdbs (List[str]): list of protein structural pathways

    Returns:
        all_protbb (List[Proteinbb]): list of protein backbone dataset
    """
    all_protbb = []
    
    for pdb in tqdm(all_pdbs):
        all_protbb.append(calculator(pdb))
    
    return all_protbb


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


class ProteinDataset(Dataset):
    def __init__(
        self,
        protbbs: List[Proteinbb],
        noise = 0.0,
        n_neighbors = 32,
    ):
        super().__init__()
        
        self.protbbs = protbbs
        self.noise = noise
        self.n_neighbors = n_neighbors
        
    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, torch.LongTensor]:
        protbb = self.protbbs[idx]
        
        node, edge, target = get_neighbors(
            protbb,
            n_neighbors = self.n_neighbors,
            noise_level = self.noise,
        )
        
        return node, edge, target
    
    def __len__(self):
        return len(self.protbbs)

        
class ProteinDataModule(pl.LightningDataModule):
    """protein data precess modele

    Args:
        pl : _description_
    """
    def __init__(
        self,
        args,
        processed_data: str,
        noise = 0.0,
        n_neighbors = 32,
    ):
        super().__init__()
        self.processed_data = processed_data
        self.args = args
        self.n_neighbors = n_neighbors
        self.noise = noise
    
    def setup(self, stage: Optional[str] = None) -> None:
        self.data = torch.load(self.processed_data, map_location='cpu')
        
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
    
    stru_dir = os.path.join(os.getcwd(), 'data', 's669_AF_PDBs')
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir',
        type=str,
        help='The path to the training data directory.', 
        default=stru_dir
    )
    parser.add_argument(
        '--output', 
        type=str, 
        help='The file path where the processed data is stored',
        default=f'{stru_dir}.pt'
    )
    
    args = parser.parse_args()
    
    pdb_filenames = os.listdir(args.input_dir)
    pdb_filenames = [os.path.join(args.input_dir, f) for f in pdb_filenames]
    all_protbb = Process_stru_dir(pdb_filenames)
    torch.save(all_protbb, args.output)
    
    # data = []
    # all_protbb = torch.load('s669_AF_PDBs.pt')
    
    # for protbb in tqdm(all_protbb):
    #     node, edge, target = get_neighbors(protbb, train=True)
    #     data.append([node, edge, target])
    
    # torch.save(data, 'processed_data.pt')

    