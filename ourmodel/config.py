### Define amino acid properties


# residue charge. 0 mean neutral, 1 mean negative, 2 mean positive
# Positive: Arg, Lys, His
# Negative: Asp, Glu
residue_to_charge = {
    'ALA': 0,
    'ARG': 2,
    'ASN': 0,
    'ASP': 1,
    'CYS': 0,
    'GLN': 0,
    'GLU': 1,
    'GLY': 0,
    'HIS': 2,
    'ILE': 0,
    'LEU': 0,
    'LYS': 2,
    'MET': 0,
    'PHE': 0,
    'PRO': 0,
    'SER': 0,
    'THR': 0,
    'TRP': 0,
    'TYR': 0,
    'VAL': 0,
}

# residue aromatic. 1 mean aromatic, 0 mean non-aromatic
# His, Phe, Trp, Tyr
residue_to_aromatic = {
    'ALA': 0,
    'ARG': 0,
    'ASN': 0,
    'ASP': 0,
    'CYS': 0,
    'GLN': 0,
    'GLU': 0,
    'GLY': 0,
    'HIS': 1,
    'ILE': 0,
    'LEU': 0,
    'LYS': 0,
    'MET': 0,
    'PHE': 1,
    'PRO': 0,
    'SER': 0,
    'THR': 0,
    'TRP': 1,
    'TYR': 1,
    'VAL': 0,
}

# resicue hydrophobicity
# 3 mean vary hydrophobic amino acid
# 2 mean less hydrophobic amino acid
# 1 mean part hydrophobic amino acid
residue_to_hydrophobic = {
    'ALA': 2,
    'ARG': 1,
    'ASN': 0,
    'ASP': 0,
    'CYS': 2,
    'GLN': 0,
    'GLU': 0,
    'GLY': 2,
    'HIS': 2,
    'ILE': 3,
    'LEU': 3,
    'LYS': 1,
    'MET': 3,
    'PHE': 3,
    'PRO': 2,
    'SER': 2,
    'THR': 2,
    'TRP': 3,
    'TYR': 2,
    'VAL': 3,
}


# residue polarity
# 2 mean clearly polar amino acid
# 1 mean less polar amino acid
residue_to_polarity = {
    'ALA': 1,
    'ARG': 2,
    'ASN': 2,
    'ASP': 2,
    'CYS': 0,
    'GLN': 2,
    'GLU': 2,
    'GLY': 1,
    'HIS': 1,
    'ILE': 0,
    'LEU': 0,
    'LYS': 2,
    'MET': 0,
    'PHE': 0,
    'PRO': 1,
    'SER': 1,
    'THR': 1,
    'TRP': 0,
    'TYR': 1,
    'VAL': 0,
}

# size of residue
# 0 mean tiny amino acid: Gly, Ala, Ser
# 1 mean amino acid: Cys, Asp, Pro, Asn, Thr
# 2: Glu, Val, Gln
# 3: His, Met, Ile, Leu, Lys, Arg
# 4: Phe, Trp, Tyr
residue_to_size = {
    'ALA': 0,
    'ARG': 3,
    'ASN': 1,
    'ASP': 1,
    'CYS': 1,
    'GLN': 2,
    'GLU': 2,
    'GLY': 0,
    'HIS': 3,
    'ILE': 3,
    'LEU': 3,
    'LYS': 3,
    'MET': 3,
    'PHE': 4,
    'PRO': 1,
    'SER': 0,
    'THR': 1,
    'TRP': 4,
    'TYR': 4,
    'VAL': 2,
}


# secondary structure to index 
ss_to_index = {
    'H': 0, # Helix
    'E': 1, # sheet
    'C': 2, # others
}


def rsa_to_three_state(rsa):
    """translate RSA to three-state
    Args:
        rsa (float): relative solvent accessibility
    Returns:
        state (int): three-state RSA [0, 1, 2]. 
        0 mean Buried(B); 1 mean Intermediate(I); 2 mean Exposed(E)
    """
    state = 0
    if rsa < 0.09:
        state = 0
    elif 0.09 <= rsa < 0.36:
        state = 1
    else:
        state = 2
    return state
