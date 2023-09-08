import torch
import ase
import numpy as np
from torch_geometric.data import Data
from ase.constraints import FixAtoms
import lmdb
import pickle
from ase import Atoms
import os
from ase.optimize import BFGS


def convert_atoms_data(atoms: ase.Atoms):

    atomic_numbers = torch.Tensor(atoms.get_atomic_numbers())
    positions = torch.Tensor(atoms.get_positions())
    cell = torch.Tensor(np.array(atoms.get_cell())).view(1,3,3)
    natoms = positions.shape[0]
    tags = torch.Tensor(atoms.get_tags())
    fixed_idx = torch.zeros(natoms)
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            fixed_idx[constraint.index] = 1
            
    data_dict = {'cell': cell, 'pos': positions, 'atomic_numbers': atomic_numbers, 
                 'natoms': natoms, 'tags': tags, 'fixed': fixed_idx}

    if 'r_energy' in atoms.info.keys():
        data_dict['y'] = atoms.info['r_energy']
    if 'r_force' in atoms.info.keys():
        data_dict['force'] = atoms.info['r_force']
    if 'un_energy' in atoms.info.keys():
        data_dict['unrelax_energy'] = atoms.info['un_energy']
        
    data_dict['slab_formula'] = atoms.get_chemical_formula()
    data_dict.update(atoms.info)
    
    data = Data(**data_dict)
    data.fixed = fixed_idx

    return data


def generate_lmdb(atoms_list, pathname, pre_data_list=[]):
    """
    atoms_list:: Can be either a list of atoms objects or list of Data objects
    """
   
    if not atoms_list:
        data_list = pre_data_list
    else: 
        data_list = [convert_atoms_data(atoms) for atoms in atoms_list] \
        if type(atoms_list[0]).__name__ == 'Atoms' else atoms_list
    
    pathname = pathname + '.lmdb' if '.lmdb' not in pathname else pathname
    db = lmdb.open(
        pathname,
        map_size=1099511627 * 4,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    
    for i in range(len(data_list)):
        data = data_list[i]
        txn = db.begin(write=True)        
        length=txn.stat()['entries']        
        txn.put(f"{length}".encode('ascii'), pickle.dumps(data, protocol=0))
        txn.commit()
        db.sync()
    db.close()    
    
    
def lmdb_size(pathname):
    
    db = lmdb.open(
    pathname,
    map_size=1099511627 * 3,
    subdir=False,
    meminit=False,
    map_async=True,
    ) 
    txn=db.begin(write=True)
    lmdbsize=txn.stat()['entries']  
    db.close()

    return lmdbsize 
