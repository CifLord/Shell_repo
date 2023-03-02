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


def convert_atoms_data(atoms: ase.Atoms, bulk_formula: str,
                       bulk_energy: float, bulk_idx,
                       r_energy=0, r_force=0, un_energy=0):
    # input
    #    atoms object
    #    r_energy: add relaxed energy if add assign r_energy to a value
    #    r_force: add relaxed force matrix if assign r_force to a value
    #    un_force: add un-relaxed energy if assign un_energy to a value
    # output:
    #    Data object

    atomic_numbers = torch.Tensor(atoms.get_atomic_numbers())
    positions = torch.Tensor(atoms.get_positions())
    cell = torch.Tensor(np.array(atoms.get_cell())).view(1,3,3)
    natoms = positions.shape[0]
    tags = torch.Tensor(atoms.get_tags())
    fixed_idx = torch.zeros(natoms)
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            fixed_idx[constraint.index] = 1
    data = Data(cell=cell, pos=positions, atomic_numbers=atomic_numbers,
                natoms=natoms, tags=tags)
    data.fixed = fixed_idx

    if r_energy != 0:
        data.y=r_energy
    if r_force != 0:
        data.force=r_force
    if un_energy !=0:
        data.unrelax_energy=un_energy

    data.slab_formula = atoms.get_chemical_formula()
    data.miller = atoms.info['miller_index']
    data.bulk_formula = bulk_formula
    data.bulk_energy = bulk_energy
    data.entry_id = bulk_idx

    return data


def get_split_list(df):

    list1 = list(set(df[df['abundance']>2]['mp']))
    cond1 = (df['abundance']>1)
    cond2 = (df['abundance']<=2)
    list2 = list(set(df[cond1 & cond2]['mp']))
    cond1 = (df['abundance']>0)
    cond2 = (df['abundance']<=1)
    list3 = list(set(df[cond1 & cond2]['mp']))
    list4 = list(set(df[df['abundance']<0]['mp']))

    return list1, list2, list3, list4


def sv_slabs(data_list: object, pathname: str):

    pathname = pathname + '.lmdb'
    db = lmdb.open(
        pathname,
        map_size=1099511627 * 2,
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

    
        
    
        
def cal_slab_energy(data,calc):

    testobj =Atoms(data.atomic_numbers,positions=data.pos,tags=data.tags,cell=data.cell.squeeze(),pbc=True)
    testobj.calc = calc
    unrelax_slab_energy = testobj.get_potential_energy()
    opt = BFGS(testobj)
    opt.run(fmax=0.05, steps=100)
    relax_slab_energy = testobj.get_potential_energy()
    forces=testobj.get_forces()
    pos_relaxed=testobj.get_positions()
    
    return unrelax_slab_energy, relax_slab_energy,forces, pos_relaxed
