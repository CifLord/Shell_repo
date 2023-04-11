import torch, os, threading
from tqdm import tqdm

from ase.constraints import FixAtoms
from ase import Atoms
from ase.optimize import BFGS

from ocpmodels.common.relaxation.ase_utils import OCPCalculator

from structure_generation.lmdb_generator import generate_lmdb

        
def cal_slab_energy(data, calc, traj_output=False):

    testobj = Atoms(data.atomic_numbers, positions=data.pos, tags=data.tags, 
                    cell=data.cell.squeeze(), pbc=True)
    testobj.calc = calc
    unrelax_slab_energy = testobj.get_potential_energy()
    
    if data.fixed:
        c = FixAtoms(mask=data.fixed)
        testobj.set_constraint(c)

    if traj_output == True:
        os.makedirs("./trajs", exist_ok=True)
        files=len(os.listdir('./trajs/'))
        opt = BFGS(testobj, trajectory=f"./trajs/data.slab_formula+{files}"+".traj")
    
    opt = BFGS(testobj)
    opt.run(fmax=0.05, steps=100)
    relax_slab_energy = testobj.get_potential_energy()
    forces=testobj.get_forces()
    pos_relaxed=testobj.get_positions()
    
    return unrelax_slab_energy, relax_slab_energy, forces, pos_relaxed


def add_info(data, calc):
    
    unrelax_slab_energy, relax_slab_energy,forces, pos_relaxed = \
    cal_slab_energy(data, calc, traj_output=True)
    
    data.y = relax_slab_energy
    data.unrelax_energy = unrelax_slab_energy
    data.pos_relaxed = torch.Tensor(pos_relaxed)
    data.force = torch.Tensor(forces)
        
    return data  
    

class MyTread(threading.Thread):

    def __init__(self, datalist, pathname):
        
        threading.Thread.__init__(self)
        
        self.data_list = datalist
        self.pathname = pathname
    
    def run(self):        
        
        # power outage: T1: 800 T2,750, T3 700
        data_list_E = []
        tot = len(self.data_list)
        
        config_yml='../ocp/configs/oc22/s2ef/gemnet-oc/gemnet_oc_oc20_oc22.yml'    
        checkpoint="../ocp/prediction/gemnet_oc_base_oc20_oc22.pt"     
        calc = OCPCalculator(config_yml, checkpoint, device='0')
                    
        for data in tqdm(self.data_list):            
            # run predictions here
            data = add_info(data, calc)
            data_list_E.append(data)
            
            if len(data_list_E)>=10: 
                generate_lmdb(data_list_E, self.pathname)
                data_list_E = []

        generate_lmdb(data_list_E, self.pathname)
