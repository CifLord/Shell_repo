import torch, os, threading
from tqdm import tqdm
import sys
import vaspjob
f = vaspjob.__file__
repo_dir = f.replace(os.path.join(f.split('/')[-2], f.split('/')[-1]), '')
sys.path.append(repo_dir)
from ocpmodels.datasets import LmdbDataset

from ase.constraints import FixAtoms
from ase import Atoms
from ase.optimize import BFGS
import numpy as np
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from structure_generation.lmdb_generator import generate_lmdb
from ase.constraints import Hookean
from ase.geometry.analysis import Analysis
import random


def add_hookean_constraint(image, des_rt = 2., rec_rt = 1., spring_constant=7.5, tol=0.15):
    """Applies a Hookean restorative force to prevent adsorbate desorption, dissociation and
    surface reconstruction.
    All bonded pairs in the image will be found with ase Analysis class. The threshold length
    below which no restorative force will be applied is set as the bond length times the tolerance.
    If bond length of the atom pair can not be found with _load_bond_length_data function
    from pymatgen, current distance between two atoms, or a default bond length will be used.
    This method requires the atoms object to be tagged. Adapted from the FineTuna repository:
        https://github.com/ulissigroup/finetuna
    cite the following::
        Musielewicz, J., Wang, X., Tian, T., & Ulissi, Z. (2022). FINETUNA: fine-tuning 
            accelerated molecular simulations. Machine Learning: Science and Technology, 
            3(3), 03LT01. https://doi.org/10.1088/2632-2153/ac8fe0
    
    Args:
        image (atoms): tagged ASE atoms object, 0 for bulk, 1 for surface, 2 for adsorbate.
               
        des_rt (float, optional): desorption threshold. Apply a spring to a randomly selected
        adsorbate atom so that the adsorbate doesn't fly away from the surface. Defaults to 2,
        i.e.: if the selected atom move 2A above its current z position, apply the restorative
        force.
        
        rec_rt (float, optional): reconstruction threshold. Apply springs to the surface atoms
        to prevent surface reconstruction. Defaults to 1A, i.e.: if a surface atom move 1A away 
        from its current position, apply the restorative force.
        
        spring_constant (int, optional): Hooke’s law (spring) constant. Defaults to 5.
        tol (float, optional): relative tolerance to the bond length. Defaults to 0.3, i.e.: if
        the bond is 30% over the bond length, apply the restorative force.
    """
    
    ana = Analysis(image)
    cons = image.constraints
    tags = image.get_tags()
    surface_indices = np.where(tags==1)[0]  #[i for i, tag in enumerate(tags) if tag == 1]
    ads_indices = np.where(tags==2)[0]      #[i for i, tag in enumerate(tags) if tag == 2]
    if len(ads_indices)==0:
        pass
    else:
        for i in ads_indices:
            if ana.unique_bonds[0][i]:
                for j in ana.unique_bonds[0][i]:
                    syms = tuple(sorted([image[i].symbol, image[j].symbol]))
                    rt = (1 + tol) * ana.get_bond_value(0, [i, j])
                    cons.append(Hookean(a1=i, a2=int(j), rt=rt, k=spring_constant))
                    print(
                        f"Applied a Hookean spring between atom {image[i].symbol} and", \
                        f"atom {image[j].symbol} with a threshold of {rt:.2f} and", \
                        f"spring constant of {spring_constant}"
                    )
        rand_ads_index = random.choice(ads_indices)
        rand_ads_z = image[rand_ads_index].position[2]
        cons.append(Hookean(a1=rand_ads_index, a2=(0., 0., 1., -(rand_ads_z + des_rt)), k=spring_constant))
    for i in surface_indices:
        cons.append(Hookean(a1=i, a2=image[i].position, rt=rec_rt, k=spring_constant))
    image.set_constraint(cons)

        
def cal_slab_energy(data, calc, traj_output=False, debug=False):

    testobj = Atoms(data.atomic_numbers, positions=data.pos, tags=data.tags, 
                    cell=data.cell.squeeze(), pbc=True)
    testobj.calc = calc
    unrelax_slab_energy = testobj.get_potential_energy()
    
    # add selective dynamics
    if len(data.fixed):#maybe previous method is more stable...
        c = FixAtoms(mask=data.fixed)
        testobj.set_constraint(c)
    
    # added spring constant to prevent massive
    # surface reconstruction and desorption
    add_hookean_constraint(testobj)

    if traj_output == True:
        os.makedirs("./trajs", exist_ok=True)
        files=len(os.listdir('./trajs/'))
        opt = BFGS(testobj, trajectory=f"./trajs/data.slab_formula+{files}"+".traj")
    
    else:
        opt = BFGS(testobj)
    
    if debug:
        relax_slab_energy = 0
        forces = [[0]*3]*len(testobj)
        pos_relaxed = [[0]*3]*len(testobj)
    else:
        opt.run(fmax=0.05, steps=100)
        del testobj.constraints[1:]
        opt.run(fmax=0.05, steps=30)
        relax_slab_energy = testobj.get_potential_energy()
        forces=testobj.get_forces()
        pos_relaxed=testobj.get_positions()
    
    return unrelax_slab_energy, relax_slab_energy, forces, pos_relaxed


def add_info(data, calc, debug=False):
    
    unrelax_slab_energy, relax_slab_energy,forces, pos_relaxed = \
    cal_slab_energy(data, calc, traj_output=True, debug=debug)
    
    data.y = relax_slab_energy
    data.unrelax_energy = unrelax_slab_energy
    data.pos_relaxed = torch.Tensor(pos_relaxed)
    data.force = torch.Tensor(forces)
    if torch.max(data.force)>=0.055:
        data.preds='Failed'
        
    return data  
    

class MyThread(threading.Thread):

    def __init__(self, datalist, pathname, gpus=0, debug=False):
        
        threading.Thread.__init__(self)
        
        self.data_list = datalist
        self.pathname = pathname
        self.gpus=gpus
        self.debug = debug
        self.rids_list = [dat.rid for dat in LmdbDataset({'src': pathname})]
        
    
    def run(self):
    
        data_list_E = []        
        config_yml=os.path.join(repo_dir, 'ocp/configs/oc22/s2ef/gemnet-oc/gemnet_oc_oc20_oc22.yml')
        checkpoint=os.path.join(repo_dir, "ocp/prediction/gemnet_oc_base_oc20_oc22.pt")
        calc = OCPCalculator(config_yml, checkpoint, cpu=False)
                    
        for data in tqdm(self.data_list): 
            if data.rid in self.rids_list:
                continue
            # run predictions here
            data = add_info(data, calc, debug=self.debug)
            data_list_E.append(data)
            
            if len(data_list_E)>=10: 
                generate_lmdb(data_list_E, self.pathname)
                data_list_E = []

        generate_lmdb(data_list_E, self.pathname)
