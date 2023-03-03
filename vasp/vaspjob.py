#!/usr/bin/env python
#SBATCH -p batch
#SBATCH -o myMPI.o%j
#SBATCH -N 1 -n 48
#SBATCH -t 00:30:00

from ase.calculators.vasp import Vasp
from ase.constraints import FixAtoms
from ase.io import read, write
from ase import *

from pymatgen.core.composition import Composition

import os, argparse, pymatgen, yaml
import numpy as np
from pathlib import Path


def read_options():

    parser = argparse.ArgumentParser()
    parser.add_argument("-a","--is_adslab", dest="is_adslab", default=False,
                        type=bool, help="If this is an adslab, set dipole corrections")
    parser.add_argument("-o","--inputs_only", dest="inputs_only", default=False,
                        type=bool, help="Writes VASP inputs only, do not run VASP if True")


    args = parser.parse_args()

    return args

if __name__ == "__main__":    
    """
    Vasp job managed by ASE to specifically run OC22-like slab calculations
    """
    
    args = read_options()
    os.environ['VASP_EXEC']='vasp_std'

    atoms = read('POSCAR')
    
    # default settings for oxide
    vasp_params = dict(xc='PBE', gga='PE', lreal=False, encut=500, ediff=1e-4, 
                       ediffg=-0.05, ispin=2, symprec=1e-10, isif=0, nsw=300,
                       lwave=False, ismear=0, sigma=0.2, isym=0, lcharg=False, 
                       lvtot=False, ibrion=2, potim=0.5, nelm=150, ialgo=48, npar=4)
    
    # add Hubbard U corrections
    hubbard_u_dict = {'Co': 3.32, 'Cr': 3.7, 'Fe': 5.3, 'Mn': 3.9, 
                      'Mo': 4.38, 'Ni': 6.2, 'V': 3.25, 'W': 6.2}
    if any([site.symbol in hubbard_u_dict.keys() for site in atoms]):
        vasp_params['ldau'] = True
        ldau_luj = {site.symbol: {} for site in atoms}
        for el in ldauj_luj.keys():
            ldau_luj[el]['U'] = hubbard_u_dict[site.symbol] if site.symbol \
            in hubbard_u_dict.keys() else 0
            ldau_luj[el]['J'] = 0
            ldau_luj[el]['L'] = 2 if site.symbol in hubbard_u_dict.keys() else 0
            
            
        vasp_params['ldau_luj'] = ldau_luj
        vasp_params['ldauprint'] = 0 
        vasp_params['ldautype'] = 2 
        
        # contains f-electrons
        if any(z > 56 for z in atoms.get_atomic_numbers()):
            vasp_params["lmaxmix"] = 6
        # contains d-electrons
        elif any(z > 20 for z in atoms.get_atomic_numbers()):
            vasp_params["lmaxmix"] = 4
    
    # Add dipole correction if adslab
    if args.is_adslab:
        vasp_params['ldipol'] = True
        vasp_params['idipol'] = 3
        weights = [Composition(site.symbol).weight for site in atoms]
        # center of mass for the slab
        vasp_params['dipol'] = np.average(atoms.get_scaled_positions(),
                                          weights=weights, axis=0)
            
    # Calculate appropriate kpoints
    vasp_params['kpts'] = (np.ceil(30/atoms.cell.lengths()[0]), 
                           np.ceil(30/atoms.cell.lengths()[1]), 1)
    
    # set magmoms
    mags = {'Ce': 5, 'Ce3+': 1, 'Co': 0.6, 'Co3+': 0.6, 'Co4+': 1, 'Cr': 5, 
            'Dy3+': 5, 'Er3+': 3, 'Eu': 10, 'Eu2+': 7, 'Eu3+': 6, 'Fe': 5, 
            'Gd3+': 7, 'Ho3+': 4, 'La3+': 0.6, 'Lu3+': 0.6, 'Mn': 5, 
            'Mn3+': 4, 'Mn4+': 3, 'Mo': 5, 'Nd3+': 3, 'Ni': 5, 'Pm3+': 4, 
            'Pr3+': 2, 'Sm3+': 5, 'Tb3+': 6, 'Tm3+': 2, 'V': 5, 'W': 5, 'Yb3+': 1}   
    
    vasp_params['magmom'] = {site.symbol: mags[site.symbol] if site.symbol in \
                             mags.keys() else 0.6 for site in atoms}
    
    # Run VASP
    calc = Vasp(**vasp_params)
    if args.inputs_only:
        calc.write_input(read('POSCAR'))
    else:
        atoms.set_calculator(calc)
        e = atoms.get_potential_energy()