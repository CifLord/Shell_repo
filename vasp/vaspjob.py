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
        vasp_params['ldauj'] = [0]*len(atoms)
        vasp_params['ldauu'] = [hubbard_u_dict[site.symbol] if site.symbol \
                                in hubbard_u_dict.keys() else 0 for site in atoms]
        vasp_params['ldauprint'] = 0 
        vasp_params['ldautype'] = 2 
        vasp_params['ldaul'] = [2 if site.symbol \
                                in hubbard_u_dict.keys() else 0 for site in atoms]
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
    d = pymatgen.io.vasp.__file__
    mags = yaml.safe_load(Path(d.replace(d.split('/')[-1], 'VASPIncarBase.yaml')).read_text())
    vasp_params['magmom'] = [mags['INCAR']['MAGMOM'][site.symbol] if site.symbol in \
                             mags['INCAR']['MAGMOM'].keys() else 0.6 for site in atoms]
    
    # Run VASP
    calc = Vasp(**vasp_params)
    if args.inputs_only:
        calc.write_inputs()
    else:
    atoms.set_calculator(calc)
    e = atoms.get_potential_energy()