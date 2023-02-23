import numpy as np
from matplotlib import pylab as plt

from pymatgen.core.composition import Composition
from pymatgen.analysis.surface_analysis import SlabEntry
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms

def get_slab_entry(dat):

    atoms=Atoms(dat.atomic_numbers,
                positions=dat.pos,
                tags=dat.tags,
                cell=dat.cell.squeeze(), pbc=True)
    
    return SlabEntry(AseAtomsAdaptor.get_structure(atoms), dat.y, dat.miller)


def get_surface_energy(dat):
    bulk_entry = ComputedEntry(dat.bulk_formula, dat.bulk_energy)
    gas_entry = ComputedEntry('O2', 2*-7.204) # the ref energy for O in OC20
    slabentry = get_slab_entry(dat)
    
    return slabentry.surface_energy(bulk_entry, [gas_entry])


def plot_surface_energies(list_of_dat, dmu=0):
    """
    Function takes a list of OCP Data objects and plots the surface 
        energy against the Miller Index of the slab. The surface energy 
        is always a constant value at a certain chemical potential of 
        oxygen. Chempot defaults to 0 eV, as such the surface energy is 
        defined as: 
            SE = (Eslab - N*Ebulk - nEO)/2A 
        where n is the number of missing/excess O relative to bulk 
        stoichiometry. O-defficient, O-excess, and stoichiometric slabs 
        labelled as red, green, blue respectively. Miller Index (x-axis)
        organized by its norm.
    
    Parameters:
        - list_of_dat (list): List containing Data objects for which to 
            plot surface energy.
        - dmu (float): Chemical potential (eV) of oxygen at a fixed value. 
    """
    
    # Determine stoichiometry colors and surface energy
    hkl_to_se_dict = {}
    for dat in list_of_dat:
        se = get_surface_energy(dat)
        if type(se).__name__ == 'float':
            c = 'r'
        else:
            c = 'g' if se.coeff('delu_O') < 0 else 'b'
            se = se.subs('delu_O', dmu)
        if dat.miller not in hkl_to_se_dict.keys():
            hkl_to_se_dict[dat.miller] = []
        hkl_to_se_dict[dat.miller].append([c, se])
        
    # sort x-axis based on normalized hkl
    hkl_norms = [np.linalg.norm(hkl) for hkl in hkl_to_se_dict.keys()]
    hkl_to_se_dict.keys()
    hkl_norms, hkl_list = zip(*sorted(zip(hkl_norms, hkl_to_se_dict.keys())))

    # plot hkl vs se
    for i, hkl in enumerate(hkl_list):
        for color, se in hkl_to_se_dict[hkl]:
            plt.scatter(i, se, c=color, edgecolor='k', s=50)
            
    # Make it pretty
    plt.ylabel(r'Surface energy (eV/$Å^{-2}$)', fontsize=12.5)
    plt.xticks(ticks=[i for i, hkl in enumerate(hkl_list)], 
               labels=hkl_list, rotation=90)
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    plt.scatter(-100, -100, c='r', edgecolor='k', s=50, label='Stoich.')
    plt.scatter(-100, -100, c='g', edgecolor='k', s=50, label='O-excess')
    plt.scatter(-100, -100, c='b', edgecolor='k', s=50, label='O-defficient')
    plt.legend()
    
    return plt