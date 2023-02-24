import random 
import numpy as np
from sympy import Symbol
from matplotlib import pylab as plt

from pymatgen.analysis.surface_analysis import SurfaceEnergyPlotter
from pymatgen.core.composition import Composition
from pymatgen.analysis.surface_analysis import SlabEntry
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms


def get_dmu(T, P, a=-7.86007886e-02, b=1.14111159e+00, c=-8.34636289e-04):
    # Get Delta mu_O as a function of T and P
    k = 8.617333262145 * 10**-5 # eV/K
    g = c*T**(b)+a
    g0 = c*0**(b)+a
    # g0 shifts dmu to 0 (ie we want a reference to T=0K)
    return (1/2)*(g-g0 + k*T*np.log(P/0.1)) 


def random_color_generator():
    rgb_indices = [0, 1, 2]
    color = [0, 0, 0, 1]
    random.shuffle(rgb_indices)
    for i, ind in enumerate(rgb_indices):
        if i == 2:
            break
        color[ind] = np.random.uniform(0, 1)
    return color

def get_slab_entry(dat, color=None):

    atoms=Atoms(dat.atomic_numbers,
                positions=dat.pos,
                tags=dat.tags,
                cell=dat.cell.squeeze(), pbc=True)
    
    return SlabEntry(AseAtomsAdaptor.get_structure(atoms), dat.y, dat.miller, 
                     label=dat.miller, color=color)


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


def make_surface_energy_plotter(list_of_dat):
    dat = list_of_dat[0]
    bulk_entry = ComputedEntry(dat.bulk_formula, dat.bulk_energy)
    gas_entry = ComputedEntry('O2', 2*-7.204) # the ref energy for O in OC20
    
    # color code Miller indices
    hkl_color_dict = {}
    for dat in list_of_dat:
        hkl_color_dict[dat.miller] = random_color_generator()
            
    # Get the SurfaceEnergyPlotter object for doing surface energy analysis
    slab_entries = [get_slab_entry(dat, color=hkl_color_dict[dat.miller]) for dat in list_of_dat]
    surfplot = SurfaceEnergyPlotter(slab_entries, bulk_entry, ref_entries=[gas_entry])
    
    return surfplot

def plot_chempot_vs_surface_energy(list_of_dat, chempot_range=[-2,0]):
    
    surfplot = make_surface_energy_plotter(list_of_dat)
    
    return surfplot.chempot_vs_gamma(Symbol('delu_O'), 
                                     chempot_range, show_unstable=False)


def plot_P_vs_T(list_of_dat, T_range, lnP_range, increment=100):
    
    dmus = []
    for T in np.linspace(T_range[0], T_range[1], increment):
        for P in np.linspace(lnP_range[0], lnP_range[1], increment):
            dmus.append(get_dmu(T, 10**(P)))

    surfplt = make_surface_energy_plotter(list_of_dat)
            
    hkl = list(surfplt.all_slab_entries.keys())[0]
    stab_entry_stable_dict = {}
    for dmu in np.linspace(min(dmus), max(dmus), increment):
        entry = surfplt.get_stable_entry_at_u(hkl, delu_dict={Symbol('delu_O'): dmu})[0]
        if entry not in stab_entry_stable_dict.keys():
            stab_entry_stable_dict[entry] = []
        stab_entry_stable_dict[entry].append(dmu)

    stab_comp_dict = {}
    stab_comp_color = {}
    for entry in stab_entry_stable_dict.keys():
        stab_comp_dict[entry.composition.formula.replace(' ', '')] = [min(stab_entry_stable_dict[entry]), 
                                                                      max(stab_entry_stable_dict[entry])]
        stab_comp_color[entry.composition.formula.replace(' ', '')] = random_color_generator()

    all_lines = []
    for T in np.linspace(T_range[0], T_range[1], increment):
        P_length_dict = {}
        for P in np.linspace(lnP_range[0], lnP_range[1], increment):
            dmu = get_dmu(T, 10**(P))
            color = None
            for comp in stab_comp_dict.keys():
                if comp not in P_length_dict.keys():
                    P_length_dict[comp] = []
                if stab_comp_dict[comp][0] < dmu < stab_comp_dict[comp][1]:
                    P_length_dict[comp].append(P)

        for comp in P_length_dict.keys():
            if not P_length_dict[comp]:
                continue
            all_lines.append([T, min(P_length_dict[comp]), 
                              max(P_length_dict[comp]), stab_comp_color[comp], comp])

    labeled = []
    for l in all_lines:
        if l[4] not in labeled:
            plt.plot([l[0], l[0]], [l[1], l[2]], color=l[3], linewidth=3, label=l[4])
            labeled.append(l[4])
        else:
            plt.plot([l[0], l[0]], [l[1], l[2]], color=l[3], linewidth=3)        

    plt.xlabel('Temperature (K)', fontsize=12.5)
    plt.ylabel('Pressure ln(P) (MPa)', fontsize=12.5)
    plt.xlim(T_range)
    plt.ylim(lnP_range)
    plt.plot([373.15, 373.15], plt.ylim(), 'k')
    plt.plot(plt.xlim(), [-1, -1], 'k')
    plt.legend()
    
    return plt