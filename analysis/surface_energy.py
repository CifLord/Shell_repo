import random, os, json, itertools
import numpy as np
from sympy import Symbol, solve
from matplotlib import pylab as plt

from analysis.surface_analysis import SurfaceEnergyPlotter
from pymatgen.core.surface import Slab, Lattice
from pymatgen.core.composition import Composition
from mp_api.client import MPRester

from analysis.surface_analysis import SlabEntry
from pymatgen.analysis.wulff import hkl_tuple_to_str
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.core.periodic_table import Element
from ase import Atoms

from database import queryengine as q
f = q.__file__.replace(q.__file__.split('/')[-1], '')
pd_oc22_dict = json.load(open(os.path.join(f, 'pd_oc22_dict.json'), 'r'))


def get_ref_entries(bulk_entry, ref_element='O', MAPIKEY=None):
    """
    For multicomponent systems greater than 2 elements, there is a maximum 
        of n-1 chemical potential terms whereby. Typically when these 
        components contain more than one metal, the energy per atom of a 
        bulk metal in its ground state is used as a reference. e.g. 
        \mu_Ta = \Delta mu_Ta + E_bulk_Ta
        This is wrong as a multicomponent system does not always decompose 
        into its elemental components. Insted, it is more likely for a 
        compound to decompose into other compounds of n-1 elements. As such, 
        the chemical potential that should be used to account for 
        multicomponent oxides of n metals is the chemical potential of an 
        oxide with n-1 metals. e.g. For a slab of CeTa3O9 with formula 
        Ce6Ta21O66, the formation energy will look like this:
        E_slab(Ce6Ta21O66) - (7E_bulk(CeTa3O9) + 3\mu_O - \mu_Ce)
        where \mu_Ce can be substituted for:
        \mu_Ce = E_bulk(CeO2) - 2\mu_O
        This keeps the formation energy as a function of \mu_O, making it 
        much easier to analyze and allowing us to keep the thermodynamic 
        relationship of \mu_O(T, P). This helper function will take in a 
        metal and then create an entry for \mu_M where the energy will be:
        E(M) = (E_bulk(MxOy) - y\mu_O)/x
        This can be substituted back into our surface energy solver, however 
        keep in mind that the chempot term of M left over in the equation 
        must be set to 0.
        
    Args:
        bulk_entry (pmg ComputedEntry): Object representing the bulk of the 
            material whose chempot components we are interested in.
        ref_element (str): Element for which all chempots reference to. e.g. 
            for bulk CeTa3O9, if the ref_element is O, all chempots must be 
            a function of the oxygen chempot.
        MAPIKEY (str): Materials Project API key for querying MP database.
    """
    
    elements_todo = sorted([el for el in bulk_entry.composition.as_dict().keys()], 
                           key=lambda el: Element(el).X)
    del elements_todo[0]
    
    chemsys = bulk_entry.composition.chemical_system
    if chemsys in pd_oc22_dict.keys():
        pd = PhaseDiagram.from_dict(json.loads(pd_oc22_dict[chemsys]))
    else:
        mprester = MPRester(MAPIKEY) if MAPIKEY else MPRester()

        # I'll start by getting all components in my chemical 
        # system of "A-B-O" from the Materials Project. 
        entries = mprester.get_entries_in_chemsys(bulk_entry.composition.chemical_system)

        # choose what elements to use as chemical potential reference, remember the 
        # number of chempots is equal to n-1 where n is the number of elements in 
        # the bulk. We will exclude the element with the largest electronegativity. 
        # Therefore elements like O, N, C etc are always included in the chempot    

        # I'll filter out any components with the same number of elements as 
        # the bulk, I only want decomposition to other compounds, no polymorphs
        pdentries = [PDEntry(entry.composition, entry.energy) for entry in entries \
                     if len(bulk_entry.composition.as_dict().keys()) \
                     != len(entry.composition.as_dict().keys())]

        # With that, I can construct the PhaseDiagram 
        # object and determine the decomposition
        pd = PhaseDiagram(pdentries)
    
    ref_entries = []
    # get the ref entry from the phase diagram
    for entry in pd.qhull_entries:
        if list(entry.composition.as_dict().keys()) == [ref_element]:
            break
    ref_entry = ComputedEntry(list(entry.composition.as_dict().keys())[0], 
                              Symbol('delu_%s' %(ref_element))+entry.energy_per_atom)
    ref_entries.append(ref_entry)
    
    # now get all the other entries for the chemical potential
    for el in elements_todo:
        if el == ref_element:
            continue
        for entry in pd.qhull_entries:
            # for now, I'll need to stick to a binary solution, e.g. use
            # only binary oxides as the reference
            if el in entry.composition.as_dict().keys() and \
            ref_element in entry.composition.as_dict().keys()\
            and len(entry.composition.as_dict().keys()) == 2:
                # the energy of this element is the 
                # formation energy of the oxide
                ebulk = entry.energy / entry.composition.get_integer_formula_and_factor()[1]
                nref = entry.composition.reduced_composition.as_dict()[ref_element]
                energy = ebulk - nref*(Symbol('delu_%s' %(ref_element))+ref_entry.energy_per_atom)
                entry = ComputedEntry(el, energy=energy)
                ref_entries.append(entry)
                break
    
    return ref_entries


def get_dmu_PT(T, P, a=-7.86007886e-02, b=1.14111159e+00, c=-8.34636289e-04):
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

def get_slab_object(dat, relaxed=False):
    
    try:
        slab = Slab(Lattice(dat.cell), dat.atomic_numbers, dat.pmg_init_slab_fcoords,
                    dat.miller_index, None, 0, None, site_properties=dat.site_properties)
    except AttributeError:
        if hasattr(dat, 'ads_pos_relaxed'):
            coords = dat.ads_pos_relaxed if relaxed else slab.cart_coords 
        else:
            coords = dat.pos_relaxed if relaxed else slab.cart_coords 
        return Slab(Lattice(dat.cell), dat.atomic_numbers, coords, dat.miller_index, None, 
                    0, None, coords_are_cartesian=True)
    
    if hasattr(dat, 'ads_pos_relaxed'):
        coords = dat.ads_pos_relaxed if relaxed else slab.cart_coords 
    else:
        coords = dat.pos_relaxed if relaxed else slab.cart_coords 
        
    site_properties = slab.site_properties
    # site_properties['selective_dynamics'] = [[True]*3 if t != 'subsurface' else [False]*3 
    #                                          for t in slab.site_properties['surface_properties']]
        
    return Slab(Lattice(dat.cell), dat.atomic_numbers, coords, slab.miller_index, 
                slab.oriented_unit_cell, slab.shift, slab.scale_factor, 
                coords_are_cartesian=True, site_properties=site_properties)

def get_slab_entry(dat, color=None, relaxed=False, clean_slab_entry=None, ads_entries=None, data={}):
    
    slab = get_slab_object(dat, relaxed=relaxed)
    e = dat.y if 'adslab-' not in dat.rid else dat.y
    return SlabEntry(slab, e, tuple(slab.miller_index), 
                     label=tuple(slab.miller_index), color=color, 
                     entry_id=dat.rid, clean_entry=clean_slab_entry, 
                     adsorbates=ads_entries, data=data)

def get_surface_energy(dat, ref_entries=None, MAPIKEY=None):
    bulk_entry = ComputedEntry(dat.bulk_formula, dat.bulk_energy)
    ref_entries = get_ref_entries(bulk_entry, MAPIKEY=MAPIKEY) if not ref_entries else ref_entries
    slabentry = get_slab_entry(dat)
    surface_energy = slabentry.surface_energy(bulk_entry, ref_entries=ref_entries, referenced=False)
    
    # surface energy is currently a function of mu_X, 
    # rewrite to make it a function of delta mu_O only
    ref_entries_dict = {Symbol('u_%s' %(list(entry.composition.as_dict().keys())[0])): \
                        entry.energy for entry in ref_entries}

    if type(surface_energy).__name__ != 'float':
        return surface_energy.subs(ref_entries_dict)
    else:
        return surface_energy


def plot_surface_energies(list_of_dat, dmu=0, hkil=False, stable_only=False, ref_entries=None, MAPIKEY=None):
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
        se = get_surface_energy(dat, ref_entries=ref_entries, MAPIKEY=MAPIKEY)
        if type(se).__name__ == 'float':
            c = 'r'
        else:
            c = 'g' if se.coeff('delu_O') < 0 else 'b'
            se = se.subs('delu_O', dmu)
        
        hkl = tuple(dat.miller_index)
        if hkl not in hkl_to_se_dict.keys():
            hkl_to_se_dict[hkl] = []
        hkl_to_se_dict[hkl].append([c, se])
        
    # sort x-axis based on normalized hkl
    hkl_norms = [np.linalg.norm(hkl) for hkl in hkl_to_se_dict.keys()]
    hkl_to_se_dict.keys()
    hkl_norms, hkl_list = zip(*sorted(zip(hkl_norms, hkl_to_se_dict.keys())))

    # plot hkl vs se
    for i, hkl in enumerate(hkl_list):
        if stable_only:
            color, se = sorted(hkl_to_se_dict[hkl], key=lambda s:s[1])[0]
            plt.scatter(i, se, c=color, edgecolor='k', s=50)
        else:
            for color, se in hkl_to_se_dict[hkl]:
                plt.scatter(i, se, c=color, edgecolor='k', s=50)

    # Make it pretty
    plt.ylabel(r'Surface energy (eV/$Å^{-2}$)', fontsize=12.5)
    
    if hkil:
        hkl_list = [(hkl[0], hkl[1], -1*(hkl[0]+hkl[1]), hkl[2]) for hkl in hkl_list]
    hkl_strings = [hkl_tuple_to_str(hkl) for hkl in hkl_list]
        
    plt.xticks(ticks=[i for i, hkl in enumerate(hkl_strings)], 
               labels=hkl_strings, rotation=90)
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    plt.scatter(-100, -100, c='r', edgecolor='k', s=50, label='Stoich.')
    plt.scatter(-100, -100, c='g', edgecolor='k', s=50, label='O-excess')
    plt.scatter(-100, -100, c='b', edgecolor='k', s=50, label='O-defficient')
    plt.legend()
    
    return plt

def preset_slabentry_se(slabentry, bulk_entry, ref_entries=None, MAPIKEY=None):
    ref_entries = get_ref_entries(bulk_entry, MAPIKEY=MAPIKEY) if not ref_entries else ref_entries
    ref_entries_dict = {Symbol('u_%s' %(list(entry.composition.as_dict().keys())[0])): \
                        entry.energy for entry in ref_entries}
    se = slabentry.surface_energy(bulk_entry, ref_entries=ref_entries, referenced=False)
    if type(se).__name__ != 'float':
        se = se.subs(ref_entries_dict)
    slabentry.preset_surface_energy = se
    

def make_surface_energy_plotter(list_of_dat, bulk_structure=None, MAPIKEY=None, relaxed=False):
    dat = list_of_dat[0]
    if bulk_structure:
        bulk_entry = ComputedStructureEntry(bulk_structure, dat.bulk_energy)
    else:
        bulk_entry = ComputedEntry(dat.bulk_formula, dat.bulk_energy)

    # color code Miller indices
    hkl_color_dict = {}
    for dat in list_of_dat:
        hkl = tuple(dat.miller_index)
        hkl_color_dict[hkl] = random_color_generator()
            
    # get the slab entries and preset their surface energies as functions of delta mu_O only
    hkl = tuple(dat.miller_index)
    slab_entries = [get_slab_entry(dat, color=hkl_color_dict[hkl],
                                   data={'mpid': dat.entry_id}, 
                                   relaxed=relaxed) for dat in list_of_dat]
    ref_entries = get_ref_entries(bulk_entry, MAPIKEY=MAPIKEY)
    for slabentry in slab_entries:
        preset_slabentry_se(slabentry, bulk_entry, ref_entries=ref_entries)

    # Get the SurfaceEnergyPlotter object for doing surface energy analysis
    for gas_entry in ref_entries:
        if gas_entry.composition.reduced_formula == 'O2':
            break
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


def get_surface_pbx_facets(queryengine, criteria, T=298.15):

    entries = list(queryengine.get_slab_entries(criteria).values())
    ref_entries = {}
    for entry in entries:
        bulk_entry = ComputedStructureEntry.from_dict(bulk_oxides_dict[entry.data['mpid']])
        if entry.data['mpid'] not in ref_entries:
            ref_entries[entry.data['mpid']] = get_ref_entries(bulk_entry, 
                                                               MAPIKEY=queryengine.MAPIKEY)
        preset_slabentry_se(entry, bulk_entry, MAPIKEY=queryengine.MAPIKEY,
                            ref_entries=ref_entries[entry.data['mpid']])
        
    slabentry_dict, pbx_line_dict = {}, {}
    for slabentry in entries:
        mpid = slabentry.data['mpid']
        if mpid not in slabentry_dict.keys():
            slabentry_dict[mpid] = {}
            pbx_line_dict[mpid] = {}
        hkl = slabentry.miller_index
        if hkl not in slabentry_dict[mpid].keys():
            slabentry_dict[mpid][hkl] = []
            pbx_line_dict[mpid][hkl] = {}
        slabentry_dict[mpid][hkl].append(slabentry)
    
    """
    The surface energy of nonstoichometric oxide surface as a fucntion of pH and O 
    (e.g. surface grand potential) is used to determine the Pourbaix diagram. It is 
    given as γ = (E^* - Σnμ_i)/2A. Since we're building a Pourbaix diagram on a per 
    facet basis, we can just ignore 2A for now since all facets we are considering 
    have the same surface area. μ_i can be rewritten as the chemical potential of 
    oxygen only, e.g. μ_O. Here, we assume the following chemical reaction takes 
    place under an aqueous condition: μ_H2O(l) --> 2μ_H + μ_O. As such, 
    μ_O = μ_H2O(l) - 2μ_H. We assume μ_H2O(l) = E_H2O (Gibbs free energy of H2O in 
    a box). We then assume charge transfer when a H atom is in the reaction, e.g.:
    μ_H = μ_H^+ + μ_e- = 1/2G_H2 - eU + kBTlna^+ via the Nernst Equation. Ergo
    μ_O = G_H2O(l) - G_H2 + 2eU - 2kBTlna^+. Since we don't have G, we'll just add
    a correction value for converting between E and G: 
    μ_O = E_H2O(l) - E_H2 + 2eU - 2kBTlna^+ + ΔG_corr
    Lastly, we can also reference μ_O to the energy of O2 (keep it simple and 
    reference to 1/2E_O2 for now) which gives us Δμ_O = μ_O-1/2EO_2 or:
    Δμ_O = E_H2O(l) - E_H2 + 2eU - 2kBTlna^+ + ΔG_corr - 1/2E_O2
    As such, the surface grand potential can be written as:
    2Aγ(U, pH) = E^* - n(E_H2O(l) - E_H2 + 2eU - 2kBTlna^+ + ΔG_corr - 1/2E_O2) 
    where n determines the number of excess/defficient species relative to oxygen.
    """
    muH2O = -14.231 # DFT energy per formula of H2O from MP 
    muH2 = -6.7714828 # DFT energy per formula of H2 from MP 
    EO = -4.946243415 # DFT energy per atom of O2 from MP
    muO = muH2O - muH2 - 2*queryengine.get_e_transfer_corr(T, U=Symbol('U'), pH=Symbol('pH'))
    
    for mpid in slabentry_dict.keys():
        bulk_formula = queryengine.surface_properties.distinct('bulk_reduced_formula',
                                                               {'entry_id': mpid})[0]

        for hkl in slabentry_dict[mpid].keys():
            print(hkl)
            stable_facet = []
            
            # get most stable amongst unique stoich.
            stable_stoich_dict = {}
            for slabentry in slabentry_dict[mpid][hkl]:
                if slabentry.composition not in stable_stoich_dict.keys():
                    stable_stoich_dict[slabentry.composition] = []
                stable_stoich_dict[slabentry.composition].append(slabentry)
            for comp in stable_stoich_dict.keys():
                stable_stoich_dict[comp] = sorted(stable_stoich_dict[comp], key=lambda entry: entry.energy)
            stable_slabentries = [stable_stoich_dict[c][0] for c in stable_stoich_dict.keys()]

            # Solve system of two eqns for each pair of surfaces as U(pH)
            for c in itertools.combinations(stable_slabentries, 2):
                eqn1 = c[0].preset_surface_energy.subs({'delu_O': muO-EO + queryengine.Gcorr['O']})
                eqn2 = c[1].preset_surface_energy.subs({'delu_O': muO-EO + queryengine.Gcorr['O']})
                pbx_line_dict[mpid][hkl][c] = solve(eqn1 - eqn2, Symbol('U'))
                print(solve(eqn1 - eqn2, Symbol('U')))
                
    return pbx_line_dict


bulk_oxides_dict = json.load(open(os.path.join(f, 'bulk_oxides_20220621.json'), 'r'))
bulk_oxides_dict = {entry['entry_id']: entry for entry in bulk_oxides_dict}


def get_surface_pbx_diagram(queryengine, criteria, increment=500, Ulim=[-1, 3], plot=True, bare_only=True):

    entries = list(queryengine.get_slab_entries(criteria).values())
    if bare_only:
        entries = [entry for entry in entries if 'adslab-' not in entry.entry_id]
    ref_entries = {}
    for entry in entries:
        bulk_entry = ComputedStructureEntry.from_dict(bulk_oxides_dict[entry.data['mpid']])
        if entry.data['mpid'] not in ref_entries:
            ref_entries[entry.data['mpid']] = get_ref_entries(bulk_entry, 
                                                               MAPIKEY=queryengine.MAPIKEY)
        preset_slabentry_se(entry, bulk_entry, MAPIKEY=queryengine.MAPIKEY,
                            ref_entries=ref_entries[entry.data['mpid']])
        
    slabentry_dict, stable_slabentry_dict = {}, {}
    for slabentry in entries:
        mpid = slabentry.data['mpid']
        if mpid not in slabentry_dict.keys():
            slabentry_dict[mpid] = {}
            stable_slabentry_dict[mpid] = {}
        hkl = slabentry.miller_index
        if hkl not in slabentry_dict[mpid].keys():
            slabentry_dict[mpid][hkl] = []
            stable_slabentry_dict[mpid][hkl] = []
        slabentry_dict[mpid][hkl].append(slabentry)
    
    muH2O = -14.231 # DFT energy per formula of H2O from MP 
    muH2 = -6.7714828 # DFT energy per formula of H2 from MP 
    EO = -4.946243415 # DFT energy per atom of O2 from MP
    for mpid in slabentry_dict.keys():
        
        bulk_formula = queryengine.surface_properties.distinct('bulk_reduced_formula',
                                                               {'entry_id': mpid})[0]

        for hkl in slabentry_dict[mpid].keys():

            pH_range = np.linspace(0, 14, increment)
            U_range = np.linspace(Ulim[0], Ulim[1], increment)
            # print(mpid, hkl, len(slabentry_dict[mpid][hkl]))
            
            stable_facet = []
            for i, pH in enumerate(pH_range):
                new_row = []
                for ii, U in enumerate(U_range):

                    """
                    The surface energy of nonstoichometric oxide surface as a fucntion of pH and O 
                    (e.g. surface grand potential) is used to determine the Pourbaix diagram. It is 
                    given as γ = (E^* - Σnμ_i)/2A. Since we're building a Pourbaix diagram on a per 
                    facet basis, we can just ignore 2A for now since all facets we are considering 
                    have the same surface area. μ_i can be rewritten as the chemical potential of 
                    oxygen only, e.g. μ_O. Here, we assume the following chemical reaction takes 
                    place under an aqueous condition: μ_H2O(l) --> 2μ_H + μ_O. As such, 
                    μ_O = μ_H2O(l) - 2μ_H. We assume μ_H2O(l) = E_H2O (Gibbs free energy of H2O in 
                    a box). We then assume charge transfer when a H atom is in the reaction, e.g.:
                    μ_H = μ_H^+ + μ_e- = 1/2G_H2 - eU + kBTlna^+ via the Nernst Equation. Ergo
                    μ_O = G_H2O(l) - G_H2 + 2eU - 2kBTlna^+. Since we don't have G, we'll just add
                    a correction value for converting between E and G: 
                    μ_O = E_H2O(l) - E_H2 + 2eU - 2kBTlna^+ + ΔG_corr
                    Lastly, we can also reference μ_O to the energy of O2 (keep it simple and 
                    reference to 1/2E_O2 for now) which gives us Δμ_O = μ_O-1/2EO_2 or:
                    Δμ_O = E_H2O(l) - E_H2 + 2eU - 2kBTlna^+ + ΔG_corr - 1/2E_O2
                    As such, the surface grand potential can be written as:
                    2Aγ(U, pH) = E^* - n(E_H2O(l) - E_H2 + 2eU - 2kBTlna^+ + ΔG_corr - 1/2E_O2) 
                    where n determines the number of excess/defficient species relative to oxygen.
                    """
                    muO = muH2O - muH2 - 2*queryengine.get_e_transfer_corr(298.15, U=U, pH=pH)
                    DG = [slabentry.preset_surface_energy.subs({'delu_O': muO-EO + queryengine.Gcorr['O']}) \
                          if type(slabentry.preset_surface_energy).__name__ != 'float' \
                          else slabentry.preset_surface_energy \
                          for slabentry in slabentry_dict[mpid][hkl]]
                    a = DG.index(min(DG))
                    entry = slabentry_dict[mpid][hkl][a]
                    if entry not in stable_slabentry_dict[mpid][hkl]:
                        stable_slabentry_dict[mpid][hkl].append(entry)
                    new_row.append(a)
                    
                stable_facet.append(new_row)
                
            if plot:
                
                pH_range, U_range = np.meshgrid(pH_range, U_range)
                stable_facet = np.array(stable_facet)
                plt.xlim([0,14])
                plt.ylim(Ulim)
                plt.contourf(pH_range, U_range, stable_facet, cmap='rainbow',
                             vmin=-np.abs(stable_facet).max(),
                             vmax=np.abs(stable_facet).max())

                plt.title('%s %s %s' %(bulk_formula, mpid, tuple(hkl)))
                plt.xlabel(r'pH', fontsize=20)
                plt.ylabel(r'U (V)', fontsize=20)

                plt.plot([0,14], [1.2,1.2], 'k--')
                plt.plot([0,14], [2,2], 'k--')

                plt.savefig('%s_%s_%s%s%s.pdf' %(bulk_formula, mpid, hkl[0],hkl[1],hkl[2]))
                plt.show()
                plt.close()

    return stable_slabentry_dict