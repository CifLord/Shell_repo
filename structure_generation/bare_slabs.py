from pymatgen.core.surface import SlabGenerator, generate_all_slabs, \
get_symmetrically_distinct_miller_indices, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import VoronoiNN, CrystalNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from collections import defaultdict
from ase.constraints import FixAtoms
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.ext.matproj import MPRester
#from mp_api.client import MPRester
from database.generate_metadata import write_metadata_json
from structure_generation.MXide_adsorption import make_superslab_with_partition, get_repeat_from_min_lw

import numpy as np
import json


"""
Metadata and additional API information for adsorbed slabs:
    entry_id (str): Materials Project ID of the corresponding bulk
    database (str): Materials database where the bulk was obtained from
    bulk_rid (str): 20 digit random ID of the corresponding bulk e.g. 
        bulk-h4qfsgj3432kl113DVpD
    rid (str): 20 digit random ID of the adsorbed slab e.g. 
        slab-t43frgfgy5lFh3f3CVGD
    miller_index (tuple): Miller index of the facet 
    bulk_formula (str): Reduced formula of the corresponding bulk 
    bulk_composition (dict): Composition of the bulk with elements 
        as keys and the number of atoms as values 
    bulk_chemsys (str): Elemental components of the bulk e.g. for 
        LiFe(PO4), the chemsys is 'Li-Fe-P-O'
    pmg_slab (dict): Pymatgen slab object as a dictionary. Contains useful 
        information about miller index, scale factor, bulk wyckoff positions, etc. 
    calc_type (str): What kind of system is this calculation for 
        ('bare_slab', 'adsorbed_slab', 'adsorbate_in_a_box', 'bulk') 
    func (str): The DFT functional used or the method used to obtain the 
        energy/relaxation data (Beef-vdW, PBE, PBEsol, rPBE, GemNet-OC (for ML) etc...)
"""

from database import generate_metadata
f = generate_metadata.__file__
bulk_oxides_20220621 = json.load(open(f.replace(f.split('/')[-1], 'bulk_oxides_20220621.json'), 'rb'))
bulk_oxides_dict = {entry['entry_id']: ComputedStructureEntry.from_dict(entry) \
                    for entry in bulk_oxides_20220621}


def tag_surface_atoms(bulk, slab, height_tol=2, count_undercoordination=False):
    '''
    Sets the tags of an `ase.Atoms` object. Any atom that we consider a "bulk"
    atom will have a tag of 0, and any atom that we consider a "surface" atom
    will have a tag of 1. We use a combination of Voronoi neighbor algorithms
    (adapted from from `pymatgen.core.surface.Slab.get_surface_sites`; see
    https://pymatgen.org/pymatgen.core.surface.html) and a distance cutoff.
    Arg:
        bulk_atoms      `ase.Atoms` format of the respective bulk structure
        surface_atoms   The surface where you are trying to find surface sites in
                        `ase.Atoms` format
        count_undercoordination: Whether or not to consider undercoordination in 
            identifying surface sites. Might make the algorithm take longer
    '''
    
    # oxid_guess = bulk.composition.oxi_state_guesses()
    # oxid_guess = oxid_guess or [{e.symbol: 0 for e in bulk.composition}]
    # bulk.add_oxidation_state_by_element(oxid_guess[0])
    # slab.add_oxidation_state_by_element(oxid_guess[0])
    height_tags = find_surface_atoms_by_height(slab, height_tol=height_tol)
    if count_undercoordination:
        voronoi_tags = find_surface_atoms_with_voronoi(bulk, slab)
    else:
        voronoi_tags = [0]*len(height_tags)
    
    # If either of the methods consider an atom a "surface atom", then tag it as such.
    tags = [max(v_tag, h_tag) for v_tag, h_tag in zip(voronoi_tags, height_tags)]                
    slab.add_site_property('tag', tags)
    
    
def calculate_coordination_of_bulk_atoms(bulk):
    '''
    Finds all unique atoms in a bulk structure and then determines their
    coordination number. Then parses these coordination numbers into a
    dictionary whose keys are the elements of the atoms and whose values are
    their possible coordination numbers.
    For example: `bulk_cns = {'Pt': {3., 12.}, 'Pd': {12.}}`
    Arg:
        bulk_atoms  An `ase.Atoms` object of the bulk structure.
    Returns:
        bulk_cn_dict    A defaultdict whose keys are the elements within
                        `bulk_atoms` and whose values are a set of integers of the
                        coordination numbers of that element.
    '''
    voronoi_nn = VoronoiNN(tol=0.1)  # 0.1 chosen for better detection

    # Object type conversion so we can use Voronoi
    bulk = bulk.get_primitive_structure()
    sga = SpacegroupAnalyzer(bulk)
    sym_struct = sga.get_symmetrized_structure()

    # We'll only loop over the symmetrically distinct sites for speed's sake
    bulk_cn_dict = defaultdict(set)
    cnn = CrystalNN()
    for idx in sym_struct.equivalent_indices:
        site = sym_struct[idx[0]]
        cn = voronoi_nn.get_cn(sym_struct, idx[0], use_weights=True)
        
        cn = round(cn, 5)
        cn = len(cnn.get_nn_info(sym_struct, idx[0]))
        bulk_cn_dict[site.species_string].add(cn)
    return bulk_cn_dict

    
def find_surface_atoms_with_voronoi(bulk, slab):
    '''
    Labels atoms as surface or bulk atoms according to their coordination
    relative to their bulk structure. If an atom's coordination is less than it
    normally is in a bulk, then we consider it a surface atom. We calculate the
    coordination using pymatgen's Voronoi algorithms.
    Note that if a single element has different sites within a bulk and these
    sites have different coordinations, then we consider slab atoms
    "under-coordinated" only if they are less coordinated than the most under
    undercoordinated bulk atom. For example:  Say we have a bulk with two Cu
    sites. One site has a coordination of 12 and another a coordination of 9.
    If a slab atom has a coordination of 10, we will consider it a bulk atom.
    Args:
        bulk_atoms      `ase.Atoms` of the bulk structure the surface was cut
                        from.
        surface_atoms   `ase.Atoms` of the surface
    Returns:
        tags    A list of 0's and 1's whose indices align with the atoms in
                `surface_atoms`. 0's indicate a bulk atom and 1 indicates a
                surface atom.
    '''
    # Initializations
    center_of_mass = calculate_center_of_mass(slab)
    bulk_cn_dict = calculate_coordination_of_bulk_atoms(bulk)
    voronoi_nn = VoronoiNN(tol=0.1)  # 0.1 chosen for better detection
    cnn = CrystalNN()
    tags = []
    for idx, site in enumerate(slab):

        # Tag as surface atom only if it's above the center of mass
        if site.frac_coords[2] > center_of_mass[2]:
            try:
                # Tag as surface if atom is under-coordinated
                cn = voronoi_nn.get_cn(slab, idx, use_weights=True)
                cn = round(cn, 5)
                cn = len(cnn.get_nn_info(slab, idx))
                if cn < min(bulk_cn_dict[site.species_string]):
                    tags.append(1)
                else:
                    tags.append(0)

            # Tag as surface if we get a pathological error
            except RuntimeError:
                tags.append(1)
        else:
            tags.append(0)

    return tags


def calculate_center_of_mass(struct):
    '''
    Determine the surface atoms indices from here
    '''
    weights = [site.species.weight for site in struct]
    center_of_mass = np.average(struct.frac_coords,
                                weights=weights, axis=0)
    return center_of_mass


def find_surface_atoms_by_height(slab, height_tol=2):
    '''
    As discussed in the docstring for `_find_surface_atoms_with_voronoi`,
    sometimes we might accidentally tag a surface atom as a bulk atom if there
    are multiple coordination environments for that atom type within the bulk.
    One heuristic that we use to address this is to simply figure out if an
    atom is close to the surface. This function will figure that out.
    Specifically:  We consider an atom a surface atom if it is within 2
    Angstroms of the heighest atom in the z-direction (or more accurately, the
    direction of the 3rd unit cell vector).
    Arg:
        surface_atoms   The surface where you are trying to find surface sites in
                        `ase.Atoms` format
    Returns:
        tags            A list that contains the indices of
                        the surface atoms
    '''
    unit_cell_height = np.linalg.norm(slab.lattice.matrix[2])
    scaled_positions = slab.frac_coords
    scaled_max_height = max(scaled_position[2] for scaled_position in scaled_positions)
    scaled_threshold_top = scaled_max_height - height_tol / unit_cell_height
    
    tags = [0 if scaled_position[2] < scaled_threshold_top else 1
            for scaled_position in scaled_positions]
        
    return tags


def slab_generator(entry_id, mmi, slab_size, vacuum_size, tol=0.1, MAPIKEY=None,
                   height_tol=2, min_lw=8, functional='GemNet-OC', count_undercoordination=False):
    
    """
    Generates bare slabs of all facets and terminations up to a max Miller index 
        (mmi). Returns a list of atoms objects. In each atoms object, also 
        include additional metadata for database management and post-processing.
    """
        
    if entry_id not in bulk_oxides_dict.keys():
        mprester = MPRester(MAPIKEY) if MAPIKEY else MPRester()
        bulk_entry = mprester.get_entry_by_material_id(entry_id, inc_structure=True,
                                                       conventional_unit_cell=True)[0]
    else:
        bulk_entry = bulk_oxides_dict[entry_id]
    
    bulk = bulk_entry.structure
    all_slabs = generate_all_slabs(bulk, mmi, slab_size, vacuum_size,
                                   center_slab=True, max_normal_search=1, symmetrize=True, tol=tol)
    
    comp = bulk.composition.reduced_formula
    
    atoms_slabs = []
    for slab in all_slabs:
        
        new_slab = slab.copy()
        tag_surface_atoms(bulk, new_slab, height_tol=height_tol, count_undercoordination=False)
                        
        # Get the symmetry operations to identify equivalent sites on both sides
        new_slab.add_site_property('original_index', [i for i, site in enumerate(new_slab)])
        sg = SpacegroupAnalyzer(new_slab)
        sym_slab = sg.get_symmetrized_structure()
        
        # Identify equivalent sites on other surface
        new_tags = []
        for site in sym_slab:
            if site.tag == 1:
                if site.original_index not in new_tags:
                    new_tags.append(site.original_index)

                for eq_site in sym_slab.find_equivalent_sites(site):
                    if eq_site.original_index not in new_tags:
                        new_tags.append(eq_site.original_index)
        
        # Tag both surfaces
        tags = [0 if i not in new_tags else 1 for i, site in enumerate(new_slab)]    
        new_slab.add_site_property('tag', tags)
        
        msuper = get_repeat_from_min_lw(new_slab, 8)
        new_slab = make_superslab_with_partition(new_slab, msuper)

        # get metadata 
        database = 'MP' if 'mp-' in entry_id else None
        metadata = write_metadata_json(new_slab, 'bare_slab', bulk_entry,  
                                       database=database, functional=functional)
        
        # ASE Atoms object format
        atoms = AseAtomsAdaptor.get_atoms(new_slab, **{'info': metadata})
        atoms.set_tags([site.tag for site in new_slab])
        atoms.set_constraint(FixAtoms([i for i, site in enumerate(new_slab) if site.tag == 0]))
        atoms_slabs.append(atoms)
        
    return atoms_slabs
