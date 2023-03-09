from pymatgen.core.surface import SlabGenerator, generate_all_slabs, \
get_symmetrically_distinct_miller_indices, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import VoronoiNN, CrystalNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from collections import defaultdict
from ase.constraints import FixAtoms
from pymatgen.entries.computed_entries import ComputedStructureEntry

from database.generate_metadata import write_metadata_json

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

def tag_surface_atoms(bulk, slab, height_tol=2):
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
    '''
    
    height_tags = find_surface_atoms_by_height(slab, height_tol=height_tol)
    slab.add_site_property('tag', height_tags)


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


def get_repeat_from_min_lw(slab, min_lw):
    """
    Modified version of algorithm from adsorption.py for determining the super cell 
        matrix of the slab given min_lw. This will location the smallest super slab 
        cell with min_lw by including square root 3x3 transformation matrices
    """
    
    xlength = np.linalg.norm(slab.lattice.matrix[0])
    ylength = np.linalg.norm(slab.lattice.matrix[1])
    xrep = np.ceil(min_lw / xlength)
    yrep = np.ceil(min_lw / ylength)
    rtslab = slab.copy()
    rtslab.make_supercell([[1,1,0], [1,-1,0], [0,0,1]])
    rt_matrix = rtslab.lattice.matrix
    xlength_rt = np.linalg.norm(rt_matrix[0])
    ylength_rt = np.linalg.norm(rt_matrix[1])
    xrep_rt = np.ceil(min_lw / xlength_rt)
    yrep_rt = np.ceil(min_lw / ylength_rt)

    xrep = xrep*np.array([1,0,0]) if xrep*xlength < xrep_rt*xlength_rt else xrep_rt*np.array([1,1,0]) 
    yrep = yrep*np.array([0,1,0]) if yrep*ylength < yrep_rt*ylength_rt else yrep_rt*np.array([1,-1,0]) 
    zrep = [0,0,1]
    return [xrep, yrep, zrep]


def slab_generator(entry_id, mmi, slab_size, vacuum_size, tol=0.1, 
                   height_tol=2, min_lw=8, functional='GemNet-OC'):
    
    """
    Generates bare slabs of all facets and terminations up to a max Miller index 
        (mmi). Returns a list of atoms objects. In each atoms object, also 
        include additional metadata for database management and post-processing.
    """
    
    bulk_entry = [ComputedStructureEntry.from_dict(entry) for entry in bulk_oxides_20220621 \
                  if entry['entry_id'] == entry_id][0]
    bulk = bulk_entry.structure
    all_slabs = generate_all_slabs(bulk, mmi, slab_size, vacuum_size,
                                   center_slab=True, max_normal_search=1, symmetrize=True, tol=tol)
    
    comp = bulk.composition.reduced_formula
    
    atoms_slabs = []
    for slab in all_slabs:
        
        new_slab = slab.copy()
        tag_surface_atoms(bulk, new_slab, height_tol=height_tol)
                        
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
        
        msuper = get_repeat_from_min_lw(new_slab, min_lw)
        new_slab.make_supercell(msuper)

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
