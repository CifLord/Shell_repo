from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.surface import Slab
from pymatgen.core.structure import Molecule
from pymatgen.util.coord import all_distances, pbc_shortest_vectors
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.adsorption import AdsorbateSiteFinder

from mp_api.client import MPRester
from database.generate_metadata import write_metadata_json
from ase.constraints import FixAtoms
from ase import Atoms

import numpy as np
import json, copy


"""
Metadata and additional API information for adsorbed slabs:
    entry_id (str): Materials Project ID of the corresponding bulk
    database (str): Materials database where the bulk was obtained from
    adsorbate (str): Name_of_adsorbate
    nads (int): Number of adsorbate molecules
    bulk_rid (str): 20 digit random ID of the corresponding bulk e.g. 
        bulk-h4qfsgj3432kl113DVpD
    slab_rid (str): 20 digit random ID of the corresponding bare slab
        e.g. slab-t43frgfgy5lFh3f3CVGD
    ads_rid (str): 20 digit random ID of the corresponding gas phase 
        reference of the adsorbate e.g. ads-a03Wlk2205GFXqu95ATs
    rid (str): 20 digit random ID of the adsorbed slab e.g. 
        adslab-zx668jolRrGdhqwWqA47
    miller_index (tuple): Miller index of the facet 
    bulk_formula (str): Reduced formula of the corresponding bulk 
    bulk_composition (dict): Composition of the bulk with elements 
        as keys and the number of atoms as values 
    bulk_chemsys (str): Elemental components of the bulk e.g. for 
        LiFe(PO4), the chemsys is 'Li-Fe-P-O'
    pmg_slab (dict): Pymatgen slab object as a dictionary. Contains useful 
        information about miller index, scale factor, bulk wyckoff positions, etc. 
    calc_type (str): What kidn of system is this calculation for 
        ('bare_slab', 'adsorbed_slab', 'adsorbate_in_a_box', 'bulk') 
    func (str): The DFT functional used or the method used to obtain the 
        energy/relaxation data (Beef-vdW, PBE, PBEsol, rPBE, GemNet-OC (for ML) etc...)
"""

from database import generate_metadata
f = generate_metadata.__file__
bulk_bimetallics_20230921 = json.load(open(f.replace(f.split('/')[-1], 'bulk_bimetallics_20230921.json'), 'rb'))
bulk_bimetallics_dict = {entry['entry_id']: ComputedStructureEntry.from_dict(entry) \
                         for entry in bulk_bimetallics_20230921}
ads_dict = {'O': [Molecule(["O"], [[0,0,0]])], 'H': [Molecule(["H"], [[0,0,0]])], 
            'N': [Molecule(["N"], [[0,0,0]])], 'C': [Molecule(["C"], [[0,0,0]])]}

MAPIKEY='HO6BA47aEuOPR8Puc0qshrc6G9596FNa' #Liqiang Key

def strip_entry(input_string):
    # when get the entry_id in lmdb, we may get the str like mp-0000-GGA
    # this method strip the str to mp-0000
    second_hyphen_index = input_string.find('-', input_string.find('-') + 1)
    if second_hyphen_index != -1:
        return (input_string[:second_hyphen_index])
    else:
        return (input_string)

def surface_adsorption(slab_data, functional='GemNet-OC', coverage_list=[1], 
                       MAPIKEY=MAPIKEY, ads_dict=ads_dict):
    """
    Gets all adsorbed slab for a slab. Will always return 6 adslabs, 
        1 O* saturated slab and 5 OH saturated slabs. 4 of the OH  
        adslabs will have all OH molecules pointing in one of the 4 
        cardinal directions. 1 OH adslab will place OH* in such a way as 
        to minimize the O-H bondlengths with all O* and surface O atoms.
    """
    
    # get bulk entry
    entry_id=slab_data.entry_id
    entry_id=strip_entry(entry_id)
    if entry_id not in bulk_bimetallics_dict.keys():
        mprester = MPRester(MAPIKEY) if MAPIKEY else MPRester()
        bulk_entry = mprester.get_entry_by_material_id(entry_id, inc_structure=True,
                                                       conventional_unit_cell=True)
        bulk_entry=bulk_entry[0] # new key need to index
    else:
        bulk_entry = bulk_bimetallics_dict[entry_id]
    
    # get pmg slab
    init_slab = Slab.from_dict(json.loads(slab_data.init_pmg_slab))
    
    # I am assuming the atomic_numbers, pos, and 
    # cell are corresponding to the relaxed slab?
    # convert relaxed slab to Slab
    atoms = Atoms(slab_data.atomic_numbers,
                  positions=slab_data.pos,# Tentatively, changed to pox---slab_data.pos_relaxed
                  tags=slab_data.tags,
                  cell=slab_data.cell.squeeze(), pbc=True)
    relaxed_slab = AseAtomsAdaptor.get_structure(atoms)
    relaxed_slab = Slab(relaxed_slab.lattice, relaxed_slab.species,
                        relaxed_slab.frac_coords, init_slab.miller_index,
                        init_slab.oriented_unit_cell, init_slab.shift, 
                        init_slab.scale_factor, site_properties=init_slab.site_properties)

    adsitegen = AdsorbateSiteFinder(init_slab, selective_dynamics=False, height=0.9)
    
    props = adsitegen.slab.site_properties
    for k in props.keys():
        if k not in relaxed_slab.site_properties.keys():
            relaxed_slab.add_site_property(k, props[k])
    
    all_adslabs = []
    relaxed_adslabs = {'N': [], 'O': [], 'H': [], 'C': []}
    adslabs = adsitegen.generate_adsorption_structures(Molecule(["O"], [[0,0,0]]), 
                                                       repeat=[1,1,1])
    
    for adsname in ['N', 'O', 'H', 'C']:
        for adslab in adslabs:
            relslab = adslab.copy()
            relslab.replace_species({Element('O'): Element(adsname)})
            setattr(relslab, 'adsorbate', adsname)
            relaxed_adslabs[adsname].append(relslab)
                    
    sm = StructureMatcher()
    relaxed_adslabs_list = []
    for ads in relaxed_adslabs.keys():
        relaxed_adslabs_list.extend([g[0] for g in sm.group_structures(relaxed_adslabs[ads])])
    
    # Build list of Atoms objects
    adslab_atoms = []
    for adslab in relaxed_adslabs_list:
        
        nads = 1
        new_tags = [] 
        for site in adslab:
            if site.tag == None:
                new_tags.append(2)
            elif site.frac_coords[2] < 0.5:
                new_tags.append(0)
            else:
                new_tags.append(site.tag)
        adslab.add_site_property('tag', new_tags)
        adsite = [site for site in adslab if site.species_string in ['O', 'N', 'H', 'C']][0]
        adslab.remove_species([adsite.species_string])
        adslab.append(adsite.species, adsite.frac_coords, 
                      properties={'original_index': None, 'supercell': None, 
                                  'selective_dynamics': [True, True, True], 
                                  'bulk_equivalent': None, 'site_type': None, 'tag': 2, 
                                  'surface_properties': 'adsorbate', 'bulk_wyckoff': None})

        # get metadata 
        database = 'MP' if 'mp-' in bulk_entry.entry_id else None
        metadata = write_metadata_json(adslab, 'adsorbed_slab', bulk_entry, 
                                       name_of_adsorbate=adslab.adsorbate,
                                       database=database, slab_rid=slab_data.rid,
                                       functional=functional, additional_data={'nads': nads})
        
        # ASE Atoms object format, set up selective dynamics so  
        # that only the top surface and adsorbate relax this time
        atoms = AseAtomsAdaptor.get_atoms(adslab, **{'info': metadata})
        atoms.set_tags(new_tags)
        atoms.set_constraint(FixAtoms([i for i, site in enumerate(adslab) if site.tag == 0]))
        adslab_atoms.append(atoms)
    
    return adslab_atoms

# def adsorb_one(slab, adsitegen, adsorbate):
    
#     adslabs = adslabgen.generate_adsorption_structures(adsorbate, repeat=[1,1,1])


#     all_adslabs = []
#     for coord in adsitegen.MX_adsites:
#         for i, mol in enumerate(transformed_ads_list):
#             adslab = slab.copy()
#             for site in mol:
#                 adslab.append(site.species, coord+site.coords, coords_are_cartesian=True,
#                               properties={'original_index': None, 'supercell': None, 
#                                           'selective_dynamics': [True, True, True], 
#                                           'bulk_equivalent': None, 'site_type': None, 'tag': 2, 
#                                           'surface_properties': 'adsorbate', 'bulk_wyckoff': None})
#             all_adslabs.append(adslab)

#     return all_adslabs