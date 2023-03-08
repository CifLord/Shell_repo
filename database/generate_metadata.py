# from atomate.vasp.drones import VaspDrone 
import json, random, string, os, glob, shutil
import numpy as np

from pymatgen.core.structure import Structure, Molecule, Composition
from pymatgen.core.surface import Slab
from pymatgen.analysis.surface_analysis import *

from pymongo import MongoClient
from matplotlib import pylab as plt
        
def write_metadata_json(structure, calc_type, bulk_entry, fname=None, 
                        name_of_adsorbate=None, database=None, bulk_rid=None, 
                        slab_rid=None, ads_rid=None, additional_data=None, 
                        functional=None, custom_rid=None):
    
    rid = custom_rid if custom_rid else ''.join(random.choice(string.ascii_lowercase + \
                                                              string.ascii_uppercase \
                                                              + string.digits) for _ in range(20))
    
    metadata = {}
    d = json.dumps(structure.as_dict(), indent=True)
    if calc_type != 'adsorbate_in_a_box':
        bulk_formula = bulk_entry.composition.formula
        bulk_reduced_formula = bulk_entry.composition.reduced_formula
        bulk_composition = bulk_entry.composition.as_dict()
        bulk_chemsys = bulk_entry.composition.chemical_system
        bulk_energy = bulk_entry.energy

    if calc_type == 'adsorbate_in_a_box':
        name_of_adsorbate = structure.composition.to_pretty_string() \
        if not name_of_adsorbate else name_of_adsorbate 
        metadata = {'adsorbate': name_of_adsorbate, 'calc_type': calc_type, 
                    'rid': 'ads-%s' %(rid), 'init_pmg_structure': d, 'func': functional}

    if calc_type == 'bulk':
        metadata =  {'entry_id': bulk_entry.entry_id, 'database': database, 
                     'rid': 'bulk-%s' %(rid), 'calc_type': calc_type, 
                     'init_pmg_structure': d, 'func': functional}

    if calc_type == 'bare_slab':
         metadata = {'entry_id': bulk_entry.entry_id, 'database': database, 
                     'bulk_rid': bulk_rid, 'rid': 'slab-%s' %(rid), 
                     'miller_index': list(structure.miller_index), 
                     'bulk_reduced_formula': bulk_reduced_formula,
                     'bulk_formula': bulk_formula, 'bulk_composition': bulk_composition, 
                     'bulk_chemsys': bulk_chemsys, 'bulk_energy': bulk_energy,
                     'init_pmg_slab': d, 'calc_type': calc_type, 'func': functional}

    if calc_type == 'adsorbed_slab':
        metadata = {'entry_id': bulk_entry.entry_id, 'database': database, 
                    'adsorbate': name_of_adsorbate,
                    'bulk_rid': bulk_rid, 'slab_rid': slab_rid, 
                    'ads_rid': ads_rid, 'rid': 'adslab-%s' %(rid), 
                    'miller_index': list(structure.miller_index), 
                    'bulk_reduced_formula': bulk_reduced_formula,
                    'bulk_formula': bulk_formula, 'bulk_composition': bulk_composition, 
                    'bulk_chemsys': bulk_chemsys, 'bulk_energy': bulk_energy,
                    'init_pmg_slab': d, 'calc_type': calc_type, 'func': functional}

    if additional_data:
        metadata.update(additional_data)
    if fname:
        with open(os.path.join(fname, 'metadata.json'), 'w') as outfile:
            outfile.write(json.dumps(metadata, indent=True))
        outfile.close()
    else:
        return metadata