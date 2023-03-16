from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.surface import Slab
from pymatgen.core.structure import Molecule
from pymatgen.util.coord import all_distances, pbc_shortest_vectors
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.structure_matcher import StructureMatcher

from database.generate_metadata import write_metadata_json
from structure_generation.MXide_adsorption import MXideAdsorbateGenerator

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
bulk_oxides_20220621 = json.load(open(f.replace(f.split('/')[-1], 'bulk_oxides_20220621.json'), 'rb'))

# For slab saturation, 
Ox = Molecule(["O"], [[0,0,0]])
OH = Molecule(["O","H"], [[0,0,0], 
                          np.array([0, 0.99232, 0.61263])/\
                          np.linalg.norm(np.array([0, 0.99232, 0.61263]))*1.08540])

# MXideGenerator is set up to bind all O sites in the adsorbate one by one. To save 
# time and avoid this, we will temporarily set the middle O site in OOH to N so that 
# only the O site at (0,0,0) is bonded. Then we will substite N with O back into the adslab
OOH_up = Molecule(["O","N","H"], [[0, 0, 0], [-1.067, -0.403, 0.796],[-0.696, -0.272, 1.706]])
OOH_down = Molecule(["O","N","H"], [[0,0,0], [-1.067, -0.403, 0.796], [-1.84688848, -0.68892498, 0.25477651]])
ads_dict = {'O': [Ox], 'OH': [OH], 'OOH': [OOH_down, OOH_up]}

def surface_adsorption(slab_data, functional='GemNet-OC', coverage_list=[1]):
    """
    Gets all adsorbed slab for a slab. Will always return 6 adslabs, 
        1 O* saturated slab and 5 OH saturated slabs. 4 of the OH  
        adslabs will have all OH molecules pointing in one of the 4 
        cardinal directions. 1 OH adslab will place OH* in such a way as 
        to minimize the O-H bondlengths with all O* and surface O atoms.
    """
    
    # get bulk entry
    bulk_entry = [ComputedStructureEntry.from_dict(entry) for entry in bulk_oxides_20220621 \
                  if entry['entry_id'] == slab_data.entry_id][0]
    
    # get pmg slab
    init_slab = Slab.from_dict(json.loads(slab_data.init_pmg_slab))
    
    # I am assuming the atomic_numbers, pos, and 
    # cell are corresponding to the relaxed slab?
    # convert relaxed slab to Slab
    atoms = Atoms(slab_data.atomic_numbers,
                  positions=slab_data.pos_relaxed,
                  tags=slab_data.tags,
                  cell=slab_data.cell.squeeze(), pbc=True)
    relaxed_slab = AseAtomsAdaptor.get_structure(atoms)
    relaxed_slab = Slab(relaxed_slab.lattice, relaxed_slab.species,
                        relaxed_slab.frac_coords, init_slab.miller_index,
                        init_slab.oriented_unit_cell, init_slab.shift, 
                        init_slab.scale_factor, site_properties=init_slab.site_properties)

    mxidegen = MXideAdsorbateGenerator(init_slab, positions=['MX_adsites'], 
                                       selective_dynamics=True, repeat=[1,1,1])
    
    props = mxidegen.slab.site_properties
    for k in props.keys():
        if k not in relaxed_slab.site_properties.keys():
            relaxed_slab.add_site_property(k, props[k])
    
    all_adslabs = []
    relaxed_adslabs = {}
    for adsname in ads_dict.keys():
        relaxed_adslabs[adsname] = []
        if coverage_list == 'saturated' and adsname == 'OOH':
            continue
        for mol in ads_dict[adsname]:
            relslab = relaxed_slab.copy()
            if coverage_list == 'saturated':
                adslabs = adsorb_saturate(relslab, mxidegen, mol)
            else:
                adslabs = adsorb_one(relslab, mxidegen, mol)
                
            for adslab in adslabs:
                adslab.replace_species({Element('N'): Element('O')})
                setattr(adslab, 'adsorbate', adsname)
            relaxed_adslabs[adsname].extend(adslabs)

    if coverage_list == 'saturated':
        OHstar = max_OH_interaction_adsorption(relslab, mxidegen)
        setattr(OHstar, 'adsorbate', 'OH')
        relaxed_adslabs['OH'].append(OHstar)
        
    # # superimpose adsites onto relaxed_slab
    # relaxed_adslabs = {}
    # for adslab in all_adslabs:
    #     rel_slabs = relaxed_slab.copy()
    #     for site in adslab:
    #         if site.surface_properties == 'adsorbate':
    #             rel_slabs.append(site.species_string, site.frac_coords, properties=site.properties)
    #     setattr(rel_slabs, 'adsorbate', adslab.adsorbate)
    #     if rel_slabs.adsorbate not in relaxed_adslabs.keys():
    #         relaxed_adslabs[rel_slabs.adsorbate] = []
    #     relaxed_adslabs[rel_slabs.adsorbate].append(rel_slabs)
        
    sm = StructureMatcher()
    relaxed_adslabs_list = []
    for ads in relaxed_adslabs.keys():
        relaxed_adslabs_list.extend([g[0] for g in sm.group_structures(relaxed_adslabs[ads])])
    
    # Build list of Atoms objects
    adslab_atoms = []
    for adslab in relaxed_adslabs_list:
        
        # name adsorbates
        if adslab.adsorbate == 'O':
            nads = len([site for site in adslab if site.surface_properties == 'adsorbate'])
        elif adslab.adsorbate == 'OH':
            nads = len([site for site in adslab if site.surface_properties == 'adsorbate'])/2
        elif adslab.adsorbate == 'OOH':
            nads = len([site for site in adslab if site.surface_properties == 'adsorbate'])/3

        new_tags = [] 
        for site in adslab:
            if site.tag == None:
                new_tags.append(2)
            elif site.frac_coords[2] < 0.5:
                new_tags.append(0)
            else:
                new_tags.append(site.tag)
        adslab.add_site_property('tag', new_tags)

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

def adsorb_one(slab, mxidegen, adsorbate):

    transformed_ads_list = mxidegen.get_transformed_molecule_MXides\
    (adsorbate, [np.deg2rad(deg) for deg in np.linspace(0, 360, 4)])

    all_adslabs = []
    for coord in mxidegen.MX_adsites:
        for i, mol in enumerate(transformed_ads_list):
            adslab = slab.copy()
            for site in mol:
                adslab.append(site.species, coord+site.coords, coords_are_cartesian=True,
                              properties={'original_index': None, 'supercell': None, 
                                          'selective_dynamics': [True, True, True], 
                                          'bulk_equivalent': None, 'site_type': None, 'tag': 2, 
                                          'surface_properties': 'adsorbate', 'bulk_wyckoff': None})
            all_adslabs.append(adslab)

    return all_adslabs

def adsorb_saturate(slab, mxidegen, adsorbate):

    transformed_ads_list = mxidegen.get_transformed_molecule_MXides\
    (adsorbate, [np.deg2rad(deg) for deg in np.linspace(0, 360, 4)])

    all_adslabs = []
    for i, mol in enumerate(transformed_ads_list):
        adslab = slab.copy()
        for coord in mxidegen.MX_adsites:
            for site in mol:
                adslab.append(site.species, coord+site.coords, coords_are_cartesian=True, 
                              properties={'original_index': None, 'supercell': None, 
                                          'selective_dynamics': [True, True, True], 
                                          'bulk_equivalent': None, 'site_type': None, 'tag': 2, 
                                          'surface_properties': 'adsorbate', 'bulk_wyckoff': None})
        all_adslabs.append(adslab)

    return all_adslabs

def max_OH_interaction_adsorption(s, mxidegen, incr=100):
    """
    Algorithm to saturate a surface with OH by rotating all OH 
        molecules in such a way to minimize H-O bondlengths 
        between H* and all surface O and O* sites. This minimization 
        will hopefully get us a relatively stable config for adsorption.
    """
    
    # Get all surface O sites and potential O* sites
    surf_Osites = copy.copy(mxidegen.MX_adsites)
    for site in mxidegen.slab:
        if all([site.surface_properties == 'surface', 
                site.frac_coords[-1] > 0.5, site.species_string == 'O']):
            surf_Osites.append(site.coords)

    transformed_ads_list = mxidegen.get_transformed_molecule_MXides\
    (OH, [np.deg2rad(deg) for deg in np.linspace(0, 360, incr)])

    satslab = s.copy()
    for coord in mxidegen.MX_adsites:
        all_OH_ave_dists = []
        
        for i, mol in enumerate(transformed_ads_list):
            slab = mxidegen.slab.copy()
            for site in mol:
                slab.append(site.species, coord+site.coords, coords_are_cartesian=True)

            shortest_vect = pbc_shortest_vectors(slab.lattice, slab[-1].frac_coords, 
                                                 [slab.lattice.get_fractional_coords(c) \
                                                  for c in surf_Osites])
            all_OH_dists = []
            for i, c in enumerate(surf_Osites):
                # Get all distances between the H site of the adsorbate
                # and its surrrounding O sites at the surface
                all_OH_dists.append(all_distances([slab[-1].coords+shortest_vect[0][i]],
                                                  [slab[-1].coords])[0][0])
            all_OH_ave_dists.append(np.mean(all_OH_dists))

        all_OH_ave_dists, transformed_ads_list = zip(*sorted(zip(all_OH_ave_dists, 
                                                                 transformed_ads_list)))

        for site in transformed_ads_list[0]:
            satslab.append(site.species, site.coords+coord, coords_are_cartesian=True, 
                           properties={'tag': 2, 'surface_properties': 'adsorbate',
                                       'selective_dynamics': [True, True, True]})

    return satslab