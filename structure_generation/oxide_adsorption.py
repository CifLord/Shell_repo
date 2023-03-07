from structure_generation.MXide_adsorption import MXideAdsorbateGenerator
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.surface import Slab
from pymatgen.core.structure import Molecule
from pymatgen.util.coord import all_distances, pbc_shortest_vectors

import numpy as np
import json, copy

# For slab saturation, 
Ox = Molecule(["O"], [[0,0,0]])
OH = Molecule(["O","H"], [[0,0,0], 
                          np.array([0, 0.99232, 0.61263])/np.linalg.norm(np.array([0, 0.99232, 0.61263]))*1.08540])

def surface_adsorption_saturation(slab):
    """
    Gets all adsorbed slab for a slab. Will always return 6 adslabs, 
        1 O* saturated slab and 5 OH saturated slabs. 4 of the OH  
        adslabs will have all OH molecules pointing in one of the 4 
        cardinal directions. 1 OH adslab will place OH* in such a way as 
        to minimize the O-H bondlengths with all O* and surface O atoms.
    """
    
    # get pmg slab
    init_slab = Slab.from_dict(json.loads(slab.info['pmg_slab']))
    
    # convert relaxed slab to Slab
    slab = AseAtomsAdaptor.get_structure(slab)
    relaxed_slab = Slab(slab.lattice, slab.species, slab.frac_coords,
                        init_slab.miller_index, init_slab.oriented_unit_cell, 
                        init_slab.shift, init_slab.scale_factor, 
                        site_properties=init_slab.site_properties)
        
    mxidegen = MXideAdsorbateGenerator(relaxed_slab, positions=['MX_adsites'], selective_dynamics=True)
    adslabs = mxidegen.generate_adsorption_structures(OH, coverage_list='saturated', consistent_rotation=True)
    adslabs.extend(mxidegen.generate_adsorption_structures(Ox, coverage_list='saturated', consistent_rotation=True))

    adslabs.append(max_OH_interaction_adsorption(mxidegen))
    
    return adslabs

def max_OH_interaction_adsorption(mxidegen, incr=100):
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

    satslab = mxidegen.slab.copy()
    for coord in mxidegen.MX_adsites:

        transformed_ads_list = mxidegen.get_transformed_molecule_MXides(OH, [np.deg2rad(deg) for deg in np.linspace(0, 360, incr)])

        all_OH_ave_dists = []
        for i, mol in enumerate(transformed_ads_list):
            slab = mxidegen.slab.copy()
            for site in mol:
                slab.append(site.species, coord+site.coords, coords_are_cartesian=True)

            shortest_vect = pbc_shortest_vectors(slab.lattice, slab[-1].frac_coords, 
                                                 [slab.lattice.get_fractional_coords(c) for c in surf_Osites])
            all_OH_dists = []
            for i, c in enumerate(surf_Osites):
                all_OH_dists.append(all_distances([slab[143].coords+shortest_vect[0][i]], [slab[143].coords])[0][0])
            all_OH_ave_dists.append(np.mean(all_OH_dists))

        all_OH_ave_dists, transformed_ads_list = zip(*sorted(zip(all_OH_ave_dists, transformed_ads_list)))

        for site in transformed_ads_list[0]:
            satslab.append(site.species, site.coords+coord, coords_are_cartesian=True)

    return satslab