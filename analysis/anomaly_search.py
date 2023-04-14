from pymatgen.core.structure import Structure, StructureError
from pymatgen.core.surface import Slab
from database.queryengine import get_slab_object
from torch_geometric.data.data import Data


def is_dissociation(entry, s=None):
    if not s:
        try:
            s = get_slab_object(Data.from_dict(entry), relaxed=True)
        except KeyError:
            return True
        
    if slab.site_properties['surface_properties'].count('adsorbate') == 1:
        return False
    
    adsname = entry['adsorbate']
    blengths = {('O', 'O'): 1.7, ('H', 'O'): 1.21}

    diss_atoms = []
    for site in s:
        if site.surface_properties == 'adsorbate':
            nn = [n for n in s.get_neighbors(site, 2, include_index=True) if n.species_string in ['O', 'H']]
            attached = []
            for n in nn:
                if n.surface_properties == 'adsorbate':
                    d = site.distance(n)
                    attached.append(blengths[tuple(sorted([n.species_string, site.species_string]))] < d)
            diss_atoms.append(all(attached))
    return any(diss_atoms)

                
def is_desorbed(entry, desorb_tol=3, s=None):
    try:
        s = get_slab_object(Data.from_dict(entry), relaxed=True)
    except KeyError:
        return True

    adsname = entry['adsorbate']
    ccoords = sorted([site.coords[2] for site in s if site.surface_properties == 'surface'])
    adscoords = sorted([site.coords[2] for site in s if site.surface_properties == 'adsorbate'])
    return  adscoords[0] - ccoords[-1] > desorb_tol or adscoords[0] - ccoords[0] < 0


def is_invalid(entry, s=None):
    if not s:
        try:
            s = get_slab_object(Data.from_dict(entry), relaxed=True)
        except KeyError:
            return True
        
    try:
        Structure(s.lattice, s.species, s.frac_coords, validate_proximity=True)
        return False
    except StructureError:
        return True
    
    
def wrong_side(entry, s=None):
    if not s:
        try:
            s = get_slab_object(Data.from_dict(entry), relaxed=True)
        except KeyError:
            return True

    coords_c = [site.coords[2] for site in s if 
                    site.surface_properties != 'adsorbate']
    ads_coords_c = [site.coords[2] for site in s if site.surface_properties == 'adsorbate']
    if min(ads_coords_c) > 0.5:
        return False
    else:
        return True
    
    
def filter_entries(entries, check_dissociation=True, check_desorbed=True, 
                   check_invalid=True, check_wrong_side=True, get_bad_entries=False):
    val_entries = []
    for entry in entries:
        if 'adslab' in entry['rid']:
            if check_dissociation and is_dissociation(entry):
                if get_bad_entries:
                    val_entries.append(entry)
                continue
            if check_desorbed and is_desorbed(entry):
                if get_bad_entries:
                    val_entries.append(entry)
                continue
            if check_wrong_side and wrong_side(entry):
                if get_bad_entries:
                    val_entries.append(entry)
                continue

        if check_invalid and is_invalid(entry):
            if get_bad_entries:
                val_entries.append(entry)
            continue
        if not get_bad_entries:
            val_entries.append(entry)
        
    return val_entries