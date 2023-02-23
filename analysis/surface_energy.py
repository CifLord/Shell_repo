from pymatgen.analysis.surface_analysis import SlabEntry
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.io.ase import AseAtomsAdaptor

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