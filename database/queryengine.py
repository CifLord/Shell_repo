from torch_geometric.data.data import Data
from pymatgen.db import QueryEngine
import json, random, string, os, glob, shutil
import numpy as np

from pymatgen.core.structure import Structure, Molecule, Composition
from pymatgen.core.surface import Slab
from pymatgen.analysis.surface_analysis import *
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.ext.matproj import MPRester

from pymongo import MongoClient
from pymongo.server_api import ServerApi
from matplotlib import pylab as plt

from database.generate_metadata import NpEncoder
from analysis.surface_energy import get_slab_entry


kB = 1.380649 * 10**(-23)
JtoeV = 6.242e+18


class SurfaceQueryEngine(QueryEngine):

    """
    API that interacts with a database of OC22 predictions for OER. Queries the needed data to produce
        analysis and plots such as rxn diagarams, activity maps, phase diagrams, stability diagrams, 
        Wulff shapes, etc.We should have 4 types of documents, each formatted in a standardized fashion: 
        Adslab, Slab, Adsorbate, and Bulk. All entries should have a random ID and the initial pre-relaxed structure
        - Adsorbate: Document describing the VASP calculation of an adsorbate in a box. All documents 
            must have a distinct ID associated with it. In addition to standard inputs parsed by the 
            TaskDrone, will also have the following metadata:
            {'adsorbate': name_of_adsorbate, 'adsorbate_id', 
            'calc_type': 'adsorbate_in_a_box', 'rid': 'ads-BGg3rg4gerG6'}
        - Bulk: Document describing the VASP calculation of a standard bulk structure. In addition to 
            standard inputs parsed by the TaskDrone, will also have the following metadata:
            {'entry_id': mpid, auid, etc, 'database': 'Materials_Project, AFLOW, OQMD etc', 
            rid: 'bulk-rgt4h65u45gg', 'calc_type': 'bulk'}
        - Slab: Document describing the VASP calculation of a Slab. In addition to 
            standard inputs parsed by the TaskDrone, will also have the following metadata:
            {'entry_id': mpid, auid, etc, 'database': 'Materials_Project, AFLOW, OQMD etc', 
            'bulk_rid': 'bulk-rgt4h65u45gg', 'rid': 'slab-eege4h4herhe4', 'miller_index': (1,1,1), 
            'Slab': pmg_slab_as_dict, 'calc_type': 'bare_slab'}
        - AdSlab: Document describing the VASP calculation of an adsorbed slab. In addition to 
            standard inputs parsed by the TaskDrone, will also have the following metadata:
            {'entry_id': mpid, auid, etc, 'database': 'Materials_Project, AFLOW, OQMD etc', 
            'bulk_rid': 'bulk-rgt4h65u45gg', 'slab_rid': 'slab-eege4h4herhe4', 
            'ads_rid': 'ads-BGg3rg4gerG6', 'rid': 'adslab-reg3g53g3h4h2hj204', 
            'miller_index': (1,1,1), 'Slab': pmg_slab_as_dict, 'calc_type': 'bare_slab'}
    """
    def __init__(self, MAPIKEY=None):

        conn = MongoClient(host="mongodb://127.0.0.1", port=27017)
        db = conn.get_database('richardtran415')
        surface_properties = db['Shell']

        self.surface_properties = surface_properties
        self.encoder = NpEncoder()
        self.MAPIKEY = MAPIKEY
        self.mprester = MPRester(MAPIKEY)
        self.surf_plt = None
        self.slab_entries = None
        
        # Total DFT energy of adsorbates in a box
        self.ads_in_a_box = {'C': -.12621228E+01, 'CO': -.14791073E+02, 'H': -.11171013E+01, 
                             'H2O': -.14230373E+02, 'HO': -.77495501E+01, 'HO2': -.13285977E+02, 
                             'N': -.31236799E+01, 'O': -.15469677E+01, 'O2': -9.88286631,
                             'H2': -.67714828E+01, 'N2': -.16618080E+02}
        # References for the adsorbate
        self.mol_entry = {'OH': ComputedEntry('OH', self.ads_in_a_box['H2O']-0.5*self.ads_in_a_box['H2']),
                          'O': ComputedEntry('O', self.ads_in_a_box['H2O']-self.ads_in_a_box['H2']),
                          'OOH': ComputedEntry('OOH', 2*self.ads_in_a_box['H2O']-1.5*self.ads_in_a_box['H2'])}
        # Correction terms for Gibbs adsorption energy (see OC22 paper). 
        # Used to turn DFT adsorption energy into Gibbs adsorption energy
        self.Gcorr = {'OH': 0.26, 'O': -0.03, 'OOH': 0.22}
                
    def insert_data_object(self, dat):
        
        d = dat.to_dict()
        new_dat = {}
        for k in d.keys():
            if 'Tensor' == type(d[k]).__name__:
                new_dat[k] = d[k].tolist()
            else:
                new_dat[k] = d[k]
        self.surface_properties.insert_one(new_dat)
        
    def get_slab_entries(self, criteria, relaxed=True):
        
        clean_dict, adslab_dict = {}, {}
        dat_list = [Data.from_dict(doc) for doc in self.surface_properties.find(criteria)]
        
        slab_entries = {}
        # get the clean slabs first to build the adslabs
        for dat in dat_list:
            if 'adslab-' not in dat.rid:
                slab_entries[dat.rid] = get_slab_entry(dat, relaxed=relaxed, 
                                                       data={'rid': dat.rid})
        
        # now get the adslab entries
        for dat in dat_list:
            if 'adslab-' in dat.rid:
                
                if dat.slab_rid not in slab_entries.keys():
                    doc = self.surface_properties.find_one({'rid': dat.slab_rid, 'adsorbate': dat.adsorbate})
                    if not doc:
                        continue
                    clean_dat = Data.from_dict(doc)
                    slab_entries[dat.slab_rid] = get_slab_entry(clean_dat, relaxed=relaxed,
                                                                data={'rid': dat.rid})
                    
                entry = get_slab_entry(dat, relaxed=relaxed, 
                                       clean_slab_entry=slab_entries[dat.slab_rid],
                                       ads_entries=[self.mol_entry[dat.adsorbate]],
                                       data={'rid': dat.rid, 'adsorbate': dat.adsorbate})
                entry.Nads_in_slab = dat.nads*sum(self.mol_entry[dat.adsorbate].composition.as_dict().values())
                slab_entries[dat.rid] = entry
                
        self.slab_entries = slab_entries
        return slab_entries
        
    def get_surfe_plotter(self, criteria=None, relaxed=True):
        
        if not criteria:
            slab_entries_dict = self.slab_entries_dict
        else:
            slab_entries_dict = self.get_slab_entries(criteria, relaxed=relaxed)
        
        entry_id_to_surfplt_dict = {}
        for rid in slab_entries_dict.keys():
            slabentry = slab_entries_dict[rid]
            if slabentry.entry_id not in entry_id_to_surfplt_dict.keys():
                entry_id_to_surfplt_dict[slabentry.entry_id] = []
            entry_id_to_surfplt_dict[slabentry.entry_id].append(slabentry)
            
        surfplt_dict = {}
        for entry_id in entry_id_to_surfplt_dict.keys():
            slabentries = entry_id_to_surfplt_dict[entry_id]
            hkl_color_dict = {}
            for slabentry in slabentries:
                # color code Miller indices
                hkl = tuple(slabentry.miller_index)
                if hkl not in hkl_color_dict:
                    hkl_color_dict[hkl] = random_color_generator()
                slabentry.color = hkl_color_dict[hkl]

            bulk_entry = self.mprester.get_entry_by_material_id(entry_id, inc_structure=True, 
                                                                conventional_unit_cell=True)
            # get the slab entries and preset their surface energies as functions of delta mu_O only
            ref_entries = get_ref_entries(bulk_entry, MAPIKEY=self.MAPIKEY)
            for slabentry in slabentries:
                preset_slabentry_se(slabentry, bulk_entry, ref_entries=ref_entries)
            
            # Get the SurfaceEnergyPlotter object for doing surface energy analysis
            for gas_entry in ref_entries:
                if gas_entry.composition.reduced_formula == 'O2':
                    break
            surfplt_dict[entry_id] = SurfaceEnergyPlotter(slabentries, bulk_entry, ref_entries=[gas_entry])
        
        self.surf_plt = surfplt_dict
        return surfplt_dict
    
    def get_e_transfer_corr(self, T=0, U=0, pH=0):
        proton_activity = 10**(-1*pH)
        return -1*U + kB*T * JtoeV * np.log(proton_activity)
    
    def get_gibbs_adsorption_energies(self, adsorbate, criteria=None, T=0, U=0, pH=0, P=0.1):
            
        dmuO = get_dmu(T, P)

        if not criteria:
            surfplt_dict = self.surf_plt
        else:
            surfplt_dict = self.get_surfe_plotter(criteria, relaxed=False)
        
        Gads_dict = {}
        for mpid in surfplt_dict.keys():
            surfplt = surfplt_dict[mpid]
            Gads_dict[mpid] = {}
            for hkl in surfplt.all_slab_entries.keys():
                
                entry = surfplt.get_stable_entry_at_u(hkl, delu_dict={Symbol('delu_O'): dmuO})[0]
                slab_rid = entry.data['rid']
                
                Eads = []
                for entry in self.slab_entries.values():
                    if 'adslab-' in entry.data['rid']:
                        if entry.data['adsorbate'] == adsorbate:
                            Eads.append(entry.gibbs_binding_energy(eads=True))
                            
                Gads_dict[mpid][hkl] = sorted(Eads)[0] + self.Gcorr[adsorbate]
                
        return Gads_dict
        
    def build_rxn_diagram(self, criteria=None, T=0, U=0, pH=0, P=0.1, plot_ideal=True):
        G1 = 0
        G2_dict = self.get_gibbs_adsorption_energies('OH', criteria=criteria, T=T, U=U, pH=pH, P=P)
        G3_dict = self.get_gibbs_adsorption_energies('O', criteria=criteria, T=T, U=U, pH=pH, P=P)
        G4_dict = self.get_gibbs_adsorption_energies('OOH', criteria=criteria, T=T, U=U, pH=pH, P=P)
        G5 = -1*(self.ads_in_a_box['O2']/2)
        
        rxn_diagram_dict = {}
        for mpid in G2_dict.keys():
            rxn_diagram_dict[mpid] = {}
            for hkl in G2_dict[mpid].keys():
                G2 = G2_dict[mpid][hkl]
                G3 = G3_dict[mpid][hkl]
                G4 = G4_dict[mpid][hkl]
                
                plots = []
                plots.append(plt.plot([0, 1], [G1, G1], 'k-'))
                plots.append(plt.plot([1, 1], [G1, G2], 'k-'))
                plots.append(plt.plot([1, 2], [G2, G2], 'k-'))
                plots.append(plt.plot([2, 2], [G2, G3], 'k-'))
                plots.append(plt.plot([2, 3], [G3, G3], 'k-'))
                plots.append(plt.plot([3, 3], [G3, G4], 'k-'))
                plots.append(plt.plot([3, 4], [G4, G4], 'k-'))
                plots.append(plt.plot([4, 4], [G4, G5], 'k-'))
                plots.append(plt.plot([4, 5], [G5, G5], 'k-'))
                
                if plot_ideal:
                    plots.append(plt.plot([0, 1], [G1, G1], 'r-'))
                    plots.append(plt.plot([1, 1], [G1, G5/4], 'r-'))
                    plots.append(plt.plot([1, 2], [G5/4, G5/4], 'r-'))
                    plots.append(plt.plot([2, 2], [G5/4, G5/2], 'r-'))
                    plots.append(plt.plot([2, 3], [G5/2, G5/2], 'r-'))
                    plots.append(plt.plot([3, 3], [G5/2, G5*(3/4)], 'r-'))
                    plots.append(plt.plot([3, 4], [G5*(3/4), G5*(3/4)], 'r-'))
                    plots.append(plt.plot([4, 4], [G5*(3/4), G5], 'r-'))
                    plots.append(plt.plot([4, 5], [G5, G5], 'r-'))

                rxn_diagram_dict[mpid][hkl] = plots
                        
        return rxn_diagram_dict