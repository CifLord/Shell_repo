from torch_geometric.data.data import Data
from pymatgen.db import QueryEngine
import numpy as np

from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.cli.pmg_query import MPRester

from pymongo import MongoClient
from matplotlib import pylab as plt

from database.generate_metadata import NpEncoder
from analysis.surface_energy import *
from analysis.surface_analysis import *

kB = 1.380649 * 10**(-23)
JtoeV = 6.242e+18

bulk_oxides_dict = json.load(open('bulk_oxides_20220621.json', 'r'))
bulk_oxides_dict = {entry['entry_id']: entry for entry in bulk_oxides_dict}

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
    def __init__(self, MAPIKEY=None, collection='Shell05022023'):

        conn = MongoClient(host="mongodb://127.0.0.1", port=27017)
        db = conn.get_database('richardtran415')
        surface_properties = db[collection]

        self.surface_properties = surface_properties
        self.encoder = NpEncoder()
        self.MAPIKEY = MAPIKEY
        self.mprester = None # MPRester(MAPIKEY)
        self.surf_plt = None
        self.slab_entries = None
        self.rxn_diagram_xlabels = ['$2H_2O_{(l)}+*$', '$OH*$', '$O*$', '$OOH*$', '$O_{2(g)}+*$']
        self.rxn_diagram_xticks = [0.5, 1.5, 2.5, 3.5, 4.5]
        
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
        
    def find_data_objects(self, criteria):
        docs = []
        for entry in self.surface_properties.find(criteria):
            docs.extend(entry['slabs'].values())
        return [Data.from_dict(doc) for doc in docs]
        
    def get_slab_entries(self, criteria, dat_list=None, relaxed=True):
        
        clean_dict, adslab_dict = {}, {}
        dat_list = dat_list if dat_list else self.find_data_objects(criteria)
        
        slab_entries = {}
        # get the clean slabs first to build the adslabs
        for dat in dat_list:
            if 'adslab-' not in dat.rid:
                slab_entries[dat.rid] = get_slab_entry(dat, relaxed=relaxed, 
                                                       data={'mpid': dat.entry_id})
        
        # now get the adslab entries
        for dat in dat_list:
            if 'adslab-' in dat.rid:
                
                if dat.slab_rid not in slab_entries.keys():
                    doc = self.surface_properties.find_one({'mpid': dat.entry_id, 'adsorbate': dat.adsorbate})
                    if not doc:
                        continue
                    clean_dat = Data.from_dict(doc)
                    slab_entries[dat.slab_rid] = get_slab_entry(clean_dat, relaxed=relaxed,
                                                                data={'mpid': dat.entry_id})
                    
                entry = get_slab_entry(dat, relaxed=relaxed, 
                                       clean_slab_entry=slab_entries[dat.slab_rid],
                                       ads_entries=[self.mol_entry[dat.adsorbate]],
                                       data={'mpid': dat.entry_id, 'adsorbate': dat.adsorbate})
                entry.Nads_in_slab = dat.nads*sum(self.mol_entry[dat.adsorbate].composition.as_dict().values())
                slab_entries[dat.rid] = entry
                
        self.slab_entries = slab_entries
        return slab_entries
        
    def get_surfe_plotter(self, criteria=None, dat_list=None, relaxed=True):
        
        if dat_list:
            slab_entries_dict = self.get_slab_entries(criteria, dat_list=dat_list,
                                                      relaxed=relaxed)
        elif criteria == None:
            slab_entries_dict = self.slab_entries_dict
        else:
            slab_entries_dict = self.get_slab_entries(criteria, relaxed=relaxed)
        
        entry_id_to_surfplt_dict = {}
        for rid in slab_entries_dict.keys():
            slabentry = slab_entries_dict[rid]
            if slabentry.data['mpid'] not in entry_id_to_surfplt_dict.keys():
                entry_id_to_surfplt_dict[slabentry.data['mpid']] = []
            entry_id_to_surfplt_dict[slabentry.data['mpid']].append(slabentry)
            
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

            if entry_id in bulk_oxides_dict.keys():
                bulk_entry = ComputedStructureEntry.from_dict(bulk_oxides_dict[entry_id])
            else:
                bulk_entry = self.mprester.get_entry_by_material_id(entry_id, inc_structure=True, 
                                                                    conventional_unit_cell=True)[0]
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
        return -1*U + -pH*kB*T * JtoeV * np.log(10)
    
    def get_G0_OER(self):
        # * + 2H2O(l) --> * + 2H2O(l)
        return 0
    
    def get_G1_OER(self, adsentry, T=0, U=0, pH=0):
        # * + 2H2O(l) --> OH* + e- + H+ + H2O(l)
        
        EH2O = self.ads_in_a_box['H2O']
        EH2 = self.ads_in_a_box['H2']
        etransfer = self.get_e_transfer_corr(T=T, U=U, pH=pH)
        
        EOHstar = adsentry.energy
        Estar = adsentry.clean_entry.energy
        E0step = Estar + 2*EH2O
        E1step = EOHstar + ((1/2)*EH2+etransfer) + EH2O                    
        return E1step - E0step + self.Gcorr['OH']
    
    def get_G2_OER(self, adsentry, T=0, U=0, pH=0):
        # * + 2H2O(l) --> O* + 2e- + 2H+ + H2O(l)
        
        EH2O = self.ads_in_a_box['H2O']
        EH2 = self.ads_in_a_box['H2']
        etransfer = self.get_e_transfer_corr(T=T, U=U, pH=pH)
        
        
        EOstar = adsentry.energy
        Estar = adsentry.clean_entry.energy
        E0step = Estar + 2*EH2O
        E2step = EOstar + 2*((1/2)*EH2+etransfer) + EH2O
        return E2step - E0step + self.Gcorr['O']
    
    def get_G3_OER(self, adsentry, T=0, U=0, pH=0):
        # * + 2H2O(l) --> OOH* 3e- + 3H+
        
        EH2O = self.ads_in_a_box['H2O']
        EH2 = self.ads_in_a_box['H2']
        etransfer = self.get_e_transfer_corr(T=T, U=U, pH=pH)

        EOOHstar = adsentry.energy
        Estar = adsentry.clean_entry.energy
        E0step = Estar + 2*EH2O
        E3step = EOOHstar + 3*((1/2)*EH2+etransfer)
        return E3step - E0step + self.Gcorr['OOH']

    def get_G4_OER(self, T=0, U=0, pH=0):
        # * + 2H2O(l) --> * + O2(g) + 2H2(g)
        
        etransfer = self.get_e_transfer_corr(T=T, U=U, pH=pH)        
        GH2O = -12.457
        GH2 = -7.194
        GO2 = 1.23*4 + 2*GH2O - GH2*2
        return 4*((1/2)*GH2+etransfer) + GO2 - 2*GH2O
        
    def get_gibbs_adsorption_energies(self, adsorbate, criteria=None, surfplt_dict=None, T=298.15, U=0, pH=0):
                    
        muH2O = -14.231 # DFT energy per formula of H2O from MP 
        muH2 = -6.7714828 # DFT energy per formula of H2 from MP 
        EO = -4.946243415 # DFT energy per atom of O2 from MP
        etransfer = self.get_e_transfer_corr(T=T, U=U, pH=pH)
        dmuO = muH2O - muH2 - 2*etransfer - EO + self.Gcorr['O']

        if surfplt_dict:
            surfplt_dict = surfplt_dict
        elif criteria:
            surfplt_dict = self.get_surfe_plotter(criteria, relaxed=False)
        elif criteria == None and surfplt_dict==None:
            surfplt_dict = self.surf_plt
        
        Gads_dict = {}
        for mpid in surfplt_dict.keys():
            surfplt = surfplt_dict[mpid]
            Gads_dict[mpid] = {}
            for hkl in surfplt.all_slab_entries.keys():
                
                entry = surfplt.get_stable_entry_at_u(hkl, delu_dict={Symbol('delu_O'): dmuO}, no_doped=True)[0]
                slab_rid = entry.entry_id
                
                Eads = []
                for adsentry in self.slab_entries.values():
                    if 'adslab-' in adsentry.entry_id and adsentry.clean_entry.entry_id == slab_rid:
                        if adsentry.data['adsorbate'] == adsorbate:
                            if adsorbate == 'OH':
                                Eads.append(self.get_G1_OER(adsentry, T=T, U=T, pH=T))
                            elif adsorbate == 'O':
                                Eads.append(self.get_G2_OER(adsentry, T=T, U=T, pH=T))
                            elif adsorbate == 'OOH':
                                Eads.append(self.get_G3_OER(adsentry, T=T, U=T, pH=T))
                if not Eads:
                    continue
                Gads_dict[mpid][hkl] = sorted(Eads)[0] + self.Gcorr[adsorbate]
                
        return Gads_dict
    
    def get_rxn_energies(self, criteria=None, dat_list=None, T=0, U=0, pH=0):
        """
        Returns a dictionary containing the individual energies at each step 
            of the reaction diagram (G1, G2, g3, G4, G5), the reaction energies
            (DG1, DG2, DG3, DG4), and the overpotentials. Each dict is organized 
            in a nested dictionary of mpid to hkl.
        """
        
        etransfer = self.get_e_transfer_corr(T=T, U=U, pH=pH)
        G1 = 0
        dat_list = dat_list if dat_list else self.find_data_objects(criteria)
        surfe_dict = self.get_surfe_plotter(dat_list=dat_list)
        G2_dict = self.get_gibbs_adsorption_energies('OH', surfplt_dict=surfe_dict, T=T, U=U, pH=pH)
        G3_dict = self.get_gibbs_adsorption_energies('O', surfplt_dict=surfe_dict, T=T, U=U, pH=pH)
        G4_dict = self.get_gibbs_adsorption_energies('OOH', surfplt_dict=surfe_dict, T=T, U=U, pH=pH)
        G5 = 4*1.23 + 4*etransfer
        for i, d in enumerate([G2_dict, G3_dict, G4_dict]):
            for mpid in d.keys():
                for hkl in d[mpid].keys():
                    d[mpid][hkl] += (i+1)*etransfer

        rxn_energy_dict = {'G1': G1, 'G2_dict': G2_dict, 'G3_dict': G3_dict, 'G4_dict': G4_dict, 'G5': G5} 
        
        rxn_energy_dict.update({'DG1': {}, 'DG2': {}, 'DG3': {}, 'DG4': {}, 'overpotential': {}})
        for mpid in rxn_energy_dict['G2_dict'].keys():
            rxn_energy_dict['DG1'][mpid] = {}
            rxn_energy_dict['DG2'][mpid] = {}
            rxn_energy_dict['DG3'][mpid] = {}
            rxn_energy_dict['DG4'][mpid] = {}
            rxn_energy_dict['overpotential'][mpid] = {}
            for hkl in rxn_energy_dict['G2_dict'][mpid].keys():
                if hkl not in G2_dict[mpid].keys() or \
                hkl not in G3_dict[mpid].keys() or \
                hkl not in G4_dict[mpid].keys():
                    continue
                DG1 = G2_dict[mpid][hkl] - G1
                DG2 = G3_dict[mpid][hkl] - G2_dict[mpid][hkl]
                DG3 = G4_dict[mpid][hkl] - G3_dict[mpid][hkl]
                DG4 = G5 - G4_dict[mpid][hkl]
                rxn_energy_dict['DG1'][mpid][hkl] = DG1
                rxn_energy_dict['DG2'][mpid][hkl] = DG2
                rxn_energy_dict['DG3'][mpid][hkl] = DG3
                rxn_energy_dict['DG4'][mpid][hkl] = DG4
                rxn_energy_dict['overpotential'][mpid][hkl] = max(DG1, DG2, DG3, DG4) - G5/4
        
        return rxn_energy_dict
        
    def build_rxn_diagram(self, criteria=None, T=0, U=0, pH=0):
        
        from matplotlib import pylab as rxn_plt
        
        rxn_energy_dict = self.get_rxn_energies(criteria=criteria, T=T, U=U, pH=pH)
        G1, G2_dict, G3_dict, G4_dict, G5, DG1, DG2, DG3, DG4, overpotential = rxn_energy_dict.values()
        
        rxn_diagram_dict = {}
        for mpid in G2_dict.keys():
            rxn_diagram_dict[mpid] = {}
            for hkl in G2_dict[mpid].keys():
                G2 = G2_dict[mpid][hkl]
                G3 = G3_dict[mpid][hkl]
                G4 = G4_dict[mpid][hkl]
                
                plots = []
                plots.append(rxn_plt.plot([0, 1], [G1, G1], 'r-'))
                plots.append(rxn_plt.plot([1, 1], [G1, G2], 'r-'))
                plots.append(rxn_plt.plot([1, 2], [G2, G2], 'r-'))
                plots.append(rxn_plt.plot([2, 2], [G2, G3], 'r-'))
                plots.append(rxn_plt.plot([2, 3], [G3, G3], 'r-'))
                plots.append(rxn_plt.plot([3, 3], [G3, G4], 'r-'))
                plots.append(rxn_plt.plot([3, 4], [G4, G4], 'r-'))
                plots.append(rxn_plt.plot([4, 4], [G4, G5], 'r-'))
                plots.append(rxn_plt.plot([4, 5], [G5, G5], 'r-'))                
                rxn_diagram_dict[mpid][hkl] = plots
                                
        return rxn_diagram_dict
    
    def ideal_rxn_diagram(self, T=0, U=0, pH=0):
        
        from matplotlib import pylab as ideal_plt
        
        plots = []
        etransfer = self.get_e_transfer_corr(T=T, U=U, pH=pH)
        G0 = 0
        G4 = 4*1.23
        plots.append(ideal_plt.plot([0, 1], [G0, G0], 'k--'))
        plots.append(ideal_plt.plot([1, 1], [G0, G4/4 + etransfer], 'k--'))
        plots.append(ideal_plt.plot([1, 2], [G4/4+etransfer, G4/4+etransfer], 'k--'))
        plots.append(ideal_plt.plot([2, 2], [G4/4+etransfer, G4/2+2*etransfer], 'k--'))
        plots.append(ideal_plt.plot([2, 3], [G4/2+2*etransfer, G4/2+2*etransfer], 'k--'))
        plots.append(ideal_plt.plot([3, 3], [G4/2+2*etransfer, G4*(3/4)+3*etransfer], 'k--'))
        plots.append(ideal_plt.plot([3, 4], [G4*(3/4)+3*etransfer, G4*(3/4)+3*etransfer], 'k--'))
        plots.append(ideal_plt.plot([4, 4], [G4*(3/4)+3*etransfer, G4+4*etransfer], 'k--'))
        plots.append(ideal_plt.plot([4, 5], [G4+4*etransfer, G4+4*etransfer], 'k--', label='Ideal'))
        return plots
    
    def get_equation(self, slabentry, adsorbate=None, surface_energy=False):

        if surface_energy:
            muH2O = -14.231 # DFT energy per formula of H2O from MP 
            muH2 = -6.7714828 # DFT energy per formula of H2 from MP 
            EO = -4.946243415 # DFT energy per atom of O2 from MP
            muO = muH2O - muH2 - 2*self.get_e_transfer_corr(Symbol('T'), U=Symbol('U'), pH=Symbol('pH'))

            mpid = slabentry.data['mpid']
            bulk_entry = ComputedStructureEntry.from_dict(bulk_oxides_dict[mpid])
            ref_entries = get_ref_entries(bulk_entry)
            preset_slabentry_se(slabentry, bulk_entry, ref_entries=ref_entries)
            e = slabentry.preset_surface_energy.subs({'delu_O': muO-EO + self.Gcorr['O']})

        else: # calculate Gibbs adsorption energy instead
            if adsorbate == 'OH':
                e = self.get_G1_OER(slabentry, Symbol('T'), U=Symbol('U'), pH=Symbol('pH'))
            elif adsorbate == 'O':
                e = self.get_G2_OER(slabentry, Symbol('T'), U=Symbol('U'), pH=Symbol('pH'))
            elif adsorbate == 'OOH':
                e = self.get_G3_OER(slabentry, Symbol('T'), U=Symbol('U'), pH=Symbol('pH'))
                
        d = e.as_coefficients_dict()
        d = {str(k): d[k] for k in d.keys()}

        return d
