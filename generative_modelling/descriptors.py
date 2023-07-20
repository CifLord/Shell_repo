import json, copy, os
import numpy as np
from database.queryengine import SurfaceQueryEngine

from pymatgen.entries.computed_entries import ComputedStructureEntry, Composition
from pymatgen.core.periodic_table import Element

from torch_geometric.data.data import Data
from scipy.spatial import ConvexHull, QhullError

from generative_modelling import pca_analysis as q
bulk_site_dos_info = json.load(open(os.path.join(q.__file__.replace(q.__file__.split('/')[-1], ''), 'bulk_site_dos_info.json'), 'r'))
cohesive_energy_calcs_dict = json.load(open(os.path.join(q.__file__.replace(q.__file__.split('/')[-1], ''), 'cohesive_energy_calcs_dict.json'), 'r'))
ecoh_dict = {k: cohesive_energy_calcs_dict[k]['energy'] for k in cohesive_energy_calcs_dict.keys()}
bulk_oxides_20220621 = json.load(open(os.path.join(q.__file__.replace(q.__file__.split('/')[-1], '').replace(q.__file__.split('/')[-2]+'/', ''), 
                                                   'database', 'bulk_oxides_20220621.json'), 'rb'))
bulk_oxides_dict = {entry['entry_id']: entry for entry in bulk_oxides_20220621}
nagle_X = json.load(open(os.path.join(q.__file__.replace(q.__file__.split('/')[-1], ''),
                                      'nagle_X.json'), 'r'))



class BulkDescriber():
    def __init__(self, bulk_entry, blength=3):
        
        self.blength = blength       
        self.bulk_entry = bulk_entry
        self.mpid = self.bulk_entry.entry_id
        self.bulk = self.bulk_entry.structure.copy()
        self.site_electronic_info = bulk_site_dos_info[self.mpid]

        # describe all bulk environments
        self.bulk_poly_dict = get_polyhedron_dict(self.bulk)
        self.all_bulk_env = {i: nns for i, nns in enumerate(self.bulk.get_all_neighbors(3))}
        self.all_bulk_nnn_env = {i: self.bulk.get_neighbors_in_shell(site.coords, 3.1, 3) for i, site in enumerate(self.bulk)}
        
        # Describe node properties of bulk
        self.node_properties = {}
        self.node_properties['isolated_atom_energy'] = {i: ecoh_dict[site.species_string]
                                                        for i, site in enumerate(self.bulk)}
        self.node_properties['pauling_X'] = {i: Element(site.species_string).X
                                             for i, site in enumerate(self.bulk)}
        self.node_properties['nagle_X'] = {i: nagle_X[site.species_string]
                                           for i, site in enumerate(self.bulk)}
        self.node_properties['atomic_radii'] = {i: Element(site.species_string).atomic_radius
                                                for i, site in enumerate(self.bulk)}
        self.node_properties['ionic_radii'] = {i: Element(site.species_string).average_ionic_radius 
                                               for i, site in enumerate(self.bulk)}
        self.node_properties['valence'] = {i: sum(Element(site.species_string).valence)
                                           for i, site in enumerate(self.bulk)}
        self.node_properties['efermi'] = {i: self.site_electronic_info[str(i)]['efermi'] \
                                          for i, site in enumerate(self.bulk)}
        self.node_properties['cbm'] = {i: self.site_electronic_info[str(i)]['cbm'] \
                                       for i, site in enumerate(self.bulk)}
        self.node_properties['vbm'] = {i: self.site_electronic_info[str(i)]['vbm'] \
                                       for i, site in enumerate(self.bulk)}
        self.node_properties['bandgap'] = {i: self.site_electronic_info[str(i)]['bandgap'] \
                                           for i, site in enumerate(self.bulk)}
        self.node_properties['bandcenter'] = {i: self.site_electronic_info[str(i)]['bandcenter'] \
                                              for i, site in enumerate(self.bulk)}
        self.node_properties['bandwith'] = {i: self.site_electronic_info[str(i)]['bandwith'] \
                                            for i, site in enumerate(self.bulk)}
        self.node_properties['bandfilling'] = {i: self.site_electronic_info[str(i)]['bandfilling'] \
                                               for i, site in enumerate(self.bulk)}
        self.node_properties['cohesive_energy'] = {i: ecoh_dict[site.species_string] - self.bulk_entry.energy_per_atom \
                                                   for i, site in enumerate(self.bulk)}

        

        
class OxideSlabDescriber():
    
    def __init__(self, bulk_describer, slab_doc, blength=3):

        self.queryengine = SurfaceQueryEngine('ugIKy1XLaS3jKfH88O75N6liFLtuK2sh', collection='Shell05022023')
        self.bulk_d = bulk_describer
        self.slab_doc = slab_doc
        self.slab_rid = self.slab_doc['rid']
        self.slab_entry = get_slab_entry(self.slab_doc, None, self.queryengine)
        self.slab = self.slab_entry.structure.copy()
        
        # describe all bare slab environments and broken bonds
        self.slab_poly_dict = get_polyhedron_dict(self.slab)
        self.all_slab_env = {i: nns for i, nns in enumerate(self.slab.get_all_neighbors(3))}
        self.all_slab_nnn_env = {i: self.slab.get_neighbors_in_shell(site.coords, 3.1, 3) for i, site in enumerate(self.slab)}
        self.all_slab_bb = self.get_broken_bonds_env(self.slab, self.all_slab_env, self.bulk_d.all_bulk_env)
        self.all_slab_nnn_bb = self.get_broken_bonds_env(self.slab, self.all_slab_nnn_env, self.bulk_d.all_bulk_nnn_env)
        self.all_slab_poly_bb = self.get_broken_bonds_env(self.slab, self.slab_poly_dict, self.bulk_d.bulk_poly_dict)
        
        # Meta data
        self.all_meta_descriptors = {'mpid': self.bulk_d.mpid}
        self.all_meta_descriptors['bare_se'] = self.get_surface_energy()
        
        # Now get all descriptors 
        self.all_param_descriptors = {}
        self.all_param_descriptors['bare_se'] = self.all_meta_descriptors['bare_se']
        
        # Describe poduct of node and edge properties and nearest neighbor broken bond, sqrt bb, generalized bb, sqrt generalize bb 
        #######################################################################################################################

        # Describe all combinations of broken bonds and Enodes and Ebonds, dimer_break, and E=1 for surface
        self.all_param_descriptors['bare_bb_e1'] = self.get_broken_bonds_energy(self.all_slab_bb, self.bulk_d.all_bulk_env, self.slab, 
                                                                                ecoh=None, node_energy=False, dimer_diff=False)
        self.all_param_descriptors['bare_bb_e1_sqrt'] = self.get_broken_bonds_energy(self.all_slab_bb, self.bulk_d.all_bulk_env, self.slab, 
                                                                                ecoh=None, node_energy=False, dimer_diff=False, sqrted=True)
        self.all_param_descriptors['bare_bb_edimer'] = self.get_broken_bonds_energy(self.all_slab_bb, self.bulk_d.all_bulk_env, self.slab, 
                                                                                ecoh=None, node_energy=False, dimer_diff=True)
        self.all_param_descriptors['bare_bb_edimer_sqrt'] = self.get_broken_bonds_energy(self.all_slab_bb, self.bulk_d.all_bulk_env, self.slab, 
                                                                                ecoh=None, node_energy=False, dimer_diff=True, sqrted=True)
        # generalized BB
        self.all_param_descriptors['bare_gbb_e1'] = self.get_broken_bonds_energy(self.all_slab_bb, self.bulk_d.all_bulk_env, self.slab, slab_env_dict=self.all_slab_env,
                                                                                ecoh=None, node_energy=False, dimer_diff=False, generalize_bb=True)
        self.all_param_descriptors['bare_gbb_e1_sqrt'] = self.get_broken_bonds_energy(self.all_slab_bb, self.bulk_d.all_bulk_env, self.slab, slab_env_dict=self.all_slab_env,
                                                                                ecoh=None, node_energy=False, dimer_diff=False, sqrted=True, generalize_bb=True)
        self.all_param_descriptors['bare_gbb_edimer'] = self.get_broken_bonds_energy(self.all_slab_bb, self.bulk_d.all_bulk_env, self.slab, slab_env_dict=self.all_slab_env,
                                                                                ecoh=None, node_energy=False, dimer_diff=True, generalize_bb=True)
        self.all_param_descriptors['bare_gbb_edimer_sqrt'] = self.get_broken_bonds_energy(self.all_slab_bb, self.bulk_d.all_bulk_env, self.slab, slab_env_dict=self.all_slab_env,
                                                                                ecoh=None, node_energy=False, dimer_diff=True, sqrted=True, generalize_bb=True)
        for node_name in self.bulk_d.node_properties.keys():
            self.all_param_descriptors['bare_bb_node_%s' %(node_name)] = self.get_broken_bonds_energy(self.all_slab_bb, self.bulk_d.all_bulk_env, 
                                                                                                      self.slab, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                      node_energy=False, dimer_diff=False)
            self.all_param_descriptors['bare_bb_ebond_%s' %(node_name)] = self.get_broken_bonds_energy(self.all_slab_bb, self.bulk_d.all_bulk_env, 
                                                                                                       self.slab, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                       node_energy=True, dimer_diff=False)
            self.all_param_descriptors['bare_bb_node_sqrt_%s' %(node_name)] = self.get_broken_bonds_energy(self.all_slab_bb, self.bulk_d.all_bulk_env, 
                                                                                                      self.slab, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                      node_energy=False, dimer_diff=False, sqrted=True)
            self.all_param_descriptors['bare_bb_ebond_sqrt_%s' %(node_name)] = self.get_broken_bonds_energy(self.all_slab_bb, self.bulk_d.all_bulk_env, 
                                                                                                       self.slab, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                       node_energy=True, dimer_diff=False, sqrted=True)
            # generalized BB
            self.all_param_descriptors['bare_gbb_node_%s' %(node_name)] = self.get_broken_bonds_energy(self.all_slab_bb, self.bulk_d.all_bulk_env, 
                                                                                                      self.slab, slab_env_dict=self.all_slab_env,
                                                                                                       ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                      node_energy=False, dimer_diff=False, generalize_bb=True)
            self.all_param_descriptors['bare_gbb_ebond_%s' %(node_name)] = self.get_broken_bonds_energy(self.all_slab_bb, self.bulk_d.all_bulk_env, 
                                                                                                       self.slab, slab_env_dict=self.all_slab_env,
                                                                                                        ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                       node_energy=True, dimer_diff=False, generalize_bb=True)
            self.all_param_descriptors['bare_gbb_node_sqrt_%s' %(node_name)] = self.get_broken_bonds_energy(self.all_slab_bb, self.bulk_d.all_bulk_env, 
                                                                                                      self.slab, slab_env_dict=self.all_slab_env,
                                                                                                            ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                      node_energy=False, dimer_diff=False, sqrted=True, generalize_bb=True)
            self.all_param_descriptors['bare_gbb_ebond_sqrt_%s' %(node_name)] = self.get_broken_bonds_energy(self.all_slab_bb, self.bulk_d.all_bulk_env, 
                                                                                                       self.slab, slab_env_dict=self.all_slab_env,
                                                                                                             ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                       node_energy=True, dimer_diff=False, sqrted=True, generalize_bb=True)
        #######################################################################################################################

        # Repeat for next nearest neighbor 
        #######################################################################################################################

        # Describe all combinations of broken bonds and Enodes and Ebonds, dimer_break, and E=1 for surface
        self.all_param_descriptors['bare_bb_nnn_e1'] = self.get_broken_bonds_energy(self.all_slab_nnn_bb, self.bulk_d.all_bulk_nnn_env, self.slab, 
                                                                                ecoh=None, node_energy=False, dimer_diff=False)
        self.all_param_descriptors['bare_bb_nnn_e1_sqrt'] = self.get_broken_bonds_energy(self.all_slab_nnn_bb, self.bulk_d.all_bulk_nnn_env, self.slab, 
                                                                                ecoh=None, node_energy=False, dimer_diff=False, sqrted=True)
        self.all_param_descriptors['bare_bb_nnn_edimer'] = self.get_broken_bonds_energy(self.all_slab_nnn_bb, self.bulk_d.all_bulk_nnn_env, self.slab, 
                                                                                ecoh=None, node_energy=False, dimer_diff=True)
        self.all_param_descriptors['bare_bb_nnn_edimer_sqrt'] = self.get_broken_bonds_energy(self.all_slab_nnn_bb, self.bulk_d.all_bulk_nnn_env, self.slab, 
                                                                                ecoh=None, node_energy=False, dimer_diff=True, sqrted=True)
        # generalized BB
        self.all_param_descriptors['bare_gbb_nnn_e1'] = self.get_broken_bonds_energy(self.all_slab_nnn_bb, self.bulk_d.all_bulk_nnn_env, self.slab, 
                                                                                     slab_env_dict=self.all_slab_nnn_env,
                                                                                ecoh=None, node_energy=False, dimer_diff=False, generalize_bb=True)
        self.all_param_descriptors['bare_gbb_nnn_e1_sqrt'] = self.get_broken_bonds_energy(self.all_slab_nnn_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                          self.slab, slab_env_dict=self.all_slab_nnn_env,
                                                                                ecoh=None, node_energy=False, dimer_diff=False, sqrted=True, generalize_bb=True)
        self.all_param_descriptors['bare_gbb_nnn_edimer'] = self.get_broken_bonds_energy(self.all_slab_nnn_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                         self.slab, slab_env_dict=self.all_slab_nnn_env,
                                                                                ecoh=None, node_energy=False, dimer_diff=True, generalize_bb=True)
        self.all_param_descriptors['bare_gbb_nnn_edimer_sqrt'] = self.get_broken_bonds_energy(self.all_slab_nnn_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                              self.slab, slab_env_dict=self.all_slab_nnn_env,
                                                                                ecoh=None, node_energy=False, dimer_diff=True, sqrted=True, generalize_bb=True)
        for node_name in self.bulk_d.node_properties.keys():
            self.all_param_descriptors['bare_bb_nnn_node_%s' %(node_name)] = self.get_broken_bonds_energy(self.all_slab_nnn_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                                      self.slab, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                      node_energy=False, dimer_diff=False)
            self.all_param_descriptors['bare_bb_nnn_ebond_%s' %(node_name)] = self.get_broken_bonds_energy(self.all_slab_nnn_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                                       self.slab, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                       node_energy=True, dimer_diff=False)
            self.all_param_descriptors['bare_bb_nnn_node_sqrt_%s' %(node_name)] = self.get_broken_bonds_energy(self.all_slab_nnn_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                                      self.slab, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                      node_energy=False, dimer_diff=False, sqrted=True)
            self.all_param_descriptors['bare_bb_nnn_ebond_sqrt_%s' %(node_name)] = self.get_broken_bonds_energy(self.all_slab_nnn_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                                       self.slab, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                       node_energy=True, dimer_diff=False, sqrted=True)
            # generalized BB
            self.all_param_descriptors['bare_gbb_nnn_node_%s' %(node_name)] = self.get_broken_bonds_energy(self.all_slab_nnn_bb, self.bulk_d.all_bulk_nnn_env, self.slab, 
                                                                                                           slab_env_dict=self.all_slab_nnn_env, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                      node_energy=False, dimer_diff=False, generalize_bb=True)
            self.all_param_descriptors['bare_gbb_nnn_ebond_%s' %(node_name)] = self.get_broken_bonds_energy(self.all_slab_nnn_bb, self.bulk_d.all_bulk_nnn_env, self.slab, 
                                                                                                            slab_env_dict=self.all_slab_nnn_env,
                                                                                                            ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                       node_energy=True, dimer_diff=False, generalize_bb=True)
            self.all_param_descriptors['bare_gbb_nnn_node_sqrt_%s' %(node_name)] = self.get_broken_bonds_energy(self.all_slab_nnn_bb, self.bulk_d.all_bulk_nnn_env, self.slab, 
                                                                                                                slab_env_dict=self.all_slab_nnn_env,
                                                                                                                ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                      node_energy=False, dimer_diff=False, sqrted=True, generalize_bb=True)
            self.all_param_descriptors['bare_gbb_nnn_ebond_sqrt_%s' %(node_name)] = self.get_broken_bonds_energy(self.all_slab_nnn_bb, self.bulk_d.all_bulk_nnn_env, self.slab, 
                                                                                                                 slab_env_dict=self.all_slab_nnn_env,
                                                                                                       ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                       node_energy=True, dimer_diff=False, sqrted=True, generalize_bb=True)
        #######################################################################################################################
            
        # Misc descriptors
        self.all_param_descriptors['get_tot_polyhedron_vol_diff'] = self.get_tot_polyhedron_vol_diff(norm_bulk_cn=False)
        self.all_param_descriptors['get_tot_polyhedron_vol_diff_norm_bulk'] = self.get_tot_polyhedron_vol_diff(norm_bulk_cn=True)
        self.all_param_descriptors['get_tot_polyhedron_vol_diff_sqrt'] = self.get_tot_polyhedron_vol_diff(norm_bulk_cn=False, sqrted=True)
        self.all_param_descriptors['get_tot_polyhedron_vol_diff_norm_bulk_sqrt'] = self.get_tot_polyhedron_vol_diff(norm_bulk_cn=True, sqrted=True)
        
    def get_broken_bonds_env(self, slab, slab_env_dict, bulk_env_dict):
        
        # Can be used for all_slab_env, slab_poly_env, slab_nnn_env. Returns
        # a list of bulk sites for each slab site with broken bonds.
        # To describe adslab_envs, you must first describe slab_bbs.

        bb_env = {}
        for slab_i in slab_env_dict.keys():
            # Get all bb environments to do generalized CN and BB
            bulk_i = slab[slab_i].bulk_equivalent
            if len(bulk_env_dict[bulk_i]) == len(slab_env_dict[slab_i]):
                continue

            bulk_sites = copy.deepcopy(bulk_env_dict[bulk_i])
            bulk_props = [(site.index, site.species_string) for site in bulk_sites]
            for site in slab_env_dict[slab_i]:
                found = False
                for bi, b in enumerate(bulk_props):
                    if site.index in b:
                        del bulk_props[bi]
                        del bulk_sites[bi]
                        found = True
                        break
                if not found:
                    for bi, b in enumerate(bulk_props):
                        if site.species_string in b:
                            del bulk_props[bi]
                            del bulk_sites[bi]
                            break
            bb_env[slab_i] = bulk_sites

        return bb_env

    def get_tot_polyhedron_vol_diff(self, norm_bulk_cn=False, sqrted=False):
        tot_poly_volume_diff = 0
        for slab_i in self.slab_poly_dict.keys():
            # Get all bb env for all slab sites so we can account for generalized BB
            
            bulk_i = self.slab[slab_i].bulk_equivalent
            coords = [site.coords for site in self.slab_poly_dict[slab_i]]
            coords.append(self.slab[slab_i].coords)
            try:
                slab_poly_vol = ConvexHull(coords).volume if len(coords) > 3 else 0
            except QhullError:
                slab_poly_vol = 0
            coords = [site.coords for site in self.bulk_d.bulk_poly_dict[bulk_i]]
            coords.append(self.bulk_d.bulk[bulk_i].coords)

            try:
                bulk_poly_vol = ConvexHull(coords).volume if len(coords) > 3 else 0
            except QhullError:
                bulk_poly_vol = 0

            ecoh = ecoh_dict['%s-O' %(self.slab[slab_i].species_string)] if ecoh_dict else 1

            if norm_bulk_cn:
                if bulk_poly_vol == 0:
                    tot_poly_volume_diff+=0
                else:
                    if sqrted:
                        tot_poly_volume_diff+=(((bulk_poly_vol - slab_poly_vol+0.01)/bulk_poly_vol)**(1/2))*ecoh
                    else:
                        tot_poly_volume_diff+=((bulk_poly_vol - slab_poly_vol)/bulk_poly_vol)*ecoh
            else:
                if sqrted:
                    tot_poly_volume_diff+=((bulk_poly_vol - slab_poly_vol+0.01)**(1/2))*ecoh
                else:
                    tot_poly_volume_diff+=(bulk_poly_vol - slab_poly_vol)*ecoh
        return tot_poly_volume_diff
        
    def get_broken_bonds_energy(self, bb_dict, bulk_env_dict, slab, slab_env_dict=None, ecoh=None, 
                                node_energy=False, dimer_diff=False, generalize_bb=False, sqrted=False):
        # Gets broken bond energy summation (essentially surface energy) with the 
        # cohesive energy (ecoh) set to 1 if None, the node component of the  
        # undercoordinated atom if node_energy=True, or the bond energy if node_energy=False

        model_surfe = 0 
        for slab_i in bb_dict.keys():
            if slab[slab_i].frac_coords[2] < 0.5:
                continue

            bulk_i = slab[slab_i].bulk_equivalent
            bulk_cn = len(bulk_env_dict[bulk_i])
            local_env_e = []
            for bulk_site in bb_dict[slab_i]:
                if not ecoh:
                    e = 1
                else:
                    if node_energy:
                        e = ecoh[bulk_i]
                    else:
                        # Get energy of bond instead (represented by difference in energy of atom 1 and 2)
                        e = ecoh[bulk_i] - ecoh[bulk_site.index]

                # Get the dimer breaking energy instead
                if dimer_diff:
                    s1 = self.bulk_d.bulk[bulk_i].species_string
                    s2 = bulk_site.species_string
                    if s1 == 'O' and s2 == 'O':
                        e = ecoh_dict['O-O'] - 2*ecoh_dict['O']
                    else:
                        m = s1 if s1 != 'O' else s2
                        e = ecoh_dict['%s-O' %(m)] - (ecoh_dict['O']+ecoh_dict[m])
                local_env_e.append(e)
                
            if generalize_bb and slab_env_dict:
                generalized_cn = sum([len(slab_env_dict[site.index])/len(bulk_env_dict[site.bulk_equivalent]) for site in slab_env_dict[slab_i]])
                local_env_bb = (len(bulk_env_dict[bulk_i])+0.01)-generalized_cn
            else:
                local_env_bb = len(bb_dict[slab_i])
                
            if sqrted:
                model_surfe+=np.mean(local_env_e)*(local_env_bb/bulk_cn)**(1/2)
            else:
                model_surfe+=np.mean(local_env_e)*(local_env_bb/bulk_cn)

        return model_surfe/slab.surface_area
        
     # Get the target data
    def get_surface_energy(self):
        # Get the surface energy at T, pH, U = 0, 0, 0
        se = self.queryengine.get_equation(self.slab_entry, surface_energy=True)
        if type(se).__name__ != 'float':
            return se['1']
        else:
            return float(se)

        
class OxideAdSlabDescriber:
    
    def __init__(self, bulk_describer, slab_describer, adslab_doc, blength=3):
        
        self.blength = blength
        self.bulk_d = bulk_describer
        self.slab_d = slab_describer
        self.adslab_doc = adslab_doc
        self.slab_rid = self.slab_d.slab_doc['rid']
        self.adslab_entry = get_slab_entry(self.adslab_doc, self.slab_d.slab_doc, self.slab_d.queryengine)
        self.adslab = self.adslab_entry.structure.copy()
        self.ads = self.adslab_doc['adsorbate']
        
        # now describe all adslab environments and broken bonds
        self.Oads, self.ads_metal, self.adsorbate_env = self.get_adsorbate_coordination_env()
        self.ads_poly_dict = self.get_adsorbate_polyhedron_dict()
        self.all_adslab_env = self.get_adslab_coordination_env()
        self.all_adslab_nnn_env = {i: self.adslab.get_neighbors_in_shell(site.coords, self.blength+0.1, self.blength)\
                                  for i, site in enumerate(self.adslab) if i == self.Oads.index or site.surface_properties!='adsorbate'}
        self.all_adslab_env_bb = self.slab_d.get_broken_bonds_env(self.adslab, self.all_adslab_env, self.bulk_d.all_bulk_env)
        self.adsorbate_env_bb = self.slab_d.get_broken_bonds_env(self.adslab, {self.Oads.index: self.adsorbate_env}, self.bulk_d.all_bulk_env)
        self.all_adslab_nnn_env_bb = self.slab_d.get_broken_bonds_env(self.adslab, self.all_adslab_nnn_env, self.bulk_d.all_bulk_nnn_env)
        self.ads_poly_bb = self.slab_d.get_broken_bonds_env(self.adslab, self.ads_poly_dict, self.bulk_d.bulk_poly_dict)
                
        # Meta data
        self.all_meta_descriptors = {'adsorbate': self.ads}
        self.all_meta_descriptors['Eads'] = self.get_ads_energy()
        
        # Now get all descriptors 
        self.all_param_descriptors = copy.deepcopy(self.slab_d.all_param_descriptors)

                
        # Describe the nodes between the metal site and the adsorbate
        for node_name in self.bulk_d.node_properties.keys():
            node_props = self.bulk_d.node_properties[node_name]
            ads_comp = Composition(self.ads).as_dict()
            self.all_param_descriptors['M-ads_node_diff_%s' %(node_name)] = \
            node_props[self.ads_metal.bulk_equivalent] - node_props[self.Oads.bulk_equivalent]
        self.all_param_descriptors['M-ads_dimer'] = ecoh_dict['%s-O' %(self.ads_metal.species_string)]
        self.all_param_descriptors['M-ads_diff_dimer'] = \
        ecoh_dict['%s-O' %(self.ads_metal.species_string)] - (ecoh_dict[self.ads_metal.species_string]+ecoh_dict['O'])
        
        # Describe average difference of the nodes and adsorbate energy
        for node_name in self.bulk_d.node_properties.keys():
            self.all_param_descriptors['ave_node_diff_%s' %(node_name)] = \
            self.ave_node_diff(self.bulk_d.node_properties[node_name])
        
        # Describe poduct of node and edge properties and nearest neighbor broken bond
        ######################################################################################################################################################
                        
        # Describe all combinations of broken bonds and Enodes and Ebonds and E=1 for adslab 
        self.all_param_descriptors['adslab_bb_e1'] = self.slab_d.get_broken_bonds_energy(self.all_adslab_env_bb, self.bulk_d.all_bulk_env, 
                                                                                self.adslab, ecoh=None, node_energy=False, dimer_diff=False)
        self.all_param_descriptors['adslab_bb_e1_sqrt'] = self.slab_d.get_broken_bonds_energy(self.all_adslab_env_bb, self.bulk_d.all_bulk_env, 
                                                                                       self.adslab, ecoh=None, node_energy=False, dimer_diff=False, sqrted=True)
        self.all_param_descriptors['adslab_bb_edimer'] = self.slab_d.get_broken_bonds_energy(self.all_adslab_env_bb, self.bulk_d.all_bulk_env, 
                                                                                      self.adslab, ecoh=None, node_energy=False, dimer_diff=True)
        self.all_param_descriptors['adslab_bb_edimer_sqrt'] = self.slab_d.get_broken_bonds_energy(self.all_adslab_env_bb, self.bulk_d.all_bulk_env, 
                                                                                                  self.adslab, ecoh=None, node_energy=False, dimer_diff=True, sqrted=True)
        
        # Generalized CN
        self.all_param_descriptors['adslab_gbb_e1'] = self.slab_d.get_broken_bonds_energy(self.all_adslab_env_bb, self.bulk_d.all_bulk_env, 
                                                                                          self.adslab, slab_env_dict=self.all_adslab_env, 
                                                                                          ecoh=None, node_energy=False, dimer_diff=False, generalize_bb=True)
        self.all_param_descriptors['adslab_gbb_e1_sqrt'] = self.slab_d.get_broken_bonds_energy(self.all_adslab_env_bb, self.bulk_d.all_bulk_env, 
                                                                                       self.adslab, slab_env_dict=self.all_adslab_env, 
                                                                                               ecoh=None, node_energy=False, dimer_diff=False,
                                                                                               sqrted=True, generalize_bb=True)
        self.all_param_descriptors['adslab_gbb_edimer'] = self.slab_d.get_broken_bonds_energy(self.all_adslab_env_bb, self.bulk_d.all_bulk_env, 
                                                                                      self.adslab, slab_env_dict=self.all_adslab_env, 
                                                                                              ecoh=None, node_energy=False, dimer_diff=True, generalize_bb=True)
        self.all_param_descriptors['adslab_gbb_edimer_sqrt'] = self.slab_d.get_broken_bonds_energy(self.all_adslab_env_bb, self.bulk_d.all_bulk_env, 
                                                                                                  self.adslab, slab_env_dict=self.all_adslab_env, 
                                                                                                   ecoh=None, node_energy=False, 
                                                                                                   dimer_diff=True, sqrted=True, generalize_bb=True)
        
        for node_name in self.bulk_d.node_properties.keys():
            self.all_param_descriptors['adslab_bb_node_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.all_adslab_env_bb, self.bulk_d.all_bulk_env, 
                                                                                                      self.adslab, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                      node_energy=False, dimer_diff=False)
            self.all_param_descriptors['adslab_bb_ebond_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.all_adslab_env_bb, self.bulk_d.all_bulk_env, 
                                                                                                       self.adslab, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                       node_energy=True, dimer_diff=False)
            self.all_param_descriptors['adslab_bb_node_sqrt_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.all_adslab_env_bb, self.bulk_d.all_bulk_env, 
                                                                                                      self.adslab, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                      node_energy=False, dimer_diff=False, sqrted=True)
            self.all_param_descriptors['adslab_bb_ebond_sqrt_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.all_adslab_env_bb, self.bulk_d.all_bulk_env, 
                                                                                                       self.adslab, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                       node_energy=True, dimer_diff=False, sqrted=True)

            # Generalized CN
            self.all_param_descriptors['adslab_gbb_node_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.all_adslab_env_bb, self.bulk_d.all_bulk_env, 
                                                                                                      self.adslab, slab_env_dict=self.all_adslab_env, 
                                                                                                                ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                      node_energy=False, dimer_diff=False, generalize_bb=True)
            self.all_param_descriptors['adslab_gbb_ebond_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.all_adslab_env_bb, self.bulk_d.all_bulk_env, 
                                                                                                       self.adslab, slab_env_dict=self.all_adslab_env, 
                                                                                                                 ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                       node_energy=True, dimer_diff=False, generalize_bb=True)
            self.all_param_descriptors['adslab_gbb_node_sqrt_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.all_adslab_env_bb, self.bulk_d.all_bulk_env, 
                                                                                                      self.adslab, slab_env_dict=self.all_adslab_env, 
                                                                                                                     ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                      node_energy=False, dimer_diff=False, sqrted=True, generalize_bb=True)
            self.all_param_descriptors['adslab_gbb_ebond_sqrt_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.all_adslab_env_bb, self.bulk_d.all_bulk_env, 
                                                                                                       self.adslab, slab_env_dict=self.all_adslab_env, 
                                                                                                                      ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                       node_energy=True, dimer_diff=False, sqrted=True, generalize_bb=True)
        
        # Describe all combinations of broken bonds and Enodes and Ebonds and E=1 for local adsorbate env
        self.all_param_descriptors['ads_local_bb_e1'] = self.slab_d.get_broken_bonds_energy(self.adsorbate_env_bb, self.bulk_d.all_bulk_env, 
                                                                                self.adslab, ecoh=None,
                                                                                node_energy=False, dimer_diff=False)
        self.all_param_descriptors['ads_local_bb_e1_sqrt'] = self.slab_d.get_broken_bonds_energy(self.adsorbate_env_bb, self.bulk_d.all_bulk_env, 
                                                                                self.adslab, ecoh=None,
                                                                                node_energy=False, dimer_diff=False, sqrted=True)
        self.all_param_descriptors['ads_local_bb_edimer'] = self.slab_d.get_broken_bonds_energy(self.adsorbate_env_bb, self.bulk_d.all_bulk_env, 
                                                                                      self.adslab, ecoh=None,
                                                                                      node_energy=False, dimer_diff=True)
        self.all_param_descriptors['ads_local_bb_edimer_sqrt'] = self.slab_d.get_broken_bonds_energy(self.adsorbate_env_bb, self.bulk_d.all_bulk_env, 
                                                                                      self.adslab, ecoh=None,
                                                                                      node_energy=False, dimer_diff=True, sqrted=True)

        # Generalize BB
        self.all_param_descriptors['ads_local_gbb_e1'] = self.slab_d.get_broken_bonds_energy(self.adsorbate_env_bb, self.bulk_d.all_bulk_env, 
                                                                                self.adslab, slab_env_dict=self.all_adslab_env, ecoh=None,
                                                                                node_energy=False, dimer_diff=False, generalize_bb=True)
        self.all_param_descriptors['ads_local_gbb_e1_sqrt'] = self.slab_d.get_broken_bonds_energy(self.adsorbate_env_bb, self.bulk_d.all_bulk_env, 
                                                                                self.adslab, slab_env_dict=self.all_adslab_env, ecoh=None,
                                                                                node_energy=False, dimer_diff=False, sqrted=True, generalize_bb=True)
        self.all_param_descriptors['ads_local_gbb_edimer'] = self.slab_d.get_broken_bonds_energy(self.adsorbate_env_bb, self.bulk_d.all_bulk_env, 
                                                                                      self.adslab, slab_env_dict=self.all_adslab_env, ecoh=None,
                                                                                      node_energy=False, dimer_diff=True, generalize_bb=True)
        self.all_param_descriptors['ads_local_gbb_edimer_sqrt'] = self.slab_d.get_broken_bonds_energy(self.adsorbate_env_bb, self.bulk_d.all_bulk_env, 
                                                                                      self.adslab, slab_env_dict=self.all_adslab_env, ecoh=None,
                                                                                      node_energy=False, dimer_diff=True, sqrted=True, generalize_bb=True)

        for node_name in self.bulk_d.node_properties.keys():
            self.all_param_descriptors['ads_local_bb_node_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.adsorbate_env_bb, self.bulk_d.all_bulk_env, 
                                                                                                      self.adslab, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                      node_energy=False, dimer_diff=False)
            self.all_param_descriptors['ads_local_bb_ebond_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.adsorbate_env_bb, self.bulk_d.all_bulk_env, 
                                                                                                       self.adslab, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                       node_energy=True, dimer_diff=False)
            self.all_param_descriptors['ads_local_bb_node_sqrt_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.adsorbate_env_bb, self.bulk_d.all_bulk_env, 
                                                                                                      self.adslab, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                      node_energy=False, dimer_diff=False, sqrted=True)
            self.all_param_descriptors['ads_local_bb_ebond_sqrt_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.adsorbate_env_bb, self.bulk_d.all_bulk_env, 
                                                                                                       self.adslab, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                       node_energy=True, dimer_diff=False, sqrted=True)
            
            # Generalize BB
            self.all_param_descriptors['ads_local_gbb_node_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.adsorbate_env_bb, self.bulk_d.all_bulk_env, 
                                                                                                      self.adslab, slab_env_dict=self.all_adslab_env, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                      node_energy=False, dimer_diff=False, generalize_bb=True)
            self.all_param_descriptors['ads_local_gbb_ebond_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.adsorbate_env_bb, self.bulk_d.all_bulk_env, 
                                                                                                       self.adslab, slab_env_dict=self.all_adslab_env, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                       node_energy=True, dimer_diff=False, generalize_bb=True)
            self.all_param_descriptors['ads_local_gbb_node_sqrt_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.adsorbate_env_bb, self.bulk_d.all_bulk_env, 
                                                                                                      self.adslab, slab_env_dict=self.all_adslab_env, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                      node_energy=False, dimer_diff=False, sqrted=True, generalize_bb=True)
            self.all_param_descriptors['ads_local_gbb_ebond_sqrt_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.adsorbate_env_bb, self.bulk_d.all_bulk_env, 
                                                                                                       self.adslab, slab_env_dict=self.all_adslab_env, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                       node_energy=True, dimer_diff=False, sqrted=True, generalize_bb=True)
        ######################################################################################################################################################
        
        # Repeat for next nearest neighbors
        ######################################################################################################################################################
                        
        # Describe all combinations of broken bonds and Enodes and Ebonds and E=1 for adslab 
        self.all_param_descriptors['adslab_bb_nnn_e1'] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                self.adslab, ecoh=None, node_energy=False, dimer_diff=False)
        self.all_param_descriptors['adslab_bb_nnn_e1_sqrt'] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                       self.adslab, ecoh=None, node_energy=False, dimer_diff=False, sqrted=True)
        self.all_param_descriptors['adslab_bb_nnn_edimer'] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                      self.adslab, ecoh=None, node_energy=False, dimer_diff=True)
        self.all_param_descriptors['adslab_bb_nnn_edimer_sqrt'] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                                  self.adslab, ecoh=None, node_energy=False, dimer_diff=True, sqrted=True)
        
        # Generalized CN
        self.all_param_descriptors['adslab_gbb_nnn_e1'] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                self.adslab, slab_env_dict=self.all_adslab_nnn_env, ecoh=None, node_energy=False, dimer_diff=False, generalize_bb=True)
        self.all_param_descriptors['adslab_gbb_nnn_e1_sqrt'] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                self.adslab, slab_env_dict=self.all_adslab_nnn_env, ecoh=None, node_energy=False, dimer_diff=False,
                                                                                               sqrted=True, generalize_bb=True)
        self.all_param_descriptors['adslab_gbb_nnn_edimer'] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                self.adslab, slab_env_dict=self.all_adslab_nnn_env, ecoh=None, node_energy=False, dimer_diff=True, generalize_bb=True)
        self.all_param_descriptors['adslab_gbb_nnn_edimer_sqrt'] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                self.adslab, slab_env_dict=self.all_adslab_nnn_env, ecoh=None, node_energy=False, 
                                                                                                   dimer_diff=True, sqrted=True, generalize_bb=True)
        
        for node_name in self.bulk_d.node_properties.keys():
            self.all_param_descriptors['adslab_bb_nnn_node_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                                      self.adslab, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                      node_energy=False, dimer_diff=False)
            self.all_param_descriptors['adslab_bb_nnn_ebond_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                                       self.adslab, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                       node_energy=True, dimer_diff=False)
            self.all_param_descriptors['adslab_bb_nnn_node_sqrt_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                                      self.adslab, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                      node_energy=False, dimer_diff=False, sqrted=True)
            self.all_param_descriptors['adslab_bb_nnn_ebond_sqrt_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                                       self.adslab, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                       node_energy=True, dimer_diff=False, sqrted=True)

            # Generalized CN
            self.all_param_descriptors['adslab_gbb_nnn_node_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                self.adslab, slab_env_dict=self.all_adslab_nnn_env, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                      node_energy=False, dimer_diff=False, generalize_bb=True)
            self.all_param_descriptors['adslab_gbb_nnn_ebond_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                self.adslab, slab_env_dict=self.all_adslab_nnn_env, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                       node_energy=True, dimer_diff=False, generalize_bb=True)
            self.all_param_descriptors['adslab_gbb_nnn_node_sqrt_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                self.adslab, slab_env_dict=self.all_adslab_nnn_env, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                      node_energy=False, dimer_diff=False, sqrted=True, generalize_bb=True)
            self.all_param_descriptors['adslab_gbb_nnn_ebond_sqrt_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                self.adslab, slab_env_dict=self.all_adslab_nnn_env, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                       node_energy=True, dimer_diff=False, sqrted=True, generalize_bb=True)

        # Describe all combinations of broken bonds and Enodes and Ebonds and E=1 for local adsorbate env
        self.all_param_descriptors['ads_local_bb_nnn_e1'] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                self.adslab, ecoh=None,
                                                                                node_energy=False, dimer_diff=False)
        self.all_param_descriptors['ads_local_bb_nnn_e1_sqrt'] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                self.adslab, ecoh=None,
                                                                                node_energy=False, dimer_diff=False, sqrted=True)
        self.all_param_descriptors['ads_local_bb_nnn_edimer'] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                      self.adslab, ecoh=None,
                                                                                      node_energy=False, dimer_diff=True)
        self.all_param_descriptors['ads_local_bb_nnn_edimer_sqrt'] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                      self.adslab, ecoh=None,
                                                                                      node_energy=False, dimer_diff=True, sqrted=True)

        # Generalize BB
        self.all_param_descriptors['ads_local_gbb_nnn_e1'] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                self.adslab, slab_env_dict=self.all_adslab_nnn_env, ecoh=None,
                                                                                node_energy=False, dimer_diff=False, generalize_bb=True)
        self.all_param_descriptors['ads_local_gbb_nnn_e1_sqrt'] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                self.adslab, slab_env_dict=self.all_adslab_nnn_env, ecoh=None,
                                                                                node_energy=False, dimer_diff=False, sqrted=True, generalize_bb=True)
        self.all_param_descriptors['ads_local_gbb_nnn_edimer'] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                self.adslab, slab_env_dict=self.all_adslab_nnn_env, ecoh=None,
                                                                                      node_energy=False, dimer_diff=True, generalize_bb=True)
        self.all_param_descriptors['ads_local_gbb_nnn_edimer_sqrt'] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                self.adslab, slab_env_dict=self.all_adslab_nnn_env, ecoh=None,
                                                                                      node_energy=False, dimer_diff=True, sqrted=True, generalize_bb=True)

        for node_name in self.bulk_d.node_properties.keys():
            self.all_param_descriptors['ads_local_bb_nnn_node_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                                      self.adslab, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                      node_energy=False, dimer_diff=False)
            self.all_param_descriptors['ads_local_bb_nnn_ebond_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                                       self.adslab, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                       node_energy=True, dimer_diff=False)
            self.all_param_descriptors['ads_local_bb_nnn_node_sqrt_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                                      self.adslab, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                      node_energy=False, dimer_diff=False, sqrted=True)
            self.all_param_descriptors['ads_local_bb_nnn_ebond_sqrt_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                                       self.adslab, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                       node_energy=True, dimer_diff=False, sqrted=True)
            
            # Generalize BB
            self.all_param_descriptors['ads_local_gbb_nnn_node_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                self.adslab, slab_env_dict=self.all_adslab_nnn_env, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                      node_energy=False, dimer_diff=False, generalize_bb=True)
            self.all_param_descriptors['ads_local_gbb_nnn_ebond_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                self.adslab, slab_env_dict=self.all_adslab_nnn_env, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                       node_energy=True, dimer_diff=False, generalize_bb=True)
            self.all_param_descriptors['ads_local_gbb_nnn_node_sqrt_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                self.adslab, slab_env_dict=self.all_adslab_nnn_env, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                      node_energy=False, dimer_diff=False, sqrted=True, generalize_bb=True)
            self.all_param_descriptors['ads_local_gbb_nnn_ebond_sqrt_%s' %(node_name)] = self.slab_d.get_broken_bonds_energy(self.all_adslab_nnn_env_bb, self.bulk_d.all_bulk_nnn_env, 
                                                                                self.adslab, slab_env_dict=self.all_adslab_nnn_env, ecoh=self.bulk_d.node_properties[node_name], 
                                                                                                       node_energy=True, dimer_diff=False, sqrted=True, generalize_bb=True)
        ######################################################################################################################################################
        
        # Describe all combinations of broken bonds and Enodes and Ebonds and E=1 for surface - adslab 
        all_descriptor_names = list(self.all_param_descriptors.keys())
        for k in all_descriptor_names:
            if 'adslab_bb_' in k or 'adslab_gbb_' in k:
                t = 'adslab_bb_' if 'adslab_bb_' in k else 'adslab_gbb_'
                Eadslab = self.all_param_descriptors[k]
                Eslab = self.all_param_descriptors[k.replace('adslab_', 'bare_')]
                self.all_param_descriptors['Eadslab-Eslab_%s' %(k.replace('adslab_', ''))] = Eadslab-Eslab
                
        # Misc descriptors 
        self.all_param_descriptors['vol_added_to_poly_w_ads'] = self.vol_added_to_poly_w_ads()
        self.all_param_descriptors['vol_added_to_poly_w_ads_sqrt'] = self.vol_added_to_poly_w_ads() **(1/2)
                    
    def ave_node_diff(self, node_properties):
        env_ave_node = np.mean([node_properties[site.bulk_equivalent] for site in self.adsorbate_env])
        ads_comp = Composition(self.ads).as_dict()
        ads_node = node_properties[self.Oads.bulk_equivalent]
        return env_ave_node - ads_node
        
        # Add a sqrt term
        
        # Repeat for next nearest neighbors
        
        # For polyhedrons
        
    def vol_added_to_poly_w_ads(self):

        vol_added = []
        for k in self.ads_poly_dict.keys():
            ads_poly = self.ads_poly_dict[k]
            ads_poly_Osites = [site for site in ads_poly if site.species_string == 'O']
            slab_poly = self.slab_d.slab_poly_dict[k]

            try:
                slab_poly_vol = ConvexHull([site.coords for site in slab_poly]).volume if len(slab_poly) > 3 else 0
            except QhullError:
                slab_poly_vol = 0
            try:
                adslab_poly_vol = ConvexHull([site.coords for site in ads_poly_Osites]).volume if len(ads_poly_Osites) > 3 else 0
            except QhullError:
                adslab_poly_vol = 0

            vol_added.append(adslab_poly_vol-slab_poly_vol)

        return sum(vol_added)    
    
    # Functions for getting local environment info

    def get_adsorbate_polyhedron_dict(self):
        adsites = [site for site in self.adslab if site.surface_properties == 'adsorbate']
        adsites = sorted(adsites, key=lambda site: site.coords[2])
        Oads = [site for site in adsites if site.species_string == 'O'][0]

        msites = sorted([nn for nn in self.adslab.get_neighbors(Oads, 5) 
                         if nn.species_string not in ['O', 'H']],
                        key=lambda nn: nn.distance(Oads))

        ads_poly_dict = {}
        for msite in msites:
            ads_poly = get_metal_polyhedron_Osites(self.adslab, msite)
            if any([site == Oads for site in ads_poly]):
                ads_poly_dict[msite.index] = ads_poly
        
        return ads_poly_dict

    def get_adsorbate_coordination_env(self, ignore_adspecies=True):

        adsites = [site for site in self.adslab if site.surface_properties == 'adsorbate']
        adsites = sorted(adsites, key=lambda site: site.coords[2])
        Oads = [site for site in adsites if site.species_string == 'O'][0]
        setattr(Oads, 'index', self.adslab.index(Oads))
        adsorbate_env = [ss for ss in self.adslab.get_neighbors(Oads, self.blength)]
        
        # assign the O-adsorbate a bulk equivalent
        msite = [site for site in adsorbate_env if site.species_string not in ['O', 'H']][0]
        bulk_i = [site for site in self.slab_d.all_slab_bb[msite.index] if site.species_string == 'O'][0].index
        Oads.properties['bulk_equivalent'] = bulk_i
        
        if ignore_adspecies:
            adsorbate_env = [site for site in adsorbate_env if site.surface_properties != 'adsorbate']
            return Oads, msite, adsorbate_env
        else:
            return Oads, msite, adsorbate_env

    def get_adslab_coordination_env(self):
        all_adslab_env = copy.deepcopy(self.slab_d.all_slab_env)
        for site in self.adsorbate_env:
            all_adslab_env[site.index].append(self.Oads)
        all_adslab_env[self.Oads.index] = self.adsorbate_env
        return all_adslab_env
    
    def get_ads_energy(self):
        # Get the surface energy at T, pH, U = 0, 0, 0
        return self.slab_d.queryengine.get_equation(self.adslab_entry, surface_energy=False, 
                                             adsorbate=self.adslab_entry.data['adsorbate'])['1']

    
# Helper functions
def get_slab_entry(adslab_doc, slab_doc, queryengine):
    rid = adslab_doc['rid']
    dat = Data.from_dict(adslab_doc)
    if slab_doc:
        slabdat = Data.from_dict(slab_doc)
        slab_entries = queryengine.get_slab_entries({}, dat_list=[dat, slabdat], relaxed=False)
    else:
        slab_entries = queryengine.get_slab_entries({}, dat_list=[dat], relaxed=False)
    return slab_entries[rid]   

def get_coordination_env_per_site(struct, site, blength=3):
    return [ss for ss in struct.get_neighbors(site, blength)]

def get_all_coordination_env(struct, blength=3):
    env_dict = {}
    for i, site in enumerate(struct):
        env_dict[i] = get_coordination_env_per_site(struct, site, blength=blength)
    return env_dict

def get_nnn_env(struct, blength=3, dr=3):
    nn_env_dict = {}
    for i, site in enumerate(struct):
        nn_env_dict[i] = struct.get_neighbors_in_shell(site.coords, blength+0.1, dr)
    return nn_env_dict

def get_metal_polyhedron_Osites(struct, site, include=['O']):

    # Collect nearest neighboring oxygen site to determine minimum O-M bondlength    
    Osites = []
    blength = 2.65
    while len(Osites) == 0:
        Osites = [ss for ss in struct.get_neighbors(site, blength) 
                  if ss.species_string in include]
        blength+=0.2
        if blength > 3.5:
            break

    # Collect all neighboring oxygen sites with min bondlength within tol
    if Osites:
        blengths = [ss.distance(site) for ss in Osites if ss.species_string in include]
        if max(blengths) - min(blengths) > 0.3 and max(blengths) > 3:
            blength = min(blengths)*1.2
        Osites = [ss for ss in struct.get_neighbors(site, blength) if ss.species_string in include]
    else:
        Osites = []

    return Osites

def get_polyhedron_dict(struc):
    poly_dict = {}
    for i, site in enumerate(struc):
        if site.species_string in ['O']:
            continue
        poly_dict[i] = get_metal_polyhedron_Osites(struc, site)
    return poly_dict
