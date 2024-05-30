from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.core.structure import Composition


rds_dict = {'H2O-->OH*+H^+': "$\Delta G^{i}$", 
            'OH-->O*+H^+': "$\Delta G^{ii}-\Delta G^{i}$", 
            'H2O+O*-->OOH*+H^+': "$\Delta G^{iii}-$\Delta G^{ii}$",
            'OOH*-->O2+H^+': "$4.92-\Delta G^{iii}$"}

def make_tex_table(data_dict, formulas, limit_break_only=False):
    k1 = list(data_dict.keys())[0]
    l1 = data_dict[k1]
    headings = list(data_dict.keys())
    heading_str = "Formulas & "
    row = "\ce{%s} & "
    for i, k in enumerate(headings):
        if type(data_dict[k][0]).__name__ == 'str':
            row+='%s '
        else:
            row+='%.2f '
        heading_str+=str(k)+' '
        if i < len(headings)-1:
            heading_str+= '& '
            row+= '& '
    heading_str+='\\\\'
    row+='\\\\'
    print(heading_str)
    print('\hline')
    print('\hline')

    for i, d1 in enumerate(l1):
        dat = [formulas[i]]
        dat.extend([data_dict[k][i] for k in data_dict.keys()])
        if limit_break_only and dat[4] > 0.3:
            continue
        print(row %tuple(dat))
        print('\hline')


def screen_all_data(stable_rid_dict, pbx_stable_mpids, all_activity, 
                    costanalyzer, bulk_oxides_dict, nfacet_thresh=1, 
                    op_thresh=0.75, ehull_thresh=0.1, cost_thresh=18315, 
                    check_for_ooh_eta=False, limit_break_only=False):
    
    print('Pourbaix stable', len(pbx_stable_mpids))

    count_wulff, count_active = 0, 0
    mpids_active = {}
    rids_stats = {}
    
    activity_dict = {}
    activity_dict_ooh = {}
    rate_determining_step = {}

    for mpid in tqdm(stable_rid_dict.keys()):
        activity_dict[mpid] = []
        activity_dict_ooh[mpid] = []
        rate_determining_step[mpid] = []

        if mpid not in pbx_stable_mpids:
            continue

        slab_rids = []
        for dmu in stable_rid_dict[mpid].keys():
            if not stable_rid_dict[mpid][dmu]:
                continue
            for hkl in stable_rid_dict[mpid][dmu].keys():
                slab_rid = stable_rid_dict[mpid][dmu][hkl]
                if slab_rid not in slab_rids:
                    count_wulff+=1
                    
                    if hkl in all_activity[mpid].keys():
                        if slab_rid in all_activity[mpid][hkl].keys():
                            if all_activity[mpid][hkl][slab_rid]['scaled_op'] <= op_thresh:
                                real_op = all_activity[mpid][hkl][slab_rid]['op']
                                scaled_op = all_activity[mpid][hkl][slab_rid]['scaled_op']
                                    
                                if check_for_ooh_eta:
                                    if not real_op:
                                        continue
                                    if real_op > op_thresh+0.25:
                                        continue
                                
                                activity_dict[mpid].append(all_activity[mpid][hkl][slab_rid]['scaled_op'])
                                rate_determining_step[mpid].append(all_activity[mpid][hkl][slab_rid]['rds'])
                                if all_activity[mpid][hkl][slab_rid]['op']:
                                    activity_dict_ooh[mpid].append(all_activity[mpid][hkl][slab_rid]['op'])
                                else:
                                    activity_dict_ooh[mpid].append(1000)

                                count_active+=1
                                if mpid not in mpids_active.keys():
                                    mpids_active[mpid] = []
                                if hkl not in mpids_active[mpid]:
                                    mpids_active[mpid].append(hkl)
                                
                                slab_rids.append(slab_rid)
                    
        rids_stats[mpid] = slab_rids
        
    mpids_active_multfacets = [mpid for mpid in mpids_active.keys() \
                               if len(mpids_active[mpid]) > nfacet_thresh]
    print('all stable rids on wulff', count_wulff, 'all stable rids on wulff, active',
          count_active, 'all active mpids', len(mpids_active_multfacets))

    metastable = []
    for mpid in mpids_active_multfacets:
        entry = ComputedStructureEntry.from_dict(bulk_oxides_dict[mpid])
        if entry.data['e_above_hull'] <= ehull_thresh:
            metastable.append(mpid)
    print('Metastable', len(metastable))

    cheap = []
    for mpid in metastable:
        entry = ComputedStructureEntry.from_dict(bulk_oxides_dict[mpid])
        if cost_thresh > costanalyzer.get_cost_per_kg(entry.composition):
            cheap.append(mpid)
    print('Cheap', len(cheap))
    print()
    
    formulas = [Composition(bulk_oxides_dict[mpid]['composition']).reduced_formula for mpid in cheap]
    active_hkls = [len(mpids_active[mpid]) for mpid in cheap]
    rds_list = []
    actual_op = []
    final_slab_rids = {}
    for mpid in cheap:
        i = activity_dict[mpid].index(min(activity_dict[mpid]))
        rds_list.append(rate_determining_step[mpid][i])
        final_slab_rids[mpid] = rids_stats[mpid]
        if activity_dict_ooh[mpid][i]:
            actual_op.append(activity_dict_ooh[mpid][i])
        else:
            actual_op.append(1000)
    activities = [min(activity_dict[mpid]) for mpid in cheap]
    price = [costanalyzer.get_cost_per_kg(f) for f in formulas]
    price, active_hkls, formulas, activities, cheap, rds_list, actual_op = \
                                           zip(*sorted(zip(price, active_hkls, formulas, activities,
                                                           cheap, rds_list, actual_op)))

    table_data = {'# Facet': active_hkls, 'Scaled $\eta$ (V)': activities, 'Cost': price, 
                  '$\eta$ (V)': actual_op, 'RDS': rds_list}
    make_tex_table(table_data, formulas, limit_break_only=limit_break_only)
    
    return table_data, formulas, final_slab_rids
