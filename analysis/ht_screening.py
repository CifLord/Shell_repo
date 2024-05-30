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

