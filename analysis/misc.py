def str_to_hkl(s):
    hkl = []
    norm = 1
    for i in s:
        if i in ['(', ')', ',', ' ']:
            continue
        if i == '-':
            norm = -1
            continue
        hkl.append(norm * int(i))
        norm = 1
    return tuple(hkl)