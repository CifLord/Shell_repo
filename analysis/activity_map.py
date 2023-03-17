import numpy as np
import matplotlib.pyplot as plt
import copy

def get_activity_map(increment=500, xrange=[0,3], yrange=[-1,3], title='OER activity'):

    DGOminOH = np.linspace(xrange[0],xrange[1], increment)
    DGOH = np.linspace(-1,3, increment)

    DGOminOH, DGOH = np.meshgrid(DGOminOH, DGOH)

    DGO = DGOminOH+DGOH
    DGOOH = DGOH + [[3.2]*increment]*increment
    DGOOHminO = DGOOH-DGO
    final = [[4.92]*increment]*increment - DGOOH

    e =4
    noer = []
    for i, row in enumerate(DGOH):
        new_row = []
        for ii, dgOH in enumerate(row):
            new_row.append(max([dgOH, DGOminOH[i][ii], DGOOH[i][ii]-DGO[i][ii], 4.92-DGOOH[i][ii]])-1.23)
        noer.append(new_row)

    noer = np.array(noer)
    noer = noer[:-1, :-1]

    fig, ax = plt.subplots()
    ax.set_title(title, fontsize=20)
    plt.pcolormesh(DGOminOH, DGOH, noer, cmap='rainbow', 
                   vmin=-np.abs(noer).max(), 
                   vmax=np.abs(noer).max())
    cbar = plt.colorbar()
    cbar.set_label(r'$\eta_{OER}$ (V)', fontsize=20)
    plt.clim(0, 2) 
    plt.xlabel(r'$\Delta G_{O^*}-\Delta G_{OH^*}$ (eV)', fontsize=20)
    plt.ylabel(r'$\Delta G_{OH^*}$ (eV)', fontsize=20)
    return plt
