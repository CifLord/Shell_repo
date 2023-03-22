import numpy as np
import matplotlib.pyplot as plt
import copy

def get_activity_map(increment=500, xrange=[0,3], yrange=[-1,3], T=0, U=0, proton_activity=1,
                     title='OER activity', slopeOHtoOOH=0.73, intOHtoOOH=3.44):

    DGOminOH = np.linspace(xrange[0],xrange[1], increment)
    DGOH = np.linspace(yrange[0], yrange[1], increment)

    DGOminOH, DGOH = np.meshgrid(DGOminOH, DGOH)

    DGO = DGOminOH+DGOH
    DGOOH = slopeOHtoOOH*DGOH + [[intOHtoOOH]*increment]*increment
    DGOOHminO = DGOOH-DGO
    
    kb = 1.380649 * 10**(-23)
    electron_transfer = np.array([-1*U + kb*T * np.log(proton_activity)]*4)
    
    e =4
    noer = []
    for i, row in enumerate(DGOH):
        new_row = []
        for ii, dgOH in enumerate(row):
            all_DG = np.array([dgOH, DGOminOH[i][ii], DGOOH[i][ii]-DGO[i][ii], 4.92-DGOOH[i][ii]]) + electron_transfer
            new_row.append(max(all_DG)-1.23)
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
