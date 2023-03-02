from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from ocpmodels.datasets import LmdbDataset
import os
import lmdb
import sys
import pickle
from ase import Atoms
from ase.optimize import BFGS
from tqdm import tqdm
from script_lee import sv_slabs, cal_slab_energy
import threading


class myTread(threading.Thread):
    def __init__(self,threadID,name):
        threading.Thread.__init__(self)
        self.threadID=threadID
        self.name=name
        
    def run(self):        
        data_list_E=[]
        tot=len(data_list)                
        if self.threadID==1:
            lp=range(int(tot/3))
            calc = OCPCalculator(config_yml='../configs/oc22/s2ef/gemnet-oc/gemnet_oc_oc20_oc22.yml', checkpoint="./prediction/gemnet_oc_base_oc20_oc22.pt",device='0')
        elif self.threadID==2:
            lp=range(int(tot/3),int(2*tot/3))
            calc = OCPCalculator(config_yml='../configs/oc22/s2ef/gemnet-oc/gemnet_oc_oc20_oc22.yml', checkpoint="./prediction/gemnet_oc_base_oc20_oc22.pt",device='0')
        else:
            lp=range(int(tot*2/3),tot)
            calc = OCPCalculator(config_yml='../configs/oc22/s2ef/gemnet-oc/gemnet_oc_oc20_oc22.yml', checkpoint="./prediction/gemnet_oc_base_oc20_oc22.pt",device='0')
            
        for i in tqdm(lp):
            data=data_list[i]
            unrelax_slab_energy, relax_slab_energy,forces, pos_relaxed=cal_slab_energy(data,calc)
            data.y=relax_slab_energy
            data.unrelax_energy=unrelax_slab_energy
            data.pos_relaxed=pos_relaxed
            data.force=forces
            data_list_E.append(data)
            if len(data_list_E)>=50: 
                pathname = 'datasetss/slabs_rare'+self.name
                sv_slabs(data_list_E, pathname)
                data_list_E=[]
                
        pathname = 'datasetss/slabs_rare'+self.name
        sv_slabs(data_list_E, pathname)
        data_list_E=[] 


if __name__=="__main__":

    data_list=LmdbDataset({"src": "datasetss/slabs_rare.lmdb"})
    thread1=myTread(1,'T1')
    thread2=myTread(2,'T2')
    thread3=myTread(3,'T3')

    thread1.start()   
    sys.stdout = open(os.devnull, "w")
    thread2.start()   
    thread3.start()  
   
