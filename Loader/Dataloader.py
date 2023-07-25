
import os
from torch.utils.data.distributed import DistributedSampler
#from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
import torch.distributed as dist
class DistributedDataLoader(DataLoader):
    
    '''master node:
       the main gpu responsible for synchronizations, making copies, loading models, writing logs;
       process group:
       if you want to train/test the model over K gpus, then the K process forms a group,
       which is supported by a backend (pytorch managed that for you, according to the documentation,
       nccl is the most recommended backend);
       rank: 
       within the process group, each process is identified by its rank, from 0 to K-1;
       world size: the number of processes in the group i.e. gpu number——K.
    '''
    
    def __init__(self,dataset, batch_size, rank, world_size,pin_memory=False, num_workers=0,drop_last=False):
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,drop_last=True)
        super().__init__(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, sampler=sampler,drop_last=False)


def setup(rank,world_size):
    os.environ['MASTER_ADDR']='localhost'
    #os.environ['MASTER_PORT']='12355'
    #num_threads=1
    #os.environ['OMP_NUM_THREADS']=str(num_threads)
    dist.init_process_group("nccl",rank=rank,world_size=world_size)
    
