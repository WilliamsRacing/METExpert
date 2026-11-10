"""Reusable immutable-data inference loader for repeated EMA validation/mining."""
import torch
from torch.utils.data import DataLoader
from .data import SequenceDataset


def inference_loader(records,scaler,config,device):
    core=int(config.get('inference',{}).get('core_frames',config['core_frames']))
    halo=int(config.get('inference',{}).get('halo_frames',32))
    dataset=SequenceDataset(records,scaler,core+2*halo,core,False,False,
        lru_records=int(config['training'].get('lru_records',4)))
    workers=int(config['training'].get('num_workers',0))
    kwargs={'persistent_workers':True,'prefetch_factor':int(config['training'].get('prefetch_factor',2))} if workers else {}
    return DataLoader(dataset,batch_size=int(config['inference'].get('batch_size',4)),shuffle=False,
        num_workers=workers,pin_memory=device.type=='cuda',generator=torch.Generator().manual_seed(int(config['seed'])),**kwargs)
