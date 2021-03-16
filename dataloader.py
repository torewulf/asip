import torch
import torch.nn.functional as F
import utils
import pickle
import numpy as np
import cv2
import albumentations as A
import pandas as pd
import os

class ASIPDS(torch.utils.data.Dataset):
    def __init__(self, scenes, crop_size, downsample=False):
        # Make sure patches with AMSR NaNs are removed 
        patches_folder = '/data/users/twu/ds-2/patches'
        df = pd.read_csv(os.path.join(patches_folder, 'tiles_info.txt'), header=None, names=['path', 'AMSRNaN', 'CT'])
        AMSRNaNPatches = list(df[df['AMSRNaN'] == 1]['path'].values)

        self.patch_paths = [patch for patch in utils.get_patch_paths(scenes) if patch not in AMSRNaNPatches]
        
        self.crop_size = crop_size
        self.downsample = downsample
        
        # Defining a data augmentation pipeline
        self.transform = A.Compose([A.RandomCrop(width=crop_size, height=crop_size, always_apply=True),
                                    A.HorizontalFlip(p=0.5),
                                    A.VerticalFlip(p=0.5)],
                                    additional_targets = {'INC': 'image', 'CT': 'image', 'DST': 'image', 'AMSR': 'image'})
        
        # Initializing data scalers
        self.INC_scaler = pickle.load(open('scalers/INC_scaler.pkl', 'rb'))
        self.DST_scaler = pickle.load(open('scalers/DST_scaler.pkl', 'rb'))
        self.AMSR_scaler = pickle.load(open('scalers/AMSR_scaler.pkl', 'rb'))
        
    def __len__(self):
        return len(self.patch_paths)
    
    def __getitem__(self, index):
        S1, INC, CT, DST, AMSR = utils.get_patch(self.patch_paths[index])
        
        AMSR = np.array([cv2.resize(AMSR_channel, CT.shape, interpolation=cv2.INTER_LINEAR) for AMSR_channel in AMSR])

        transformed = self.transform(image=np.transpose(S1, (1, 2, 0)), # albumentations lib requires (HxWxC)-shape
                        INC=INC,
                        CT=CT,
                        DST=DST,
                        AMSR=np.transpose(AMSR, (1, 2, 0)))

        INC = self.INC_scaler.transform(transformed['INC'].reshape(-1, 1)).reshape(transformed['INC'].shape)
        DST = self.DST_scaler.transform(transformed['DST'].reshape(-1, 1)).reshape(transformed['DST'].shape)
        AMSR = self.AMSR_scaler.transform(transformed['AMSR'].reshape(-1, 1)).reshape(transformed['AMSR'].shape)
        S1 = transformed['image']
        CT = transformed['CT'] 
        
        S1 = torch.from_numpy(np.transpose(S1, (2, 0, 1)))
        INC = torch.from_numpy(INC)
        AMSR = torch.from_numpy(np.transpose(AMSR, (2, 0, 1)))
        DST = torch.from_numpy(DST)
        CT = torch.from_numpy(CT.astype('float32'))

        if self.downsample:
            S1 = F.interpolate(S1.unsqueeze(dim=0), size=int(self.crop_size/2), mode='bilinear', align_corners=True).squeeze()
            INC = F.interpolate(INC.unsqueeze(dim=0).unsqueeze(dim=0), size=int(self.crop_size/2), mode='bilinear', align_corners=True).squeeze()
            DST = F.interpolate(DST.unsqueeze(dim=0).unsqueeze(dim=0), size=int(self.crop_size/2), mode='bilinear', align_corners=True).squeeze()
            AMSR = F.interpolate(AMSR.unsqueeze(dim=0), size=int(self.crop_size/2), mode='bilinear', align_corners=True).squeeze()
            CT = F.interpolate(CT.unsqueeze(dim=0).unsqueeze(dim=0), size=int(self.crop_size/2), mode='nearest').squeeze()

        CT = utils.CT_to_class(CT)
        
        return (S1.float(), INC.unsqueeze(dim=0).float(), DST.unsqueeze(dim=0).float(), AMSR.float()), CT.unsqueeze(dim=0).type(torch.LongTensor)
