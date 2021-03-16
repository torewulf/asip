import os
import cv2
import netCDF4
import utils
import numpy as np
from tqdm import tqdm

DATA_DIR = '/data/users/twu/ds-2/dataset-2'
PATCHES_DIR = '/data/users/twu/ds-2/patches'
PATCH_SHAPE = (800, 800)
AMSR_PATCH_SHAPE = (16, 16)
OVERLAP = 0.25

ncs = [os.path.join(DATA_DIR, file) for file in os.listdir(DATA_DIR) if file.endswith('.nc')]

amsr_labels = ['btemp_6.9h', 'btemp_6.9v', 'btemp_7.3h', 'btemp_7.3v', 
               'btemp_10.7h', 'btemp_10.7v', 'btemp_18.7h', 'btemp_18.7v', 
               'btemp_23.8h', 'btemp_23.8v', 'btemp_36.5h', 'btemp_36.5v', 
               'btemp_89.0h', 'btemp_89.0v']

if not os.path.exists(PATCHES_DIR):
    os.mkdir(PATCHES_DIR)

### PROCESSING ###
for nc_file in tqdm(ncs):
    ncf = netCDF4.Dataset(nc_file)

    scene_path = os.path.join(PATCHES_DIR, nc_file.split('/')[-1][:-3])
    if not os.path.exists(scene_path):
        os.mkdir(scene_path)

    # Extracting variables from the .nc-file
    HH = np.array(ncf.variables.get('sar_primary'))
    HV = np.array(ncf.variables.get('sar_secondary'))
    HH_nersc = np.array(ncf.variables.get('nersc_sar_primary'))
    HV_nersc = np.array(ncf.variables.get('nersc_sar_secondary'))
    IC = np.array(ncf.variables.get('polygon_icechart'))
    DST = np.array(ncf.variables.get('distance_map')).astype('float32')
    IC_codes = list(ncf.variables.get('polygon_codes'))
    CT = utils.extract_IC_attribute(IC, IC_codes, attribute='CT')
    INC = np.tile(np.array(ncf.variables.get('sar_incidenceangles')), (CT.shape[0], 1))
    AMSR = [cv2.resize(np.array(ncf.variables.get(label)), (CT.shape[1], CT.shape[0]), interpolation=cv2.INTER_LINEAR) for label in amsr_labels]

    # Replacing invalid data with NaNs
    no_data = np.logical_or(np.isnan(HH), np.isnan(HH_nersc))
    CT[no_data] = np.nan
    DST[no_data] = np.nan
    INC[no_data] = np.nan
    HH_nersc[no_data] = np.nan
    HV_nersc[no_data] = np.nan
    for AMSR_channel in AMSR: AMSR_channel[no_data] = np.nan

    # Extract all patches (shape=patch_shape) from HH and return indices of all patches without NaN values
    HH_patches, non_nan_idxs = utils.extract_patches(HH, patch_shape=PATCH_SHAPE, return_non_nan_idxs=True, overlap=OVERLAP)
    if not len(non_nan_idxs) == 0: # if valid non-NaN patches in scene
        HH_patches = HH_patches[non_nan_idxs]
        del HH
        HV_patches = utils.extract_patches(HV, patch_shape=PATCH_SHAPE, overlap=OVERLAP)[non_nan_idxs]
        del HV
        HH_nersc_patches = utils.extract_patches(HH_nersc, patch_shape=PATCH_SHAPE, overlap=OVERLAP)[non_nan_idxs]
        del HH_nersc
        HV_nersc_patches = utils.extract_patches(HV_nersc, patch_shape=PATCH_SHAPE, overlap=OVERLAP)[non_nan_idxs]
        del HV_nersc
        CT_patches = utils.extract_patches(CT, patch_shape=PATCH_SHAPE, overlap=OVERLAP)[non_nan_idxs]
        del CT
        DST_patches = utils.extract_patches(DST, patch_shape=PATCH_SHAPE, overlap=OVERLAP)[non_nan_idxs]
        del DST
        INC_patches = utils.extract_patches(INC, patch_shape=PATCH_SHAPE, overlap=OVERLAP)[non_nan_idxs]
        del INC
        AMSR_patches = [utils.extract_patches(AMSR_channel, patch_shape=PATCH_SHAPE, overlap=OVERLAP, new_shape=AMSR_PATCH_SHAPE)[non_nan_idxs] for AMSR_channel in AMSR]
        del AMSR
        AMSR_patches = np.stack(AMSR_patches, axis=1)
        
        for patch in range(HH_patches.shape[0]):
            np.save(os.path.join(scene_path, str(patch) + '_S1.npy'), np.stack((HH_patches[patch], HV_patches[patch]), axis=0).astype('float32'))
            np.save(os.path.join(scene_path, str(patch) + '_S1_nersc.npy'), np.stack((HH_nersc_patches[patch], HV_nersc_patches[patch]), axis=0).astype('float32'))
            np.save(os.path.join(scene_path, str(patch) + '_INC.npy'), INC_patches[patch].astype('float32'))
            np.save(os.path.join(scene_path, str(patch) + '_CT.npy'), CT_patches[patch].astype('uint8'))
            np.save(os.path.join(scene_path, str(patch) + '_DST.npy'), DST_patches[patch].astype('uint8'))
            np.save(os.path.join(scene_path, str(patch) + '_AMSR.npy'), AMSR_patches[patch].astype('float32')) 
