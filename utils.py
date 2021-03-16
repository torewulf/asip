def get_scene_paths(patches_folder):
    import os
    
    return [os.path.join(patches_folder, scene) for scene in os.listdir(patches_folder) if os.path.isdir(os.path.join(patches_folder, scene))]


def get_patch_len(scene_path):
    import os

    return len(os.listdir(scene_path))//6 # as 6 .npy-files are saved per patch 


def get_patch_paths(scenes):
    import utils
    import os

    return [os.path.join(scene, str(n)) for scene in scenes for n in range(utils.get_patch_len(os.path.join(scene)))]


def get_patch(patch_path):
    import numpy as np
    import os
    
    return (np.load(os.path.join(patch_path + '_S1.npy')),
           np.load(os.path.join(patch_path + '_INC.npy')),
           np.load(os.path.join(patch_path + '_CT.npy')),
           np.load(os.path.join(patch_path + '_DST.npy')),
           np.load(os.path.join(patch_path + '_AMSR.npy')),)


def CT_to_class(CT):
    """
    Converts a CT layer from raw sea ice concentrations (0-100) to class ids (0-10).
    """
    
    CTs = list(range(0, 110, 10))
    class_ids = list(range(0, 11, 1))
    for i in range(len(CTs)):
        CT[CT == CTs[i]] = class_ids[i]
                   
    return CT


def extract_IC_attribute(IC, IC_codes, attribute):
    """
    This scripts extracts an attribute from the ice charts in the ASIPv2 dataset.
    Thus far only implemented to extract CT. 

    Notes:
    If CT is 01 (less than 1/10 ice concentration or 02 (bergy water) an ice conentration of 0 is assumed. 
    If CT is 91 (9/10-10/10) or 92 (10/10 - fast ice) an ice concentration of 100% is assumed.

    args:
        IC: the 'polygon_icechart' variable
        IC_codes: the 'polygon_codes' variable
    returns:
        canvas: an ndarray containing sea ice concentrations ranging from 0%-100%.
    """
    import pandas as pd
    import numpy as np
    
    columns = IC_codes[0].split(';')
    vals = [row.split(';') for row in IC_codes[1:]]
    df = pd.DataFrame(np.vstack(np.array(vals)), columns=columns)
    
    canvas = np.zeros(IC.shape)
    for val in np.unique(IC[IC != 0]):
        canvas[IC == val] = df[attribute][df['id'].astype(int) == val].astype(int).values[0]
    
    canvas[canvas == 1] = 0
    canvas[canvas == 2] = 0
    canvas[canvas == 91] = 100
    canvas[canvas == 92] = 100
    
    return canvas


def extract_patches(data, patch_shape, new_shape=None, return_non_nan_idxs=False, overlap=0.0):
    """
    This function divides a 2d ndarray into a number of patches of shape patch_shape and returns the patches stacked in a numpy array.
    Square patches are assumed.
    If the new_shape argument is given, the patches are resized to the new shape.
    If return_non_nan_idxs=True, a list of indices of the patches without NaN-values is returned as well.
    """
    from skimage.util import view_as_windows
    import numpy as np
    import cv2
    
    windows = view_as_windows(data, window_shape=patch_shape, step=int(patch_shape[0]*(1-overlap)))
    windows = np.reshape(windows, (windows.shape[0]*windows.shape[1], windows.shape[2], windows.shape[3]))
    
    non_nan_idxs = []
    patches = []
    for idx, tile in enumerate(windows):
        if not np.isnan(tile).any():
            non_nan_idxs.append(idx)
        if new_shape:
            patches.append(cv2.resize(tile, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_LINEAR))
        else:
            patches.append(tile)
    if return_non_nan_idxs:
        return (np.array(patches), np.array(non_nan_idxs))
    else:
        return np.array(patches)
