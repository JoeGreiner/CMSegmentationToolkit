from csbdeep.models import CARE
import numpy as np


def predict_care(input_array, care_model=None, axes='ZYX'):
    print('image size =', input_array.shape)
    print('image axes =', axes)

    if care_model is None:
        care_model = CARE(config=None, name='confocal_pinhole_v3', basedir='models')

    dim_0, dim_1, dim_2 = input_array.shape

    n_tile_0 = np.max((1, dim_0 // 196))
    n_tile_1 = np.max((1, dim_1 // 196))
    n_tile_2 = np.max((1, dim_2 // 196))
    print(f"n_tile_0: {n_tile_0}, n_tile_1: {n_tile_1}, n_tile_2: {n_tile_2}")

    restored = care_model.predict(input_array, axes, n_tiles=(n_tile_0, n_tile_1, n_tile_2)) # paper evaluation was done with n_tiles=(1,4,4)
    return restored
