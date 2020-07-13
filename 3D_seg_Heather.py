import os
from os import listdir
from os.path import isfile, join
import numpy as np
from numpy.linalg import det
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import skimage
from skimage.morphology import remove_small_objects, watershed, dilation, ball, disk
from aicsimageio import AICSImage, omeTifWriter
from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_3d, image_smoothing_gaussian_slice_by_slice
from aicssegmentation.core.vessel import filament_2d_wrapper, filament_3d_wrapper
from aicssegmentation.core.utils import get_middle_frame, hole_filling, get_3dseed_from_mid_frame
from aicssegmentation.core.seg_dot import dot_3d_wrapper, dot_2d_slice_by_slice_wrapper
from skimage.filters import rank, threshold_otsu
from skimage.measure import label, marching_cubes_lewiner, regionprops
from skimage.morphology import remove_small_objects, watershed, dilation, erosion, ball
from skimage.feature import peak_local_max
import pandas as pd

def plot_1(image, image2, cmap1, cmap2):
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10,20))
    ax0.imshow(image, interpolation='nearest', cmap=cmap1)
    ax1.imshow(image2, cmap=cmap2)
    fig.tight_layout()
def plot_2(image, image2, cmap1, cmap2):
    z_size, x_size, y_size = image.shape
    nrows = np.int(np.ceil(np.sqrt(z_size)))
    ncols = np.int(z_size//nrows+1)
    fig, axes = plt.subplots(nrows, ncols*2, figsize=(3*ncols, 1.5*nrows))
    for z in range(z_size):
        i = z // ncols
        j = z % ncols * 2
        axes[i,j].imshow(image[z,...], interpolation="nearest", cmap=cmap1,)
        axes[i, j+1].imshow(image2[z,...], cmap=cmap2)
        axes[i,j].set_xticks([])
        axes[i,j].set_yticks([])
        axes[i,j+1].set_xticks([])
        axes[i,j+1].set_yticks([])
    for ax in axes.ravel():
        if not(len(ax.images)):
            fig.delaxes(ax)
    fig.tight_layout()
def nuc_seg(image, scale, sigma, f2, hol_min, hol_max, minA):
    img = intensity_normalization(image, scale)
    img_s = image_smoothing_gaussian_3d(img, sigma)
    bw = filament_2d_wrapper(img_s, f2)
    bw_2 = hole_filling(bw, hol_min, hol_max)
    seg = remove_small_objects(bw_2>0, min_size=minA, connectivity=1, in_place=False)
    seg2, n = label(seg, return_num=True)
    return (img_s, bw_2, seg2, n)
def cell_mask(image, scale, sigma):
    img1 = intensity_normalization(image, scale)
    img_s = image_smoothing_gaussian_3d(img1, sigma)
    thr = threshold_otsu(img_s)
    bw = img_s > thr*0.8
    return (img_s, bw)
def final_seg(image, img_mask, seed):
    seg = watershed(image, seed, mask=img_mask, compactness=1)
    return seg
def pex_seg(image, img_seg):
    slices = []
    Vol = []
    pex_seg_img = []
    pex_seg_n = []
    slices = ndi.find_objects(img_seg)
    img = intensity_normalization(image, scale)
    img = image_smoothing_gaussian_3d(img, sigma)
    for i, c in enumerate(slices):
        cell_V = img_seg[c]
        vt, f, n, val = marching_cubes_lewiner(cell_V)
        Tvol = 1/6*det(vt[f])
        vol = abs(sum(Tvol))
        Vol.append(vol)
        cell = image[c]
        cell_s = img[c]
        bw = dot_3d_wrapper(cell_s, s3)
        mask = remove_small_objects(bw>0, min_size=2, connectivity=1, in_place=False)
        seed = dilation(peak_local_max(cell, labels=label(mask), min_distance=2, 
                                       indices=False), selem=ball(1))
        ws_map = -1*ndi.distance_transform_edt(bw)
        seg = watershed(ws_map, label(seed), mask=mask, watershed_line=True)
        regions = regionprops(seg)
        n = len(regions)
        n = max(0, n)
        pex_seg_img.append(seg)
        pex_seg_n.append(n)
    df = pd.DataFrame({"Slices":slices, "Cell Vol":Vol, "Pex count":pex_seg_n})
    return df, pex_seg_img 

scale = [8000]
sigma = 1
s3 = [[0.5, 0.85], [0.75,0.5], [1, 0.04], [1.5, 0.04]]

mypath = "2019.11.7_pex-3d/"
files = [f for f in listdir(mypath) if isfile(join(mypath,f))]
df_list = []
for i, fl in enumerate(files):
    f = os.path.join(mypath, fl)
    im = AICSImage(f)
    IMG = im.data.astype(np.float32)
    ch1 = IMG[0,0,:,:,:].copy()
    ch2 = IMG[0,1,:,:,:].copy()
    ch2_img, ch2_bw, n_seeds, n = nuc_seg(ch2, [0.5, 500], 3, [[15,0.05]], 100, 30000, 15000)
    ch1_s, ch1_bw = cell_mask(ch1, [100], 15)
    seg = final_seg(ch1, ch1_bw, n_seeds)
    ch1_p = IMG[0,0,:,:,:].copy()
    DF, images = pex_seg(ch1_p, seg)
    name = files[i]
    DF["Cell"] = name
final_df = pd.concat(df_list)

final_df.to_csv("182_dataframe_1.csv")