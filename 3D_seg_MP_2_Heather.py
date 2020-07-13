#! /usr/bin/env python

from multiprocessing import Pool
import signal
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



def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def worker(i): 
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
    f = os.path.join(mypath, i)
    im = AICSImage(f)
    IMG = im.data.astype(np.float32)
    ch1 = IMG[0,0,:,:,:].copy()
    ch2 = IMG[0,1,:,:,:].copy()
    ch2_img, ch2_bw, n_seeds, n = nuc_seg(ch2, [0.5, 500], 3, [[15,0.05]], 100, 30000, 15000)
    ch1_s, ch1_bw = cell_mask(ch1, [100], 15)
    seg = final_seg(ch1, ch1_bw, n_seeds)
    ch1_p = IMG[0,0,:,:,:].copy()
    scale = [8000]
    sigma = 1
    s3 = [[0.5, 0.85], [0.75,0.5], [1, 0.04], [1.5, 0.04]]
    DF, images = pex_seg(ch1_p, seg)
    name = str(i)
    DF["Cell"] = name
    return DF

if __name__ == '__main__':
    mypath = "/data/baldwinha/3D-segmentation/2019.11.7_pex-3d/"
    files = [f for f in listdir(mypath) if isfile(join(mypath,f))]
    nproc = int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))
    print("Running on %d CPUs" % nproc)
    tasks = files
    df_list = []
    p = Pool(nproc, init_worker)
    try:
        df_list = p.map(worker, tasks)
    except (KeyboardInterrupt, SystemExit):
        p.terminate()
        p.join()
        sys.exit(1)
    else:
        p.close()
        p.join()
        final_df = pd.concat(df_list)
        final_df.to_csv("/data/baldwinha/3D-segmentation/182_dataframe_1.csv")
        print(final_df)
        #print("\n".join("%d * %d = %d" % (a, a, b) for a, b in zip(tasks, results)))
    
