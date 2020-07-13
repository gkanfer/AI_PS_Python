# -*- coding: utf-8 -*-

"""
Spyder Editor

This is a temporary script file.
"""

import tifffile as tfi
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance
#from PIL import fromarray
from numpy import asarray
from skimage import data,io
from skimage.filters import threshold_otsu, threshold_adaptive, threshold_local
from skimage.morphology import convex_hull_image
import matplotlib.pyplot as plt
import scipy as ndimage
from scipy.ndimage.morphology import binary_opening
from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage import data
from skimage.filters import rank
from skimage.util import img_as_ubyte
from skimage import data, util
from skimage.measure import label
from skimage.measure import perimeter
from skimage import measure
import os
import pandas as pd
from pandas import DataFrame
from scipy.ndimage import label, generate_binary_structure
from scipy.ndimage.morphology import binary_fill_holes
os.chdir('F:\Gil\AIMS\AIPS_code_in_python\example_image')
#I = io.imread('wella1A1_0004.tif',as_gray=True)
#I.show()

%matplotlib inline
 

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
       

pixels = tfi.imread('Composite.tif10.tif')

print('Data Type: %s' % pixels.dtype)
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
# convert from integers to floats
########normalisation
pixels = pixels.astype('float64')
# normalize to the range 0-1
pixels /= 65535.000
# confirm the normalization
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))

dapi=pixels[0,:,:]
plt.imshow(dapi)
GFP=pixels[1,:,:]
plt.imshow(GFP)

block_size = 21
nmask = threshold_local(dapi, block_size,offset=0.0004)
nmask2 = dapi > nmask
plt.imshow(nmask2)
nmask3 = binary_opening(nmask2, structure=np.ones((3,3))).astype(np.float64)
plt.imshow(nmask3)

nmask4 = binary_fill_holes(nmask3)
plt.imshow(nmask4)
perimeter(nmask4, neighbourhood=4)  
all_labels = measure.label(nmask4)
plt.imshow(all_labels)


label_objects, nb_labels = label(all_labels)
plt.imshow(label_objects)
sizes = np.bincount(label_objects.ravel())
#np.histogram(sizes, bins=10, range=None)
#plt.hist(sizes, bins='auto')
#plt.show()
mask_sizes = sizes > 250   #all the values larger than 250 is True
mask_sizes[0] = 0 #column 0, False will get the value of zero  
gsegg = mask_sizes[label_objects]
gsegg, nb_labels = label(gsegg) 
plt.imshow(gsegg)


###GFP extraction
Cyto = pixels[1,:,:]*40
#plt.hist(Cyto,"auto")
plt.imshow(Cyto,cmap='gray_r',  vmin=,  vmax=...)
block_size=51
inten=40
CH1_mask = threshold_local(Cyto, block_size,offset=0.1)
Ch1_mask_0 = Cyto > CH1_mask
plt.imshow(Ch1_mask_0,cmap='gray_r')
Ch1_mask_1 = binary_opening(Ch1_mask_0, structure=np.ones((3,3))).astype(np.float64) #should be a binarry image
plt.imshow(Ch1_mask_1,cmap='gray_r')  
#plt.hist(Ch1_mask_1,"auto")
quntile_num = np.quantile(Cyto, .1)
#open_2[] = Cyto > quntile_num
#Ch1_mask_2 = binary_opening(open_2 > 0.4)
Ch1_mask_2 = np.where( Cyto > quntile_num ,1,0) #should be bolian
plt.imshow(Ch1_mask_2,cmap='gray')
combine=Ch1_mask_1
combine[Ch1_mask_2 > Ch1_mask_1] = Ch1_mask_2[Ch1_mask_2 > Ch1_mask_1]
combine[gsegg > combine] = gsegg[gsegg > combine]
plt.imshow(combine,cmap='gray_r')
plt.show()
seg = watershed(Cyto, gsegg, mask=Ch1_mask_2, compactness=1)
seg = binary_fill_holes(seg)
plt.imshow(seg)
seg = watershed(seg, gsegg, mask=Ch1_mask_2, compactness=1)
seg1 = binary_fill_holes(seg)
seg2 , nb_labels = label(seg1)
plt.imshow(seg2)
plt.hist(seg,"auto")


