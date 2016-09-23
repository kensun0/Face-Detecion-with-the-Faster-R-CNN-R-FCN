# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval
from fast_rcnn.config import cfg
import glob
import string
import h5py
from PIL import Image

class st_face(imdb):
    def __init__(self):
        imdb.__init__(self, 'st_face')
        self._classes = ('__background__', # always index 0
                         'face')
        self._image_index, self._bbx_index = self._load_image_set_index()
        self._roidb_handler = self.gt_roidb

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
	return self._image_index[i]


    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
	f = h5py.File('/data/st/data/wider/wider_face_train.mat')

	prefix='/data/st/data/wider/train/'

	path1=[]
	t=f['event_list']
	for k in range(0,t.shape[-1]):
    		a=f[t[0,k]]
    		str1 = ''.join(chr(i) for i in a[:])
    		path1.append(str1)
    
	image_index=[]    
	t=f['file_list']
	for k in range(0,t.shape[-1]):
    		a=f[t[0,k]]
    		str1=path1[k]
    		for j in range(0,a.shape[-1]):
        		b=f[a[0,j]]
        		str2 = ''.join(chr(i) for i in b[:])
        		str3 = prefix+str1+'/'+str2+'.jpg'
        		image_index.append(str3)
	#####################################################
	bbx_index=[]
	t=f['face_bbx_list']
	for k in range(0,t.shape[-1]):
    		a=f[t[0,k]]
    		str1=path1[k]
    		for j in range(0,a.shape[-1]):
        		b=f[a[0,j]]
        		bbx1 = b[:]
        		bbx_index.append(bbx1)
	return image_index, bbx_index

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
	#print index
	bbx1 = self._bbx_index[index]
	bbx1[2,:] = bbx1[0,:]+bbx1[2,:]
	bbx1[3,:] = bbx1[1,:]+bbx1[3,:]
	bbx2=bbx1.T
	#print bbx2
	num_objs=bbx2.shape[0]
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
	
	img = Image.open(self._image_index[index])
	img_width=img.size[0]
	img_height=img.size[1]
	for i in range(0,num_objs):
		x1=min(max(0,bbx2[i,0]-1),img_width-1)
		y1=min(max(0,bbx2[i,1]-1),img_height-1)
		x2=min(max(0,bbx2[i,2]-1),img_width-1)
		y2=min(max(0,bbx2[i,3]-1),img_height-1)
		if x1>x2 and y1>y2:
			print x1 , y1 , x2 , y2	
		boxes[i,:] = [x1, y1, x2, y2]
        	gt_classes[i] = 1
        	overlaps[i, 1] = 1.0
        	seg_areas[i] = (x2 - x1 + 1) * (y2 - y1 + 1)

  	overlaps = scipy.sparse.csr_matrix(overlaps) 
        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
  	#print self.image_index 
        gt_roidb = [self._load_pascal_annotation(index)
                    for index in range(0,len(self._image_index))]
   
        return gt_roidb





