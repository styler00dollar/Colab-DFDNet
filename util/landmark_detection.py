import os
import numpy as np
import cv2
import math
from PIL import Image
from skimage import transform as trans
from skimage import io
import sys
sys.path.append('FaceLandmarkDetection')
import face_alignment
import dlib
from tqdm import tqdm


if __name__ == '__main__':
    images = os.listdir('test_dataset/images')
    SaveLandmarkPath = os.path.join('test_dataset/landmarks')
    FD = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda:0', flip_input=False)

    for i,img_path in enumerate(tqdm(images)):
        if img_path[-3:] != 'png':
            continue
        img_name = os.path.split(img_path)[-1]
        try:
            img = io.imread(os.path.join('test_dataset/images', img_path))
        except:
            continue
        try:
            PredsAll = FD.get_landmarks(img)
        except:
            print('\t################ Error in face detection, continue...')
            continue
        if PredsAll is None:
            print('\t################ No face, continue...')
            continue
        ins = 0
        if len(PredsAll)!=1:
            hights = []
            for l in PredsAll:
                hights.append(l[8,1] - l[19,1])
            ins = hights.index(max(hights))
        preds = PredsAll[ins]
        AddLength = np.sqrt(np.sum(np.power(preds[27][0:2]-preds[33][0:2],2)))
        SaveName = img_name + '.txt'
        np.savetxt(os.path.join(SaveLandmarkPath, SaveName),preds[:,0:2],fmt='%.3f')
