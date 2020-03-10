import cv2
import numpy as np
import os
import sys
sys.path.append('../insightface/RetinaFace')
from retinaface import RetinaFace

thresh = 0.8
scales = [1024, 1980]

count = 1

gpuid = -1 # CPU  -> 0 GPU
detector = RetinaFace('../insightface/RetinaFace/model/retinaface-R50/R50', 0, gpuid, 'net3')

path = './train/'
path_new = './train_crop/'
list_dir = os.listdir(path)
for dirs in list_dir:
    path_dirs = os.path.join(path,dirs)
    path_dirs_new = os.path.join(path_new,dirs)
    
    if not os.path.exists(path_dirs_new):
        os.mkdir(path_dirs_new)
    list_images = os.listdir(path_dirs)
    try:
        for idx, image in enumerate(list_images):
            scales = [1024, 1980]
            path_images = os.path.join(path_dirs,image)
            img = cv2.imread(path_images)
            im_shape = img.shape
            target_size = scales[0]
            max_size = scales[1]
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])
            #im_scale = 1.0
            #if im_size_min>target_size or im_size_max>max_size:
            im_scale = float(target_size) / float(im_size_min)
            # prevent bigger axis from being more than max_size:
            if np.round(im_scale * im_size_max) > max_size:
                im_scale = float(max_size) / float(im_size_max)

            scales = [im_scale]
            flip = False

            faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)

            if faces is not None:
                print('find', faces.shape[0], 'faces')
            for i in range(faces.shape[0]):
                #print('score', faces[i][4])
                box = faces[i].astype(np.int)
                #color = (255,0,0)
                color = (0,0,255)
                # print(box)
                tmp_img = img[box[1]:box[3], box[0]:box[2]]

                tmp_img = cv2.resize(tmp_img,(112,112))
                filename = "{}/{}.jpg".format(path_dirs_new,idx)
                print('writing', filename)
                cv2.imwrite(filename, tmp_img)
    except:
        pass