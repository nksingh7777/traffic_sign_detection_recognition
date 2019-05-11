import cv2
import matplotlib.pyplot as plt
import os
from color_segmentation import *

add = '/media/nikesh/local/traffic_sign_recognition/GTSRB/detection/FullIJCNN2013'
f = os.listdir(add)
i = 0
while i < len(f):
	q = input()
	if(q == 'n'):
		i += 1
	elif q == 'p':
		i -= 1
	img = cv2.imread(add + '/' + f[i])
	blur_img = blur(img)
	mask = get_mask(blur_img, COLOR_RANGE_SEQ)
	morph_mask = remove_morpho_noise(mask)
	seg = get_seg(img, morph_mask, plot = False)
	bb = get_bounding_box(morph_mask)
	q = cv2.drawContours(img, bb, -1, (0,255,0), 2)
	cv2.imshow("test", q)
	cv2.waitKey(10)
