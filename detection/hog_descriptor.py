import cv2
import numpy as np
import os
from skimage.feature import hog

ADDRESS = '/media/nikesh/local/projects/traffic_sign_recognition/GTSRB/detection/current_classes'

cv2.namedWindow("HOG", cv2.WINDOW_NORMAL)
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
quit = False
f = os.listdir(ADDRESS)
for c in f:
	if quit:
		break
	f_img = os.listdir(ADDRESS + '/' + c)
	i = 0
	while True:
		img = cv2.imread(ADDRESS + '/' + c + '/' + f_img[i])
		if np.prod(img.shape[:2]) < 900:
			i += 1
			continue
		list_text = img.shape[:2]
		img = cv2.resize(img, (40,40), interpolation = cv2.INTER_LINEAR)
		cv2.putText(img,
			"{}".format(list_text),
			(0,15),
			cv2.FONT_HERSHEY_SIMPLEX, 
			0.4,
			(0,255,0),
			1,
			cv2.LINE_AA)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		fd, hog_img = hog(gray,block_norm='L2' ,visualise = True)
		cv2.imshow("HOG", hog_img)
		cv2.imshow("img", img)

		q = cv2.waitKey(10)
		if q == ord('q'):
			quit = True
			break
		elif q == ord('n'):
			if i < len(f_img) - 1:
				i += 1
			else:
				break
		elif q == ord('p'):
			i -= 1

cv2.destroyAllWindows()
