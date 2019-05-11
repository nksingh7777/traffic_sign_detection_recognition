import cv2
import numpy as np
import os

ADDRESS = './GTSRB/detection/FullIJCNN2013'
f = os.listdir(ADDRESS)
def nothing(x):
	pass

cv2.namedWindow("canny_edge_detection", flags = cv2.WINDOW_NORMAL)
cv2.namedWindow("filter2D", flags = cv2.WINDOW_NORMAL)

cv2.createTrackbar("filter2D_kernel_size", "filter2D", 3, 15, nothing)
cv2.createTrackbar("filter2D_iterations", "filter2D", 1, 15, nothing)
cv2.createTrackbar("min_threshold", "canny_edge_detection", 20, 1000, nothing)
cv2.createTrackbar("max - min", "canny_edge_detection", 100, 1000, nothing)
cv2.createTrackbar("red_only", "canny_edge_detection", 0, 1, nothing)

i = 0

while(True):
	filter2D_kernel_size = cv2.getTrackbarPos("filter2D_kernel_size", "filter2D")
	filter2D_kernel_size = filter2D_kernel_size if filter2D_kernel_size > 0 else 1
	filter2D_iterations = cv2.getTrackbarPos("filter2D_iterations", "filter2D")
	min_threshold = cv2.getTrackbarPos("min_threshold", "canny_edge_detection")
	diff = cv2.getTrackbarPos("max - min", "canny_edge_detection")
	red_only = cv2.getTrackbarPos("red_only", "canny_edge_detection")
	
	img = cv2.imread(ADDRESS + '/' + f[i])

	if red_only:
		edge_img = img[:,:,2]
	else:
		edge_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	kernel = np.ones((filter2D_kernel_size, filter2D_kernel_size))/ (filter2D_kernel_size*filter2D_kernel_size)
	
	for _ in range(filter2D_iterations - 1):
		edge_img = cv2.filter2D(edge_img, -1, kernel)
	
	edge = cv2.Canny(edge_img, min_threshold, min_threshold + diff)
	cv2.imshow("filter2D", edge_img)
	cv2.imshow("canny_edge_detection",edge)
	q = cv2.waitKey(100)
	if q == ord('q'):
		break
	elif q == ord('n'):
		i += 1
	elif q == ord('p'):
		i -= 1

cv2.destroyAllWindows()
