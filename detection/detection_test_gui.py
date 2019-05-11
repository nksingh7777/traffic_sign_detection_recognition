import cv2
import os
from color_segmentation import *

#address of the test data folder
ADDRESS = './GTSRB/detection/FullIJCNN2013/'
f = os.listdir(ADDRESS)

#Four windows for Four preprocessing step outcomes
cv2.namedWindow("test", flags = cv2.WINDOW_NORMAL)
cv2.namedWindow("test_seg", flags = cv2.WINDOW_NORMAL)
cv2.namedWindow("test_morph_mask", flags = cv2.WINDOW_NORMAL)
cv2.namedWindow("blur", flags = cv2.WINDOW_NORMAL)
cv2.namedWindow("red_conv", flags = cv2.WINDOW_NORMAL)

#nothing function used to create Trackbar , passed as callable
def nothing(x):
	pass

#test window shows final result image and trackbars for HSV settings
#for lower limits of HSV
cv2.createTrackbar("hue_l", "test", 0, 180, nothing)
cv2.createTrackbar("value_l", "test", 25, 255, nothing)
cv2.createTrackbar("saturation_l", "test", 25, 255, nothing)

#for upper limits of HSV
cv2.createTrackbar("hue_u", "test", 8, 180, nothing)
cv2.createTrackbar("value_u", "test", 255, 255, nothing)
cv2.createTrackbar("saturation_u", "test", 255, 255, nothing)

#HUE2 IN test
cv2.createTrackbar("hue_l2", "test", 160, 180, nothing)
cv2.createTrackbar("hue_u2", "test", 180, 180, nothing)

#Trackbars for Bounding Box Area and aspect ratio
cv2.createTrackbar("min_area", "test_seg", 800, 3000, nothing)
cv2.createTrackbar("Aspect_ratio", "test_seg", 40, 100, nothing)
cv2.createTrackbar("max_area", "test_seg", 20000, 50000, nothing)
cv2.createTrackbar("solidity", "test_seg", 75, 100, nothing)

#BLUR preprocessing
cv2.createTrackbar("filter2D", "blur", 1, 21, nothing)
cv2.createTrackbar("filter2D_iterations", "blur", 1, 10, nothing)

#MORPHOLOGICAL PREPROCESSING
cv2.createTrackbar("dilation_iterations", "test_morph_mask", 2, 10, nothing)
cv2.createTrackbar("erosion_iterations", "test_morph_mask", 1, 10, nothing)
cv2.createTrackbar("dilation_kernel_size", "test_morph_mask", 3, 10, nothing)
cv2.createTrackbar("erosion_kernel_size", "test_morph_mask", 3, 10, nothing)
cv2.createTrackbar("dilation_first", "test_morph_mask", 0, 1, nothing)

#RED COLOR BLURING
cv2.createTrackbar("filter2D", "red_conv", 3, 20, nothing)
cv2.createTrackbar("filter2D_iterations", "red_conv", 1, 10, nothing)

i = 0   #file index used to read image

while(1):
	hue_l = cv2.getTrackbarPos("hue_l", "test")
	hue_u = cv2.getTrackbarPos("hue_u", "test")
	hue_l2 = cv2.getTrackbarPos("hue_l2", "test")
	hue_u2 = cv2.getTrackbarPos("hue_u2", "test")
	
	value_l = cv2.getTrackbarPos("value_l", "test")
	value_u = cv2.getTrackbarPos("value_u", "test")
	
	saturation_l = cv2.getTrackbarPos("saturation_l", "test")
	saturation_u = cv2.getTrackbarPos("saturation_u", "test")
	
	max_area = cv2.getTrackbarPos("max_area", "test_seg")
	min_area = cv2.getTrackbarPos("min_area", "test_seg")
	min_asp = cv2.getTrackbarPos("Aspect_ratio", "test_seg")/100
	solidity = cv2.getTrackbarPos("solidity", "test_seg")/100
	
	filter2D_iterations = cv2.getTrackbarPos("filter2D_iterations", "blur")
	kernel_size = cv2.getTrackbarPos("filter2D", "blur")
	if kernel_size%2 == 0:
		kernel_size += 1

	dilation_iterations = cv2.getTrackbarPos("dilation_iterations", "test_morph_mask")
	erosion_iterations = cv2.getTrackbarPos("erosion_iterations", "test_morph_mask")
	dilation_first = cv2.getTrackbarPos("dilation_first", "test_morph_mask")
	erosion_kernel_size = cv2.getTrackbarPos("erosion_kernel_size", "test_morph_mask")
	erosion_kernel_size = erosion_kernel_size if erosion_kernel_size > 0 else 1
	dilation_kernel_size = cv2.getTrackbarPos("dilation_kernel_size", "test_morph_mask")
	dilation_kernel_size = dilation_kernel_size if dilation_kernel_size > 0 else 1

	red_conv_kernel_size = cv2.getTrackbarPos("filter2D", "red_conv")
	red_conv_kernel_size = red_conv_kernel_size if red_conv_kernel_size > 0 else 1

	red_conv_iterations = cv2.getTrackbarPos("filter2D_iterations", "red_conv")
	red_conv_iterations = red_conv_iterations if red_conv_iterations > 0 else 1
	red_conv_kernel = np.ones([red_conv_kernel_size,red_conv_kernel_size])/(red_conv_kernel_size*red_conv_kernel_size)

	# color range for segmentation
	color_range = [(np.array([hue_l, value_l, saturation_l]),
					np.array([hue_u, value_u, saturation_u])),
					(np.array([hue_l2, value_l, saturation_l]),
					np.array([hue_u2, value_u, saturation_u]))]

	#Preprocessing
	img = cv2.imread(ADDRESS + '/' + f[i])
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	red_conv_layer = cv2.filter2D(img[:,:,2], -1, red_conv_kernel)
	for _ in range(red_conv_iterations - 1):
		red_conv_layer = cv2.filter2D(red_conv_layer, -1, red_conv_kernel)
	red_conv_img = np.copy(img)
	red_conv_img[:,:,2] = red_conv_layer
	blur_img = blur(red_conv_img, kernel_size)
	for _ in range(filter2D_iterations - 1):
		blur_img = blur(blur_img, kernel_size)

	mask = get_mask(blur_img, color_range)
	# morph_mask = remove_morpho_noise(mask,dilation_iterations = dilation_iterations,
		 # erosion_iterations = erosion_iterations, dilation_first = dilation_first)
	if dilation_first:
		morph_mask = cv2.dilate(mask, np.ones((dilation_kernel_size,dilation_kernel_size))/(dilation_kernel_size*dilation_kernel_size), iterations = dilation_iterations)
		morph_mask = cv2.erode(morph_mask, np.ones((erosion_kernel_size,erosion_kernel_size))/(erosion_kernel_size*erosion_kernel_size), iterations = erosion_iterations)
	else:
		morph_mask = cv2.erode(mask, np.ones((erosion_kernel_size,erosion_kernel_size))/(erosion_kernel_size*erosion_kernel_size), iterations = erosion_iterations)
		morph_mask = cv2.dilate(morph_mask, np.ones((dilation_kernel_size,dilation_kernel_size))/(dilation_kernel_size*dilation_kernel_size), iterations = dilation_iterations)

	seg = get_seg(img, morph_mask, plot = False)
	bb, contours = get_bounding_box(morph_mask, min_asp = min_asp, 
		min_area = min_area, max_area = max_area)
	hull = [cv2.convexHull(cnt) for cnt in contours]
	solidity_arr = [cv2.contourArea(cnt)/cv2.contourArea(h) for cnt, h in zip(contours, hull)]
	bb = [b for b,s in zip(bb, solidity_arr) if s > solidity]
	# hull = [h for h,s in zip(hull, solidity_arr) if s > solidity]
	# contours = [cnt for cnt,s in zip(contours, solidity_arr) if s > solidity]


	q = cv2.drawContours(img, bb, -1, (0,255,0), 2)
	# q = cv2.drawContours(q, contours, -1, (0,255,0), 2)
	# q = cv2.drawContours(q, hull, -1, (0,0,255), 2)

	seg = cv2.drawContours(seg, bb, -1, (0,255,0), 2)
	seg = cv2.drawContours(seg, contours, -1, (255,0,0), 2)
	seg = cv2.drawContours(seg, hull, -1, (0,0,255), 2)

	red_conv_img = cv2.drawContours(red_conv_img, bb, -1, (0,255,0),1)
	img = cv2.putText(img, 
		"VALUE mean - {} std - {} f - {}".format(int(hsv_img[:,:,2].mean()), int(hsv_img[:,:,2].std()), f[i]),
			(50,50), 
			cv2.FONT_HERSHEY_SIMPLEX, 
			0.8,
			(0,255,0),
			1,
			cv2.LINE_AA)
	img = cv2.putText(img, 
		"SATURATION mean - {} std - {} max - {} min - {}".format(int(hsv_img[:,:,1].mean()), int(hsv_img[:,:,1].std()), int(hsv_img[:,:,1].max()), int(hsv_img[:,:,1].min())),
			(50,100), 
			cv2.FONT_HERSHEY_SIMPLEX, 
			0.8,
			(0,255,0),
			1,
			cv2.LINE_AA)
	cv2.imshow("red_conv", red_conv_img)
	cv2.imshow("test",img)
	cv2.imshow("test_seg", seg)	
	cv2.imshow("test_morph_mask", morph_mask)
	cv2.imshow("blur", blur_img)

	q = cv2.waitKey(200)
	if q == ord('q'):
		break
	elif q == ord('n'):
		i += 1
	elif q == ord('p'):
		i -= 1

cv2.destroyAllWindows()
