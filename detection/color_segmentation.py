import cv2
import matplotlib.pyplot as plt
import numpy as np

#TODOS
	# - resizing is an import step as blur and morphological kernels depends on it heavily
	# - try new kernels for morphologyEx
	# - external contours
	# - choose between blur algo - median blur seems good as inside the signal max of red, blue, and yellow is present thus increases these colors inside the signs.
 
#In Hsv Color space
COLOR_RANGE_SEQ = [	(np.asarray(t[0]), np.asarray(t[1])) for t in [
					([160 ,40,10],[190,255,255]), 		# red
					([100 ,40, 10], [110, 255, 200]), 	# blue
					([20,200,80],[28,255,255]) 			#yellow
					# ,([46,180,40], [70,255,255])	 		#green
					]]

OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#ELLIPTICAL KERNEL AS IT FITS FOR CIRCULAR SIGNS RED BOUNDARY.
OPEN_ITERATIONS = 1
RECT_MIN_AREA = 250
RECT_MAX_AREA = 9000
RECT_MIN_ASPECT_RATIO = 0.4
#BLUR_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))/30
BLUR_KERNEL_SIZE = 5
DILATION_ITERATIONS = 1
EROSION_ITERATIONS = 1
DILATION_FIRST = 1

def blur(img, kernel_size = BLUR_KERNEL_SIZE):
	'''
	blurs the image for color segmentation
	'''
#	return cv2.filter2D(img, -1, kernel)
#	return cv2.medianBlur(cv2.filter2D(img, -1, kernel), 5)
	return cv2.filter2D(img, -1, np.ones((kernel_size, kernel_size))/(kernel_size*kernel_size))
	# return cv2.medianBlur(img, kernel_size)

def get_mask(img ,color_range_seq = COLOR_RANGE_SEQ, hsv = False):
	'''
	-----------------IMAGE AND COLORS ARE IN HSV COLORSPACE------------------
	takes a sequence of colors range (lower_color_ndarray, upper_color_ndarray)
	and gives a common mask that filters the given colors from image
	'''
	if not hsv:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	color = color_range_seq[0]
	mask = cv2.inRange(img, color[0], color[1])

	for color in color_range_seq[1:]:
		mask = cv2.bitwise_or(mask, cv2.inRange(img, color[0], color[1]))

	return mask

def get_seg(img, mask, plot = True):
	'''
	------------------img is in BGR colorspace amd seg is also in BGR-------------
	plots the resultant segment from (img,mask) and returns it. 
	'''
	seg = cv2.bitwise_and(img, img, mask = mask)
	if plot:
		plt.imshow(cv2.cvtColor(seg, cv2.COLOR_BGR2RGB))
		plt.show()
	return seg

def remove_morpho_noise(mask, kernel = OPEN_KERNEL, dilation_iterations = DILATION_ITERATIONS, erosion_iterations = EROSION_ITERATIONS, dilation_first = DILATION_FIRST):
	'''
	performs morphological opening i.e erosion + dialation on the mask and removes noise
	'''
	# return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = iterations)
	if dilation_first:
		mask = cv2.dilate(mask, kernel , iterations = dilation_iterations)
		mask = cv2.erode(mask, kernel, iterations = erosion_iterations)
	else:
		mask = cv2.erode(mask, kernel, iterations = erosion_iterations)
		mask = cv2.dilate(mask, kernel , iterations = dilation_iterations)

	return mask
def valid_rect(rect, min_asp = RECT_MIN_ASPECT_RATIO, min_area = RECT_MIN_AREA, max_area = RECT_MAX_AREA):
	'''
	condition for a bounding box to be a valid ROI
	rect is the rect object from opencv and described as - 
	( top-left corner(x,y), (width, height), angle of rotation )
	'''	
	p = rect[1]
	area = np.multiply(*p)
	if  area > max_area or area < min_area:
		return False
	
	asp_rat = min(p) / max(p)
	if (asp_rat < min_asp) :
        	return False
	return True

def get_bounding_box(mask, min_asp = RECT_MIN_ASPECT_RATIO, min_area = RECT_MIN_AREA, max_area = RECT_MAX_AREA):
	'''
	return Bounding Box based on contours from mask and valid_rect criteria
	(aspect_ratio > RECT_MIN_ASP and area > RECT_MIN_AREA)
	'''
	_,contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	rect = [cv2.minAreaRect(cnt) for cnt in contours]
	bound_box = [np.int0(cv2.boxPoints(r)) for r in rect if valid_rect(r, min_asp, min_area, max_area)]
	contours = [cnt for cnt, rect in zip(contours, rect) if valid_rect(rect, min_asp, min_area, max_area)]
	return bound_box, contours

