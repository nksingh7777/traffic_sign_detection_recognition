import sys
from color_segmentation import *

img_add = sys.argv[1]
file_add = sys.argv[2]

img = cv2.imread(img_add)

cv2.imwrite("{}/img.jpg".format(file_add), img)
blur_img = blur(img)
cv2.imwrite("{}/blur_img.jpg".format(file_add), blur_img)
mask = get_mask(blur_img, COLOR_RANGE_SEQ)
cv2.imwrite("{}/mask.jpg".format(file_add), mask)
morph_mask = remove_morpho_noise(mask)
cv2.imwrite("{}/morph_mask.jpg".format(file_add), morph_mask)
seg = get_seg(img, morph_mask)
bb = get_bounding_box(morph_mask)
q = cv2.drawContours(img, bb, -1, (0,255,0), 2)
cv2.imwrite("{}/bb.jpg".format(file_add), q)

f = open("{}/parameters".format(file_add),'w')
f.write("open_iter - {}\nrect_min_area - {}\n rect_min_aspect_ratio - {}".format(OPEN_ITERATIONS, RECT_MIN_AREA, RECT_MIN_ASPECT_RATIO))
f.close()
