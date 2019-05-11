from hog_utility import *

resize = (40,40)
hog_dir = "./HOG"
img_dir = "./images"
img_files = os.listdir(img_dir)

for f in img_files:
    print(f)
    hog_file = os.path.join(hog_dir, f)
    os.mkdir(hog_file)
    img_file = os.path.join(img_dir, f)
    images = os.listdir(img_file)
    for img_name in images:
        try:
            img = cv2.imread(os.path.join(img_file, img_name))
            if np.prod(img.shape) < 850:
                continue
            img = cv2.resize(img, resize , interpolation = cv2.INTER_LINEAR)
            hog_buffer = open(os.path.join(hog_file, img_name.split('.')[0]+'.pkl'), 'wb')
            pickle.dump(cv2_hog.compute(img).ravel(), hog_buffer)
            hog_buffer.close()
        except Exception as e:
            print(e)
            print(f,img_name)
