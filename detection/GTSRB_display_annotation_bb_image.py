import cv2
import os
import pandas as pd
import pickle
import math

f = open("interested_classes.txt", 'rb')
interested_classes = set(pickle.load(f))
f.close()
IMAGES_ADDR = './GTSRB/GTSRB_train/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/'
IMAGES_FILES = os.listdir(IMAGES_ADDR)

cv2.namedWindow('training_images', cv2.WINDOW_NORMAL)
key = None
#interested_classes = list()
for f in IMAGES_FILES:
    if f not in interested_classes:
        continue
    dir_ = "./images/"+f
    os.mkdir(dir_)
    images_addr = os.listdir(IMAGES_ADDR + f)
    annotation_file = [_ for _ in images_addr if '.csv' in _][0]
    images_addr.remove(annotation_file)
    annot_df = pd.read_csv(os.path.join(IMAGES_ADDR+f,annotation_file), sep = ';')
    if annot_df is None:
        print(annotation_file)
    for img_addr in images_addr:
        addr = os.path.join(IMAGES_ADDR+f, img_addr)
        img = cv2.imread(addr)
        img_row = annot_df.loc[annot_df['Filename']==img_addr]
        try:
            pt1 = img_row['Roi.X1'].values[0], img_row['Roi.Y1'].values[0]
            pt2 = img_row['Roi.X2'].values[0], img_row['Roi.Y2'].values[0]
            class_id = img_row['ClassId']
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img[pt1[0]:pt2[0],pt1[1]+1:pt2[1]+1]
            cv2.imwrite(dir_+'/'+img_addr, img)
            cv2.imshow('training_images', img)
        except Exception as e:
            print(e)
            print(img_addr)
        key = cv2.waitKey(10)
        if key == ord('q') or key == ord('i') or key == ord('n'):
            break
        
    if key == ord('q'):
    	break
#    elif key == ord('i'):
#        interested_classes.append(f)
#f2 = open("safe.txt", 'w')
#for sign in interested_classes:
#    f2.write(sign + ';')
#f2.close()
#print(interested_classes)
#f = open("interested_classes.txt", 'wb')
#pickle.dump(interested_classes, f)
#f.close()

