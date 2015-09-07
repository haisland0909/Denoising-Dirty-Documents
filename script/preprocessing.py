#coding: UTF8

import img_to_pickle as itp
import os
import cv2
import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
PICKLE_DIR = ROOT.replace("script", "tmp/kaggle_dirtydoc_data/pickle_data") 
SAVE_DIR = ROOT.replace("script", "tmp/kaggle_dirtydoc_data/divide_image")


def padding(img):

    pad1 = 0
    row = img.shape[0]
    if row%10 != 0:
        pad1 = 10 - row%10
        row = row + pad1
    
    pad2 = 0
    col = img.shape[1] 
    if col%10 != 0:
        pad2 = 10 - row%10
        col = col + pad2
    
    cols = [0]*col
    image = [cols] * row
    image = np.asarray(image)
    image = image.astype(np.uint8)
    
    image[pad1/2:(pad1/2)+img.shape[0], pad2/2:(pad2/2)+img.shape[1]] = img
    
    return image

def divide(data, target):
    
    save_dir = SAVE_DIR + "/" + target + "/"
    numbers = data.keys()
    
    for i in xrange(len(numbers)):
        
        img = data[numbers[i]]
        if img.shape[0] == 420:
            
            for ys in xrange(38):
                Y = ys * 10
                for xs in xrange(50):
                    X = xs * 10
                    divide_data = img[Y:Y+50, X:X+50]
                    savefile = save_dir + numbers[i] + "_" + str(xs) + "_" + str(ys) + ".jpg"
                    cv2.imwrite(savefile, divide_data)
        
        elif img.shape[0] == 258:
            
            img = padding(img)
            for ys in xrange(22):
                Y = ys * 10
                for xs in xrange(50):
                    X = xs * 10
                    divide_data = img[Y:Y+50, X:X+50]
                    savefile = save_dir + numbers[i] + "_" + str(xs) + "_" + str(ys) + ".jpg"
                    cv2.imwrite(savefile, divide_data)

        else:
            print "error"
            quit()

if __name__ == "__main__":

    data = itp.pickle_up(PICKLE_DIR, "train_gray")
    divide(data, "train")

    data = itp.pickle_up(PICKLE_DIR, "test_gray")
    divide(data, "test")

    data = itp.pickle_up(PICKLE_DIR, "clean_gray")
    divide(data, "train_label")



