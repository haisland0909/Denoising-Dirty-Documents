#coding: UTF8

import img_to_pickle as i_p
import os
import cv2
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn import preprocessing
import sklearn.linear_model
import sklearn.ensemble

import img_to_pickle as i_p
import features as f

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


def make_dividedata():

    data = i_p.pickle_up(PICKLE_DIR, "train_gray")
    divide(data, "train")

    data = i_p.pickle_up(PICKLE_DIR, "test_gray")
    divide(data, "test")

    data = i_p.pickle_up(PICKLE_DIR, "clean_gray")
    divide(data, "train_label")



def make_checkdata(mode="df"):
    
    fu = FeatureUnion(transformer_list=f.feature_transformer_rule)
    Std = preprocessing.StandardScaler()

    _, _, _, train_gray_data, test_gray_data, _, labels = i_p.load_data()
    train_keys = train_gray_data.keys()[:2]
   
    train_inputs = {}
    train_labels = {}
    for i in xrange(len(train_keys)):
        input_ = train_gray_data[train_keys[i]]
        label = labels[train_keys[i]]

        train_inputs.update({train_keys[i]:input_})
        train_labels.update({train_keys[i]:label})
 
    test_keys = test_gray_data.keys()[:2]
    test_inputs = {}
    for i in xrange(len(test_keys)):
        input_ = test_gray_data[test_keys[i]]
        test_inputs.update({test_keys[i]:input_})
        
    train_df = f.make_data_df(train_inputs, train_labels)
    test_df = f.make_test_df(test_inputs) 
    

    if mode == "df":

        train_df = train_df.reset_index()
        test_df = test_df.reset_index()
        
        train_df.columns = ["pngname", "input", "label"]
        test_df.columns = ["pngname", "input"]

        return train_df, train_keys, test_df, test_keys


    elif mode == "feature":

        X_train = fu.fit_transform(train_df)
        X_train = Std.fit_transform(X_train)
        y_train = np.concatenate(train_df["label"].apply(lambda x: x.flatten()))
        
        
        
        X_test = fu.fit_transform(test_df)
        X_test = Std.fit_transform(X_test)    
        
        return X_train, y_train, X_test


if __name__ == "__main__":

    #make_dividedata()
    make_checkdata()


