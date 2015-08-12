#codimg: UTF8

import cv2
import os
import urllib2
import pickle
import glob

def alert_rawdata(raw_dir):

    print "no datasets!"
    quit()

    """
    print "... Download datasets"
    train = raw_dir + "/" + os.path.basename("https://www.kaggle.com/c/denoising-dirty-documents/download/train.zip")
    urllib.urlretrieve("https://www.kaggle.com/c/denoising-dirty-documents/download/train.zip", train)
    test = raw_dir + "/" + os.path.basename("https://www.kaggle.com/c/denoising-dirty-documents/download/test.zip")
    urllib.urlretrieve("https://www.kaggle.com/c/denoising-dirty-documents/download/test.zip", test)
    clean = raw_dir + "/" + os.path.basename("https://www.kaggle.com/c/denoising-dirty-documents/download/train_cleaned.zip")
    urllib.urlretrieve("https://www.kaggle.com/c/denoising-dirty-documents/download/train_cleaned.zip", clean)
    """


def open_img(dir_):

    data = []

    pnglist = glob.glob(dir_)
    for png in pnglist:
        img = cv2.imread(png)
        data.append(img)

    return data

def pickle_dump(pickle_dir, name, data):
    
    savefile = pickle_dir + "/" + name + ".pickle"
    f = open(savefile, "w")
    pickle.dump(data, f)
    f.close()

def make_pickle(raw_dir, pickle_dir):
    
    train_dir = raw_dir + "/train/*"
    test_dir = raw_dir + "/train/*"
    clean_dir = raw_dir + "/train_cleaned/*"

    train_data = open_img(train_dir)
    test_data = open_img(test_dir)
    clean_data = open_img(clean_dir)

    print "... dump train_data"
    pickle_dump(pickle_dir, "train", train_data)
    print "... dump test_data"
    pickle_dump(pickle_dir, "test", test_data)
    print "... dump clean_data"
    pickle_dump(pickle_dir, "clean", clean_data)

    return train_data, test_data, clean_data


def pickle_up(pickle_dir):
    
    train_pickle = pickle_dir + "/train.pickle"
    test_pickle = pickle_dir + "/test.pickle"
    clean_pickle = pickle_dir + "/clean.pickle"

    print "... load train_data"
    f = open(train_pickle, "r")
    train_data = pickle.load(f)
    f.close()

    print "... load test_data"
    f = open(train_pickle, "r")
    test_data = pickle.load(f)
    f.close()
    
    print "... load clean_data"
    f = open(train_pickle, "r")
    clean_data = pickle.load(f)
    f.close()

    return train_data, test_data, clean_data

def load_data():
    
    ###################################
    #kaggle directory :/tmp/kaggle_dirtydoc_data
    #rawdata directory : /tmp/kaggle_dirtydoc_data/raw_data
    #pickledump directory : /tmp/kaggle_dirtydoc_data/pickle_data
    ###################################

    root_dir = "/tmp/kaggle_dirtydoc_data"
    raw_dir = root_dir + "/raw_data"
    pickle_dir = root_dir + "/pickle_data"

    if not os.path.isdir(root_dir):
        os.mkdir(root_dir)
        os.mkdir(raw_dir)
        os.mkdir(pickle_dir)
        
        alert_rawdata(raw_dir)

    elif not os.path.isdir(raw_dir + "/train"):
        
        alert_rawdata(raw_dir)

    elif not os.path.isdir(pickle_dir) or not os.path.isfile(pickle_dir+"/train.pickle"):

        if not os.path.isdir(pickle_dir):
            os.mkdir(pickle_dir)

        print "... make pickle_dump file"
        train_data, test_data, clean_data = make_pickle(raw_dir, pickle_dir)
        
    else:
        print "... load datasets"
        train_data, test_data, clean_data = pickle_up(pickle_dir)

    return train_data, test_data, clean_data

if __name__ == '__main__':

    train_data, test_data, clean_data = load_data()
    print "len(train)"
    print len(train_data)
    print "len(test)"
    print len(test_data)
    print "len(clean)"
    print len(clean_data)

    print "---------------------"
    print train_data[0]
    print type(train_data[0])
    print train_data[0].shape

