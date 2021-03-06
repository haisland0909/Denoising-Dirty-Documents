# coding: UTF8

import cv2
import os
import pickle
import glob
import numpy as np


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

    data = {}
    data_mono = {}

    pnglist = glob.glob(dir_)
    for png in pnglist:
        f_name = os.path.basename(png)
        pngname, ext = os.path.splitext(f_name)
        img = cv2.imread(png)

        data.update({pngname: img})

        img_mono = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        data_mono.update({pngname: img_mono})

    return data, data_mono


def pickle_dump(pickle_dir, name, data):

    savefile = pickle_dir + "/" + name + ".pickle"
    f = open(savefile, "w")
    pickle.dump(data, f)
    f.close()


def make_pickle(raw_dir, pickle_dir):

    train_dir = raw_dir + "/train/*"
    test_dir = raw_dir + "/test/*"
    clean_dir = raw_dir + "/train_cleaned/*"

    train_data, train_gray_data = open_img(train_dir)
    test_data, test_gray_data = open_img(test_dir)
    clean_data, clean_gray_data = open_img(clean_dir)

    print "... dump train_data"
    pickle_dump(pickle_dir, "train", train_data)
    pickle_dump(pickle_dir, "train_gray", train_gray_data)

    print "... dump test_data"
    pickle_dump(pickle_dir, "test", test_data)
    pickle_dump(pickle_dir, "test_gray", test_gray_data)

    print "... dump clean_data"
    pickle_dump(pickle_dir, "clean", clean_data)
    pickle_dump(pickle_dir, "clean_gray", clean_gray_data)
    labels = make_labels(clean_gray_data)
    pickle_dump(pickle_dir, "train_label", labels)

    return train_data, test_data, clean_data, train_gray_data, test_gray_data, clean_gray_data, labels


def pickle_up(pickle_dir, name):

    pickle_file = pickle_dir + "/" + name + ".pickle"
    print "... load " + name + "_data"
    f = open(pickle_file, "r")
    data = pickle.load(f)
    f.close()

    return data


def make_labels(src_data):

    labels = {}

    for key in src_data.keys():
        label = []
        for row in src_data[key]:
            label_c = [col / 255.0 for col in row]
            label.append(label_c)
        label = np.asarray(label)
        labels.update({key: label})

    return labels


def load_data():

    ###################################
    # kaggle directory :/tmp/kaggle_dirtydoc_data
    # rawdata directory : /tmp/kaggle_dirtydoc_data/raw_data
    # pickledump directory : /tmp/kaggle_dirtydoc_data/pickle_data
    ###################################

    root_dir = os.path.abspath(os.path.dirname(__file__)) +\
        "/../tmp/kaggle_dirtydoc_data"
    raw_dir = root_dir + "/raw_data"
    pickle_dir = root_dir + "/pickle_data"

    if not os.path.isdir(root_dir):
        os.mkdir(root_dir)
        os.mkdir(raw_dir)
        os.mkdir(pickle_dir)

        alert_rawdata(raw_dir)

    elif not os.path.isdir(raw_dir + "/train"):

        alert_rawdata(raw_dir)

    elif (not os.path.isdir(pickle_dir) or not
          os.path.isfile(pickle_dir + "/train.pickle")):

        if not os.path.isdir(pickle_dir):
            os.mkdir(pickle_dir)

        print "... make pickle_dump file"
        train_data, test_data, clean_data, train_gray_data, test_gray_data, clean_gray_data, labels = make_pickle(raw_dir, pickle_dir)

    else:
        print "... load datasets"
        train_data = pickle_up(pickle_dir, "train")
        test_data = pickle_up(pickle_dir, "test")
        clean_data = pickle_up(pickle_dir, "clean")

        train_gray_data = pickle_up(pickle_dir, "train_gray")
        test_gray_data = pickle_up(pickle_dir, "test_gray")
        clean_gray_data = pickle_up(pickle_dir, "clean_gray")

        labels = pickle_up(pickle_dir, "train_label")

    return train_data, test_data, clean_data, train_gray_data, test_gray_data, clean_gray_data, labels


if __name__ == '__main__':

    # all
    train_data, test_data, clean_data, train_gray_data, test_gray_data, clean_gray_data, labels = load_data()

    # part
    # {train, test, clean, train_gray, test_gray, clean_gray, train_label, labels}

    #pickle_dir = os.path.abspath(os.path.dirname(__file__)).replace("script", "") + "tmp/kaggle_dirtydoc_data/pickle_data"
    #labels = pickle_up(pickle_dir, "train_label")
