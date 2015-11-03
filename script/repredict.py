# coding: UTF8

from sklearn.pipeline import FeatureUnion
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import sklearn.linear_model
import img_to_pickle as i_p
import features as f
import classify
import pickle
import numpy as np
import pandas as pd
import datetime
import os


ROOT = os.path.abspath(os.path.dirname(__file__))
SUBMISSION_DIR = ROOT.replace("script", "tmp/submission")


def zero_one(x):

    return min(max(x, 0.), 1.)


def convert_testdata(test_gray_data, feature_rule=f.feature_transformer_rule):

    data_df = f.make_test_df(test_gray_data)
    fu = FeatureUnion(transformer_list=feature_rule)
    Std = preprocessing.StandardScaler()

    X_test = fu.fit_transform(data_df)
    #X_test = Std.fit_transform(X_test)

    return X_test


def reprediction():

    _, _, _, _, test_gray_data, _, _ = i_p.load_data()
    test_keys = test_gray_data.keys()

    test_df = f.make_test_df(test_gray_data)

    test_df = test_df.reset_index()
    test_df.columns = ["pngname", "input"]
    clf_dir = os.path.abspath(os.path.dirname(__file__)) +\
        "/../tmp/fit_instance/"
    savefile = clf_dir + "GB22015_10_04_07_30_36.pickle"
    fi = open(savefile, "r")
    clf = pickle.load(fi)
    fi.close()

    for i in xrange(len(test_keys)):

        test_img = test_df[(test_df["pngname"] == test_keys[i])]["input"].as_matrix()[0]

        imgname = test_keys[i]
        shape = test_img.shape

        test_img = {test_keys[i]: test_img}
        X_middle = convert_testdata(test_img, f.transformer_middle)
        middle_ratio = X_middle.mean()
        if middle_ratio >= 0.2:
            X_test = convert_testdata(test_img)
            output = clf.predict(X_test)
            output = np.asarray(output)
            zo = np.vectorize(zero_one)
            output = zo(output).reshape(shape)
        else:
            X_test = convert_testdata(test_img, f.transformer_gray)
            output = np.asarray(X_test)
            zo = np.vectorize(zero_one)
            output = zo(output).reshape(shape)
        tmp = []

        for row in xrange(len(output)):
            for column in xrange(len(output[row])):
                id_ = imgname + "_" + str(row + 1) + "_" + str(column + 1)
                value = output[row][column]

                pix = [id_, value]
                tmp.append(pix)

        if i == 0:
            predict_df = pd.DataFrame(tmp)

        else:
            tmp_df = pd.DataFrame(tmp)
            predict_df = pd.concat([predict_df, tmp_df])

    predict_df.columns = ["id", "value"]

    now = datetime.datetime.now()
    submission_path = SUBMISSION_DIR + "/submission_repredict" + now.strftime("%Y_%m_%d_%H_%M_%S") + ".csv"
    predict_df.to_csv(submission_path, header=True, index=False)


if __name__ == '__main__':

    reprediction()
