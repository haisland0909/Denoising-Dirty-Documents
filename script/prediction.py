# coding: UTF8

from sklearn.pipeline import FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import sklearn.linear_model
import img_to_pickle as i_p
import features as f
import classify
import preprocessing as pre
import pickle
import numpy as np
import pandas as pd
import datetime
import os


ROOT = os.path.abspath(os.path.dirname(__file__))
SUBMISSION_DIR = ROOT.replace("script", "tmp/submission")


clf_dict = {
    'LR': {
        "name": 'L2 Logistic Regression',
        "clf": sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, C=0.01),
    },
    'GB2': {
        "name": 'Gradient Boosting New',
        "clf": GradientBoostingRegressor(random_state=1, learning_rate=0.05,
                                         n_estimators=3000, subsample=0.8,
                                         max_features=0.3, min_samples_split=2,
                                         min_samples_leaf=1, max_depth=7)
    },
    "RF": {
        "name": "RandomForest",
        "clf": RandomForestRegressor(max_depth=7, max_features=0.4,
                                     min_samples_leaf=10, min_samples_split=2,
                                     n_jobs=-1, n_estimators=1000)
    },
    'SGDR': {
        "name": 'SGD Regression',
        "clf": sklearn.linear_model.SGDRegressor(penalty='l2'),
    }
}


def zero_one(x):

    return min(max(x, 0.), 1.)


def convert_testdata(test_gray_data):

    data_df = f.make_test_df(test_gray_data)
    fu = FeatureUnion(transformer_list=f.feature_transformer_rule)
    Std = preprocessing.StandardScaler()

    X_test = fu.fit_transform(data_df)
    #X_test = Std.fit_transform(X_test)

    return X_test


def convert_traindata(train_gray_data, labels):

    data_df = f.make_data_df(train_gray_data, labels)
    fu = FeatureUnion(transformer_list=f.feature_transformer_rule)
    Std = preprocessing.StandardScaler()

    X_train = fu.fit_transform(data_df)
    y_train = np.concatenate(data_df["label"].apply(lambda x: x.flatten()))

    X_train = Std.fit_transform(X_train)

    return X_train, y_train


def prediction(clf_name):

    print "****************classifier****************"
    print clf_dict[clf_name]["clf"]
    clf = clf_dict[clf_name]["clf"]

    _, _, _, train_gray_data, test_gray_data, _, labels = i_p.load_data()
    train_keys = train_gray_data.keys()
    test_keys = test_gray_data.keys()

    train_df = f.make_data_df(train_gray_data, labels)
    test_df = f.make_test_df(test_gray_data)

    train_df = train_df.reset_index()
    test_df = test_df.reset_index()

    train_df.columns = ["pngname", "input", "label"]
    test_df.columns = ["pngname", "input"]

    # operation check
    if clf_name == "SGDB":
        # train_df, train_keys, test_df, test_keys  = pre.make_checkdata(mode="df")
        # train_df, train_keys, _, _  = pre.make_checkdata(mode="df")

        for i in xrange(len(train_keys)):

            train_X, train_y = classify.set_traindata(train_df, train_keys[i])
            clf.partial_fit(train_X, train_y)

    else:

        # operation check
        # train_df, train_keys, _, _  = pre.make_checkdata(mode="df")
        fu = FeatureUnion(transformer_list=f.feature_transformer_rule)
        train_X = fu.fit_transform(train_df)
        train_y = np.concatenate(train_df["label"].apply(lambda x: x.flatten()))
        train_X, train_y = classify.downsampling_data(train_X, train_y, 0.2)

        clf.fit(train_X, train_y)
    clf_dir = os.path.abspath(os.path.dirname(__file__)) +\
        "/../tmp/fit_instance/"
    now = datetime.datetime.now()
    savefile = clf_dir + clf_name + now.strftime("%Y_%m_%d_%H_%M_%S") + ".pickle"
    fi = open(savefile, "w")
    pickle.dump(clf, fi)
    fi.close()

    for i in xrange(len(test_keys)):

        test_img = test_df[(test_df["pngname"] == test_keys[i])]["input"].as_matrix()[0]

        imgname = test_keys[i]
        shape = test_img.shape

        test_img = {test_keys[i]: test_img}
        X_test = convert_testdata(test_img)
        output = clf.predict(X_test)
        output = np.asarray(output)
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
    submission_path = SUBMISSION_DIR + "/submission_" + now.strftime("%Y_%m_%d_%H_%M_%S") + ".csv"
    predict_df.to_csv(submission_path, header=True, index=False)


if __name__ == '__main__':

    clf_name = "RF"
    prediction(clf_name)
