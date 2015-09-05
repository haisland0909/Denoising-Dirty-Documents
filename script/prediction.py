#coding: UTF8

from sklearn.pipeline import FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn import preprocessing
import sklearn.linear_model
import sklearn.ensemble
import img_to_pickle as i_p
import features as f
import numpy as np
import pandas as pd


clf_dict = {
    'LR': {
        "name": 'L2 Logistic Regression',
        "clf": sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, C=0.01),
    },
    'GB2': {
        "name": 'Gradient Boosting New',
        "clf": sklearn.ensemble.GradientBoostingClassifier(random_state=1, learning_rate=0.005, n_estimators=50, \
                subsample=1.0, max_features=1.0, min_samples_split=2, min_samples_leaf=2, max_depth=2)
    },
}


def convert_testdata(test_gray_data):

    data_df = f.make_test_df(test_gray_data) 
    fu = FeatureUnion(transformer_list=f.feature_transformer_rule)
    X_test = fu.fit_transform(data_df)
    
    return X_test


def convert_traindata(train_gray_data, labels):

    data_df = f.make_data_df(train_gray_data, labels) 
    fu = FeatureUnion(transformer_list=f.feature_transformer_rule)
    X_train = fu.fit_transform(data_df)
    y_train = np.concatenate(data_df["label"].apply(lambda x: x.flatten()))

    return X_train, y_train


def prediction(clf_name):

    print "****************classifier****************"
    print clf_dict[clf_name]["clf"]
    clf = clf_dict[clf_name]["clf"]
    
    _, _, _, train_gray_data, test_gray_data, _, labels = i_p.load_data()
    X_train, y_train = convert_traindata(train_gray_data, labels)
    X_test = convert_testdata(test_gray_data)

    print "... training"
    clf.fit(X, y)

    print "... test"
    output = clf.predict(X_test)



if __name__ == '__main__':

    clf_name = "LR"
    prediction(clf_name)


