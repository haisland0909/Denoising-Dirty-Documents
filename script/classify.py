'''
Created on 2015/08/28

@author: haisland0909
'''
from sklearn.pipeline import FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
import sklearn.linear_model
import sklearn.ensemble
import img_to_pickle as i_p
import features as f
import numpy as np
import pandas as pd

clf_dict = {
    'LR': {
        "name": 'L2 Logistic Regression',
        "clf": sklearn.linear_model.LogisticRegression(penalty='l2', dual=False),
        "paramteters": {'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]}
    },
    'GB2': {
        "name": 'Gradient Boosting New',
        "clf": sklearn.ensemble.GradientBoostingClassifier(random_state=1),
        "paramteters": {
            'learning_rate': [0.005, 0.01, 0.1],
            'n_estimators': [50, 250, 500],
            'subsample': [1.0, 0.8],
            'max_features': [1.0, 0.8],
            'min_samples_split': [2],
            'min_samples_leaf': [1, 2],
            'max_depth': [2, 5, 8]
        }
    }
}


def get_data():
    '''
    get X, y data

    :rtype: tuple
    '''
    _, _, _, train_gray_data, _, _, labels = i_p.load_data()
    data_df = f.make_data_df(train_gray_data, labels)
    fu = FeatureUnion(transformer_list=f.feature_transformer_rule)
    X = fu.fit_transform(data_df)
    y = np.concatenate(data_df["label"].apply(lambda x: x.flatten()))

    return (X, y)


def get_data_Kfold(mode):
    '''
    get X, y data

    :rtype: tuple
    '''

    if mode == "train":
        _, _, _, train_gray_data, _, _, labels = i_p.load_data()
        data_df = f.make_data_df(train_gray_data, labels)
        data_df = data_df.reset_index()
        data_df.columns = ["pngname", "input", "label"]

        keys = np.asarray(train_gray_data.keys())
        kf = cross_validation.KFold(n=len(keys), n_folds=5)

        return data_df, keys, kf

    elif mode == "test":

        _, _, _, _, test_gray_data, _, _ = i_p.load_data()

        return test_gray_data

    else:
        print "mode error!"
        print "set \"train\" or \"test\""
        quit()


def set_validdata(df, keys):

    fu = FeatureUnion(transformer_list=f.feature_transformer_rule)
    Std = preprocessing.StandardScaler()

    for i in xrange(len(keys)):
        if i == 0:
            valid_df = df[(df["pngname"] == keys[i])]
        else:
            valid_df = pd.concat([valid_df, df[(df["pngname"] == keys[i])]])

    valid_df = valid_df.drop("pngname", axis=1).reset_index()

    X = fu.fit_transform(valid_df)
    y = np.concatenate(valid_df["label"].apply(lambda x: x.flatten()))

    X = Std.fit_transform(X)

    return (X, y)


def set_traindata(df, key):

    fu = FeatureUnion(transformer_list=f.feature_transformer_rule)
    Std = preprocessing.StandardScaler()

    X = fu.fit_transform(df)
    y = np.concatenate(df["label"].apply(lambda x: x.flatten()))

    X = Std.fit_transform(X)

    return (X, y)


def kfold_validation_model(model_name="LR"):
    data_df, keys, kf = get_data_Kfold("train")

    """
    SGD Regression model with stochastic gradient descent
    Prnalty : L2
    """
    scores = []
    cnt = 1

    for train_index, valid_index in kf:

        print cnt
        cnt += 1

        clf = sklearn.linear_model.SGDRegressor(penalty='l2')

        train_keys = keys[train_index]
        valid_keys = keys[valid_index]

        for i in xrange(len(train_keys)):

            train_X, train_y = set_traindata(data_df, train_keys[i])
            clf.partial_fit(train_X, train_y)

        valid_X, valid_y = set_validdata(data_df, valid_keys)
        # predict_prova = clf.predict(valid_X)
        predict_y = clf.predict(valid_X)
        score = mean_absolute_error(valid_y, predict_y)
        scores.append(score)

    print scores
    print "Score_Average:", np.average(np.asarray(scores))


def cross_validation_model(model_name="LR"):
    X, y = get_data()
    clf = GridSearchCV(estimator=clf_dict[model_name]["clf"],
                       param_grid=clf_dict[model_name]["paramteters"],
                       n_jobs=3, scoring="accuracy")
    scores = cross_validation.cross_val_score(clf, X, y, cv=5)
    print scores


def downsampling_data(X, y, ratio=0.5, random_state=1):
    np.random.seed(random_state)
    assert X.shape[0] == y.size
    length = X.shape[0]
    len_range = range(0, length)
    use_length = int(length * ratio)
    use_index = np.random.choice(len_range, use_length, replace=False)
    use_X = X[use_index, :]
    use_y = y[use_index]

    return (use_X, use_y)

if __name__ == '__main__':
    # cross_validation_model()
    kfold_validation_model()
