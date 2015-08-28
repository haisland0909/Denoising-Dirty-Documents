'''
Created on 2015/08/28

@author: haisland0909
'''
from sklearn.pipeline import FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
import sklearn.linear_model
import sklearn.ensemble
import img_to_pickle as i_p
import features as f
import numpy as np

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
    _, _, _, train_gray_data, _, clean_gray_data = i_p.load_data()
    data_df = f.make_data_df(train_gray_data, clean_gray_data)
    fu = FeatureUnion(transformer_list=f.feature_transformer_rule)
    X = fu.fit_transform(data_df)
    y = np.concatenate(data_df["label"].apply(lambda x: x.flatten()))

    return (X, y)


def cross_validation_model(model_name="LR"):
    X, y = get_data()
    clf = GridSearchCV(estimator=clf_dict[model_name]["clf"],
                       param_grid=clf_dict[model_name]["paramteters"],
                       n_jobs=3, scoring="accuracy")
    scores = cross_validation.cross_val_score(clf, X, y, cv=5)
    print scores

if __name__ == '__main__':
    cross_validation_model()