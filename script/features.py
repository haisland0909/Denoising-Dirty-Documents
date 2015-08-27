'''
Created on 2015/08/27

@author: haisland0909
'''
import numpy as np
import img_to_pickle as i_p
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion


def make_data_df(train_gray_data, clean_gray_data):
    data_df = pd.DataFrame([train_gray_data, clean_gray_data],
                           index=["train", "label"]).T

    return data_df.dropna()


class GrayParam(BaseEstimator, TransformerMixin):
    '''
    gray scale feature
    '''

    def get_feature_names(self):

        return [self.__class__.__name__]

    def fit(self, data_df, y=None):
        '''
        fit

        :param padas.DataFrame data_df
        :rtype: GrayParam
        '''

        return self

    def transform(self, data_df):
        '''
        transform

        :param padas.DataFrame data_df
        :rtype: numpy.array
        '''
        train = data_df["train"]

        return np.concatenate(train.apply(lambda x: x.flatten()))[None].T\
            .astype(np.float)

feature_transformer_rule = [
    ('gray', GrayParam())
]

if __name__ == '__main__':
    train_data, test_data, clean_data, train_gray_data, test_gray_data, clean_gray_data = i_p.load_data()
    data_df = make_data_df(train_gray_data, clean_gray_data)
    transformer_list = [
        ('gray', GrayParam())
    ]
    fu = FeatureUnion(transformer_list=transformer_list)
    feature = fu.fit_transform(data_df)
    print feature.shape
    print np.isnan(feature).sum()
