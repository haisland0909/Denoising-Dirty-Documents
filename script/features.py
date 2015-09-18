'''
Created on 2015/08/27

@author: haisland0909
'''
import numpy as np
import img_to_pickle as i_p
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion


def make_test_df(test_gray_data):
    data_df = pd.DataFrame([test_gray_data], index=["input"]).T

    return data_df.dropna()


def make_data_df(train_gray_data, labels):
    data_df = pd.DataFrame([train_gray_data, labels],
                           index=["input", "label"]).T

    return data_df.dropna()


"""
def make_data_df(train_gray_data, clean_gray_data):
    data_df = pd.DataFrame([train_gray_data, clean_gray_data],
                           index=["train", "label"]).T

    return data_df.dropna()
"""


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

        :param pandas.DataFrame data_df:
        :rtype: numpy.array
        '''

        train = data_df["input"]
        # train = data_df["train"]

        return np.concatenate(train.apply(lambda x: x.flatten()))[None].T\
            .astype(np.float)


class SideofImage(BaseEstimator, TransformerMixin):
    '''
    side of image feature
    '''

    def get_feature_names(self):

        return [self.__class__.__name__]

    @staticmethod
    def get_feature_array(image_arr):
        '''
        side is 1 other is 0

        :param numpy.array image_arr:
        :rtype: numpy.array
        '''
        res = np.zeros(image_arr.shape)
        res[0, :] = 1
        res[:, 0] = 1
        res[-1, :] = 1
        res[:, -1] = 1

        return res.flatten()

    def fit(self, data_df, y=None):
        '''
        fit

        :param padas.DataFrame data_df
        :rtype: SideofImage
        '''

        return self

    def transform(self, data_df):
        '''
        transform

        :param padas.DataFrame data_df
        :rtype: numpy.array
        '''

        train = data_df["input"]
        # train = data_df["train"]

        return np.concatenate(train.apply(self.get_feature_array))[None].T\
            .astype(np.float)


class AverageImage(BaseEstimator, TransformerMixin):
    '''
    average of image feature
    '''

    def get_feature_names(self):

        return [self.__class__.__name__]

    @staticmethod
    def get_feature_array(image_arr):
        '''
        get avarage image

        :param numpy.array image_arr:
        :rtype: numpy.array
        '''
        res = np.ones(image_arr.shape)
        mean = image_arr.mean()
        res = res * mean

        return res.flatten()

    def fit(self, data_df, y=None):
        '''
        fit

        :param padas.DataFrame data_df
        :rtype: SideofImage
        '''

        return self

    def transform(self, data_df):
        '''
        transform

        :param padas.DataFrame data_df
        :rtype: numpy.array
        '''
        train = data_df["input"]
        # train = data_df["train"]

        return np.concatenate(train.apply(self.get_feature_array))[None].T\
            .astype(np.float)


class RowAverageImage(BaseEstimator, TransformerMixin):
    '''
    row average of image feature
    '''

    def get_feature_names(self):

        return [self.__class__.__name__]

    @staticmethod
    def get_feature_array(image_arr):
        '''
        get avarage image

        :param numpy.array image_arr:
        :rtype: numpy.array
        '''
        res = np.ones(image_arr.shape)
        mean = image_arr.mean(axis=1)[None].T
        res = res * mean

        return res.flatten()

    def fit(self, data_df, y=None):
        '''
        fit

        :param padas.DataFrame data_df
        :rtype: SideofImage
        '''

        return self

    def transform(self, data_df):
        '''
        transform

        :param padas.DataFrame data_df
        :rtype: numpy.array
        '''
        train = data_df["input"]
        # train = data_df["train"]

        return np.concatenate(train.apply(self.get_feature_array))[None].T\
            .astype(np.float)


class ColAverageImage(BaseEstimator, TransformerMixin):
    '''
    col average of image feature
    '''

    def get_feature_names(self):

        return [self.__class__.__name__]

    @staticmethod
    def get_feature_array(image_arr):
        '''
        get avarage image

        :param numpy.array image_arr:
        :rtype: numpy.array
        '''
        res = np.ones(image_arr.shape)
        mean = image_arr.mean(axis=0)[None]
        res = res * mean

        return res.flatten()

    def fit(self, data_df, y=None):
        '''
        fit

        :param padas.DataFrame data_df
        :rtype: SideofImage
        '''

        return self

    def transform(self, data_df):
        '''
        transform

        :param padas.DataFrame data_df
        :rtype: numpy.array
        '''
        train = data_df["input"]
        # train = data_df["train"]

        return np.concatenate(train.apply(self.get_feature_array))[None].T\
            .astype(np.float)


class RelativeCoordinateX(BaseEstimator, TransformerMixin):
    '''
    relative coordinate of image feature
    '''

    def get_feature_names(self):

        return [self.__class__.__name__]

    @staticmethod
    def get_feature_array(image_arr):
        '''
        relative coordinate

        :param numpy.array image_arr:
        :rtype: numpy.array
        '''
        x_shape, y_shape = image_arr.shape
        x = np.linspace(0, 1, x_shape)
        y = np.linspace(0, 1, y_shape)
        xv, yv = np.meshgrid(x, y)

        return xv.flatten()

    def fit(self, data_df, y=None):
        '''
        fit

        :param padas.DataFrame data_df
        :rtype: RelativeCoordinateX
        '''

        return self

    def transform(self, data_df):
        '''
        transform

        :param padas.DataFrame data_df
        :rtype: numpy.array
        '''
        train = data_df["input"]
        # train = data_df["train"]

        return np.concatenate(train.apply(self.get_feature_array))[None].T\
            .astype(np.float)


class RelativeCoordinateY(BaseEstimator, TransformerMixin):
    '''
    relative coordinate of image feature
    '''

    def get_feature_names(self):

        return [self.__class__.__name__]

    @staticmethod
    def get_feature_array(image_arr):
        '''
        relative coordinate

        :param numpy.array image_arr:
        :rtype: numpy.array
        '''
        x_shape, y_shape = image_arr.shape
        x = np.linspace(0, 1, x_shape)
        y = np.linspace(0, 1, y_shape)
        xv, yv = np.meshgrid(x, y)

        return yv.flatten()

    def fit(self, data_df, y=None):
        '''
        fit

        :param padas.DataFrame data_df
        :rtype: RelativeCoordinateY
        '''

        return self

    def transform(self, data_df):
        '''
        transform

        :param padas.DataFrame data_df
        :rtype: numpy.array
        '''
        train = data_df["input"]
        # train = data_df["train"]

        return np.concatenate(train.apply(self.get_feature_array))[None].T\
            .astype(np.float)

feature_transformer_rule = [
    ('gray', GrayParam()),
    ('avarage', AverageImage()),
    ('rowavarage', RowAverageImage()),
    ('colavarage', ColAverageImage()),
    ('coordinateX', RelativeCoordinateX()),
    ('coordinateY', RelativeCoordinateY()),
]

if __name__ == '__main__':
    _, _, _, train_gray_data, _, _, labels = i_p.load_data()
    data_df = make_data_df(train_gray_data, labels)
    transformer_list = [
        ('average', RelativeCoordinateY())
    ]
    fu = FeatureUnion(transformer_list=transformer_list)
    feature = fu.fit_transform(data_df)
    print feature
    print feature.shape
    print np.isnan(feature).sum()
