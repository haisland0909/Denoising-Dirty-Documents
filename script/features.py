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


class SobelFilter_hol(BaseEstimator, TransformerMixin):
    '''
    gray scale feature
    '''

    def get_feature_names(self):

        return [self.__class__.__name__]

    @staticmethod
    def get_filter_array(image_arr):
        '''
        get avarage image

        :param numpy.array image_arr:
        :rtype: numpy.array
        '''

        row = image_arr.shape[0] + 2
        col = image_arr.shape[1] + 2

        cols = [0] * col
        image_pad = [cols] * row
        image_pad = np.asarray(image_pad)
        image_pad = image_pad.astype(np.uint8)

        image_pad[1:1 + image_arr.shape[0], 1:1 + image_arr.shape[1]] = image_arr
        image_sobel = np.zeros(image_arr.shape)

        for row in xrange(image_sobel.shape[0]):
            for col in xrange(image_sobel.shape[1]):

                image_sobel[row][col] = -1 * image_pad[row][col]   + 0 * image_pad[row + 1][col]   +  1 * image_pad[row + 2][col] +\
                                        -2 * image_pad[row][col + 1] + 0 * image_pad[row + 1][col + 1] +  2 * image_pad[row + 2][col + 1] +\
                                        -1 * image_pad[row][col + 2] + 0 * image_pad[row + 2][col + 2] + 1 * image_pad[row + 2][col + 2]

        return image_sobel.flatten()

    def fit(self, data_df, y=None):
        '''
        fit

        :param padas.DataFrame data_df
        :rtype: SobelFilter
        '''

        return self

    def transform(self, data_df):
        '''
        transform

        :param pandas.DataFrame data_df:
        :rtype: numpy.array
        '''

        train = data_df["input"]
        # print train

        return np.concatenate(train.apply(self.get_filter_array))[None].T\
            .astype(np.float)


class SobelFilter_ver(BaseEstimator, TransformerMixin):
    '''
    gray scale feature
    '''

    def get_feature_names(self):

        return [self.__class__.__name__]

    @staticmethod
    def get_filter_array(image_arr):
        '''
        get avarage image

        :param numpy.array image_arr:
        :rtype: numpy.array
        '''

        row = image_arr.shape[0] + 2
        col = image_arr.shape[1] + 2

        cols = [0] * col
        image_pad = [cols] * row
        image_pad = np.asarray(image_pad)
        image_pad = image_pad.astype(np.uint8)

        image_pad[1:1 + image_arr.shape[0], 1:1 + image_arr.shape[1]] = image_arr
        image_sobel = np.zeros(image_arr.shape)

        for row in xrange(image_sobel.shape[0]):
            for col in xrange(image_sobel.shape[1]):

                image_sobel[row][col] = -1 * image_pad[row][col]   + -2 * image_pad[row + 1][col]  + -1 * image_pad[row + 2][col] +\
                    0 * image_pad[row][col + 1] + 0 * image_pad[row + 1][col + 1] +  0 * image_pad[row + 2][col + 1] +\
                    1 * image_pad[row][col + 2] + 2 * image_pad[row + 2][col + 2] + 1 * image_pad[row + 2][col + 2]

        return image_sobel.flatten()

    def fit(self, data_df, y=None):
        '''
        fit

        :param padas.DataFrame data_df
        :rtype: SobelFilter
        '''

        return self

    def transform(self, data_df):
        '''
        transform

        :param pandas.DataFrame data_df:
        :rtype: numpy.array
        '''

        train = data_df["input"]
        # print train

        return np.concatenate(train.apply(self.get_filter_array))[None].T\
            .astype(np.float)


class RapFilter(BaseEstimator, TransformerMixin):
    '''
    gray scale feature
    '''

    def get_feature_names(self):

        return [self.__class__.__name__]

    @staticmethod
    def get_filter_array(image_arr):
        '''
        get avarage image

        :param numpy.array image_arr:
        :rtype: numpy.array
        '''

        row = image_arr.shape[0] + 2
        col = image_arr.shape[1] + 2

        cols = [0] * col
        image_pad = [cols] * row
        image_pad = np.asarray(image_pad)
        image_pad = image_pad.astype(np.uint8)

        image_pad[1:1 + image_arr.shape[0], 1:1 + image_arr.shape[1]] = image_arr
        image_rap = np.zeros(image_arr.shape)

        for row in xrange(image_rap.shape[0]):
            for col in xrange(image_rap.shape[1]):

                image_rap[row][col] =  1 * image_pad[row][col]   +  1 * image_pad[row + 1][col]   + 1 * image_pad[row + 2][col] +\
                    1 * image_pad[row][col + 1] + -8 * image_pad[row + 1][col + 1] + 1 * image_pad[row + 2][col + 1] +\
                    1 * image_pad[row][col + 2] + 1 * image_pad[row + 2][col + 2] + 1 * image_pad[row + 2][col + 2]

        return image_rap.flatten()

    def fit(self, data_df, y=None):
        '''
        fit

        :param padas.DataFrame data_df
        :rtype: RapFilter
        '''

        return self

    def transform(self, data_df):
        '''
        transform

        :param pandas.DataFrame data_df:
        :rtype: numpy.array
        '''

        train = data_df["input"]
        # print train

        return np.concatenate(train.apply(self.get_filter_array))[None].T\
            .astype(np.float)


class GauFilter(BaseEstimator, TransformerMixin):
    '''
    gray scale feature
    '''

    def get_feature_names(self):

        return [self.__class__.__name__]

    @staticmethod
    def get_filter_array(image_arr):
        '''
        get avarage image

        :param numpy.array image_arr:
        :rtype: numpy.array
        '''

        row = image_arr.shape[0] + 2
        col = image_arr.shape[1] + 2

        cols = [0] * col
        image_pad = [cols] * row
        image_pad = np.asarray(image_pad)
        image_pad = image_pad.astype(np.uint8)

        image_pad[1:1 + image_arr.shape[0], 1:1 + image_arr.shape[1]] = image_arr
        image_gau = np.zeros(image_arr.shape)

        for row in xrange(image_gau.shape[0]):
            for col in xrange(image_gau.shape[1]):

                image_gau[row][col] =  float(image_pad[row][col])   / 9 +  float(image_pad[row + 1][col])   / 9 + float(image_pad[row + 2][col])   / 9 +\
                    float(image_pad[row][col + 1]) / 9 +  float(image_pad[row + 1][col + 1]) / 9 + float(image_pad[row + 2][col + 1]) / 9 +\
                    float(image_pad[row][col + 2]) / 9 + float(image_pad[row + 2][col + 2]) / 9 + float(image_pad[row + 2][col + 2]) / 9

        #image_gau = image_gau.astype(np.uint8)

        return image_gau.flatten()

    def fit(self, data_df, y=None):
        '''
        fit

        :param padas.DataFrame data_df
        :rtype: RapFilter
        '''

        return self

    def transform(self, data_df):
        '''
        transform

        :param pandas.DataFrame data_df:
        :rtype: numpy.array
        '''

        train = data_df["input"]
        # print train

        return np.concatenate(train.apply(self.get_filter_array))[None].T\
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
    # ('side', SideofImage()),
    ('avarage', AverageImage()),
    ('rowavarage', RowAverageImage()),
    ('colavarage', ColAverageImage()),
    ('solbel_hol', SobelFilter_hol()),
    ('solbel_ver', SobelFilter_ver()),
    ('raprasian', RapFilter()),
    ('gaussian', GauFilter()),
    ('coordinateX', RelativeCoordinateX()),
    ('coordinateY', RelativeCoordinateY()),
]

if __name__ == '__main__':
    _, _, _, train_gray_data, _, _, labels = i_p.load_data()
    data_df = make_data_df(train_gray_data, labels)
    transformer_list = [
        ('average', GrayParam())
    ]
    fu = FeatureUnion(transformer_list=transformer_list)
    feature = fu.fit_transform(data_df)
    print feature
    print feature.shape
    print np.isnan(feature).sum()
