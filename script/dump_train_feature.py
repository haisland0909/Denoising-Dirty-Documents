'''
Created on 2015/09/17

@author: haisland0909
'''
from sklearn.pipeline import FeatureUnion
import img_to_pickle as i_p
import classify as cl
import features as f
import numpy as np
import pandas as pd
import os


def dump_train():
    _, _, _, train_gray_data, test_gray_data, _, labels = i_p.load_data()

    train_df = f.make_data_df(train_gray_data, labels)
    test_df = f.make_test_df(test_gray_data)

    train_df = train_df.reset_index()
    test_df = test_df.reset_index()

    train_df.columns = ["pngname", "input", "label"]
    test_df.columns = ["pngname", "input"]

    fu = FeatureUnion(transformer_list=f.feature_transformer_rule)
    feature_name_list = fu.get_feature_names() + ["target"]
    train_X = fu.fit_transform(train_df)
    train_y = np.concatenate(train_df["label"].apply(lambda x: x.flatten()))
    train_X, train_y = cl.downsampling_data(train_X, train_y, 0.2)
    train_dump = pd.DataFrame(np.c_[train_X, train_y], columns=feature_name_list)
    dump_path = os.path.abspath(os.path.dirname(__file__)) +\
        "/../tmp/train_dump"
    train_dump.to_csv(dump_path + "/train_dump.csv", index=False)

if __name__ == '__main__':

    dump_train()
