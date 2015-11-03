'''
Created on 2015/09/29

@author: haisland0909
'''
import numpy as np
import pandas as pd
import os
from PIL import Image

ROOT = os.path.abspath(os.path.dirname(__file__))
SUBMISSION = ROOT + "/../tmp/submission/"
TMP = ROOT + "/../tmp/predict_picture/"


class Predict_Picture:

    def __init__(self):
        self._image_num = None
        self._image_arr = None
        self._coordinate = None

    def _apply_inner(self, picture_sr):
        id = picture_sr["id"].split("_")
        image_num = id[0]
        coordinate = (id[1], id[2])
        if self._image_num is None:
            self._image_num = image_num
        if self._image_num != image_num:
            self._create_image()
        if self._image_arr is None:
            self._image_arr = []
        self._image_arr.append(picture_sr["value"] * 255)
        self._coordinate = coordinate

    def _create_image(self):
        image = np.array(self._image_arr)\
            .reshape(int(self._coordinate[0]), int(self._coordinate[1]))
        print image
        image = Image.fromarray(image.astype(np.uint8))
        file_name = TMP + "%s.png" % self._image_num
        image.save(file_name)
        self._image_num = None
        self._image_arr = None

    def create_picture(self, file_name):
        file_path = SUBMISSION + file_name
        predict = pd.read_csv(file_path)
        for index, row in predict.iterrows():
            self._apply_inner(row)
        self._create_image()


if __name__ == '__main__':
    obj = Predict_Picture()
    obj.create_picture("submission_repredict2015_10_04_23_28_27.csv")
