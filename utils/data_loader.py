"""
Data loader file for CD2014 dataset
"""

import csv
import os
import numpy as np
import random
import torch.utils.data as data
from configs import data_config
import cv2
import glob

class CD2014Dataset(data.Dataset):
    """
    Data loader class
    """

    def __init__(self, dataset, use_selected=200, multiplier=16, transforms=None, shuffle=False):
        if use_selected and use_selected != -1:
            if use_selected == 200:
                selected_frs_csv = data_config.selected_frs_200_csv
            else:
                raise(f"Number of selected frames can be None or 200 but {use_selected} given")

            with open(selected_frs_csv) as f:
                reader = csv.reader(f)
                selected_frs = list(reader)

            # Create a dictionary of cat/vid -> list of selected frames
            catvid_to_selected_frs = {arr[0]:list(map(int, arr[1].split())) for arr in selected_frs}

        input_tuples = []
        for cat, vid_arr in dataset.items():
            for vid in vid_arr:
                # Find out the required frame ids (either selected or all the ones that have gt)
                if use_selected == -1:
                    last_fr = int(
                        sorted(glob.glob(os.path.join(data_config.current_fr_dir.format(cat=cat, vid=vid), "*.jpg")))[
                            -1][-10:-4])
                    fr_ids = [idx for idx in range(1, last_fr + 1)]
                elif use_selected:
                    fr_ids = catvid_to_selected_frs[f"{cat}/{vid}"]
                else:
                    roi_path = data_config.temp_roi_path.format(cat=cat, vid=vid)
                    with open(roi_path) as f:
                        reader = csv.reader(f)
                        temp_roi = list(reader)
                    temp_roi = list(map(int, temp_roi[0][0].split()))
                    fr_ids = [idx for idx in range(temp_roi[0], temp_roi[1] + 1)]

                for fr_id in fr_ids:
                    input_tuples.append((cat, vid, fr_id))

        self.input_tuples = input_tuples
        self.n_data = len(input_tuples)
        self.multiplier = multiplier
        self.transforms = transforms
        self.shuffle = shuffle

    def __getitem__(self, item):
        if self.shuffle:
            cat, vid, fr_id = random.choice(self.input_tuples)
        else:
            cat, vid, fr_id = self.input_tuples[item]

        # Construct the input

        inp = {"current_fr":None}

        inp["current_fr"] = self.__readRGB(data_config.fr_path\
                                .format(cat=cat, vid=vid, fr_id=str(fr_id).zfill(6)))

        label = self.__readGray(data_config.gt_path \
                                .format(cat=cat, vid=vid, fr_id=str(fr_id).zfill(6)))

        for transform_arr in self.transforms:
            if len(transform_arr) > 0:
                inp, label = self.__selectrandom(transform_arr)(inp, label)

        if(self.multiplier > 0):
            c, h, w = label.shape
            h_valid = int(h/self.multiplier)*self.multiplier
            w_valid = int(w/self.multiplier)*self.multiplier
            inp, label = inp[:, :h_valid, :w_valid], label[:, :h_valid, :w_valid]

        # reformat label such that FG=1, BG=0, everything else = -1
        label[label <= 0.05] = 0 # BG
        label[np.abs(label-0.5) < 0.45] = -1
        label[label >= 0.95] = 1 # FG

        return inp, label

    def __len__(self):
        return self.n_data

    def __readRGB(self, path):
        assert os.path.exists(path), f"{path} does not exist"
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float)/255

    def __readGray(self, path):
        assert os.path.exists(path), f"{path} does not exist"
        return np.expand_dims(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY), -1).astype(np.float)/255

    def __selectrandom(self, arr):
        choice = arr.copy()
        while isinstance(choice, list):
            choice = random.choice(choice)
        return choice



