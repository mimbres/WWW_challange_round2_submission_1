#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:31:52 2018

@author: sungkyun
"""
from torch.utils.data.dataset import Dataset
from torch import from_numpy
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

class FmaGenreDataset(Dataset):
    def __init__(self, csv_path, zpad=None, transform=None, segment_collect=None, scale_factor_filepath=None, sample_range=None):
        # Internal variables
        self.zpad = zpad
        self.transform = transform
        self.scale_factor_filepath = scale_factor_filepath
        self.npy_filepaths = None
        self.labels = None
        self.int_labels = None
        self.classnames = None
        self.total_classes = None
        self.track_ids = None
        self.segment_collect = segment_collect
        self.sample_range = sample_range #repack dataset by selecting examples with list [idx_start, idx_end]

        
        """
            'segment_collect = None' : default
            'segment_collect = 3'    : collect 3 consecutive segments, and use it as one item.
            'segment_collect = 9'    : collect 9 (we can't use more than 9, because of the wav length limit=30s)
        """

        # Reading .csv file
        df = pd.read_csv(csv_path, index_col=0) # ['track_id', 'npy_filepath', 'genre']
        if self.sample_range is not None:
            df = df.iloc[sample_range[0]:sample_range[1]]
        #df = df.iloc[0:84]

        self.npy_filepaths = df.iloc[:,1] # get only mp3_filepath
        self.track_ids = df.iloc[:,0]

        if 'genre' in df.columns:
            self.labels = df.iloc[:,2]

            # Convert labels to onehot
            le = preprocessing.LabelEncoder()
            """
            # Encode labels with A-B-C order as ['Blues', 'Classical', 'Country', 'Easy Listening', 'Electronic',
            'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International',
            'Jazz', 'Old-Time / Historic', 'Pop', 'Rock', 'Soul-RnB', 'Spoken']
            """
            self.total_classes = len(df.genre.unique()) # = 16
            #self.onehot_labels = np.eye(self.total_classes, dtype='float32')[le.fit_transform(self.labels)] # encode labels into onehot
            self.int_labels = le.fit_transform(self.labels)
            self.classnames = le.classes_

        else:
            self.labels = None
            
            
        """
        Segment collection:
            'segment_collect = 3' will collect 3 consecutive segments within a song as one item.
            'segment_collect = 9' will collect 9 consecutive segments within a song as one item.
            EX) segment index is [1,2,3,...10,11], for each calling __getitem__() method,
                return data[1,2,3]
                return data[2,3,4]
                ...
                return data[9,10,11]
            
            Here, we modify (reduce last segment ids for):
                - self.npy_filepaths --> reconstruct each item as a list of filepaths
                - self.track_ids --> drop last segments
                - self.int_labels --> drop last segments
        """
            
        if self.segment_collect is not None:
            # Get number of segments for each track_ids or song.
            _, seg_start_idx_all, n_seg_all = np.unique(self.track_ids.as_matrix(), return_index=True, return_counts=True)
            
            # Get all indices(of dataframes) to collect 
            indices_to_collect_start = np.asarray([], dtype='int64')
            for i in range(len(seg_start_idx_all)):
                if n_seg_all[i] - (self.segment_collect-1) > 0:
                    indices_to_collect_start = np.append(indices_to_collect_start, np.arange(seg_start_idx_all[i], seg_start_idx_all[i] + n_seg_all[i] - (self.segment_collect-1)))
                else: # if song length is short
                    indices_to_collect_start = np.append(indices_to_collect_start, seg_start_idx_all[i])
            
            # Modify self.track_ids, self.int_labels
            self.track_ids = self.track_ids[indices_to_collect_start]
            self.int_labels = self.int_labels[indices_to_collect_start]
            # New indexing
            self.track_ids.index = np.arange(len(self.track_ids))
            # int_labels does not need to index, because it is numpy array.
            
            # Now, modify self.npy_filepaths
            tmp_df = pd.DataFrame(index=np.arange(len(self.track_ids)),  columns=['npy filepath list'] )
            
            for i in range(len(self.track_ids)): 
                #cur_index = i # this 'i' is equivalent to __getitem__'s 'index'
                cur_track = self.track_ids[i]
                cur_start_id = indices_to_collect_start[i]
                cur_song_num_segs = len(df.loc[lambda df: df.track_id==cur_track]) #current song's number of segments that we have
                
                tmp_path = []
                for j in range(self.segment_collect): #3:0~2, 9: 0~8 
                    if j==0:
                        tmp_path.append(self.npy_filepaths[cur_start_id])
                    elif (self.segment_collect < cur_song_num_segs):
                        tmp_path.append(self.npy_filepaths[cur_start_id+j])
                    else:
                        tmp_path.append('no_data')
                # Set collected .npy paths into tmp_df 
                tmp_df.iloc[i] = [tmp_path]
            
            # Finally, replace self.npy_filepahs with tmp_df
            self.npy_filepaths = tmp_df
            # in __getitem__, each item will be called as self.npy_filepaths.iloc[index][0]: this is a list of filenames.
            del(tmp_df)    
                
    def __getitem__(self, index):
        
        track_id = self.track_ids[index]
        
        # Load npy
        if self.segment_collect is None:
            x = np.load(self.npy_filepaths[index])
    
            if self.zpad is not None:
                print('zpad is not implemented!!')
            #if self.transform is not None:
            if self.transform is 'normalize':
                scaler = joblib.load(self.scale_factor_filepath)
                
                x = scaler.transform(np.transpose(x.reshape(x.shape[1], x.shape[2])))
                x = np.transpose(x)
                x = x.reshape(1, x.shape[0], x.shape[1])
        elif self.segment_collect is 'full':
            print('qdwd;')
        elif self.segment_collect is not None: 
            # Here, we need to open multiple .npy files and concatenate or zero-pad them.
            x = np.array([], dtype='float32').reshape(1,0)
            
            for fpath in self.npy_filepaths.iloc[index][0]: # for all npy files in the index
                if fpath is 'no_data':
                    # if 'no_data', apply zero-padding
                    pad_l_sz = int(np.random.rand(1)*59049)
                    pad_r_sz = 59049 - pad_l_sz # pad_left + pad_right = 59049
                    x = np.pad(x[0,:], pad_width=(pad_l_sz, pad_r_sz), mode='constant', constant_values=0) # this is too slow!!
                    x = x.reshape(1, len(x))
                else:
                    # We have data, so we concatenate them
                    x = np.concatenate((x, np.load(fpath)), axis=1) 


        # Get label, if it exists
        if self.labels is not None:
            #onehot_label = torch.from_numpy(self.onehot_labels[index])
            int_label = self.int_labels[index]
            return track_id, x, int_label
        else:
            return track_id, x



    def __len__(self):
        return len(self.npy_filepaths) # return the number of examples that we have
