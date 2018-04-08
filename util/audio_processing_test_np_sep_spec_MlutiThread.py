#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi Thread version of audio pre-processing for test data

Created on Tue Jan 30 14:53:51 2018

@author: sungkyun
"""

# Preprocess Test Dataset: test set doens't include labels
import argparse
import numpy as np
import pandas as pd # required for reading .csv files
#import fma # required for reading fma formatted data
import librosa # required for audio pre-processing, loading mp3 (sudo apt-get install libav-tools)
import glob # required for obtaining test file ID
import os 
import joblib # required for multi processing
#%% Parser
parser = argparse.ArgumentParser(description='Audio Preprocessing for test data')
parser.add_argument('-nmels', '--nmels', type=int, default=192, metavar='N',
                    help='number of mel bins, default 192')
parser.add_argument('-ns', '--n_samples', type=int, default=86, metavar='N',
                    help='number of samples for each segment, default 86')
parser.add_argument('-sr', '--sr', type=int, default=22050, metavar='N',
                    help='target sampling rate, default 22050')
parser.add_argument('-ncpu', '--ncpu', type=int, default=None, metavar='N',
                    help='number of cpu threads, default = 0.9 X max_cpu')
parser.add_argument('-i', '--fma_testset_filedir', type=str, default='data/crowdai_fma_test/', metavar='N',
                    help='input test filedir, default=data/crowdai_fma_test/')
parser.add_argument('-o', '--out_filedir', type=str, default='data/preprocessed_data_spec/test/', metavar='N',
                    help='output file directory(root of .wav subdirectories and .csv file), default=data/preprocessed_data_spec/test/')


args = parser.parse_args()



#%%
# Deifinition/Args
fma_testset_filedir = args.fma_testset_filedir# You can replace .mp3 with other format
out_filedir = args.out_filedir # root of .wav subdirectories and .csv file
n_samples = args.n_samples #samples per segement, 43sample = 1 sec (sr=22050)

# Get the list of test_ids
org_test_ids = sorted(glob.glob(fma_testset_filedir + '*.mp3'))
org_test_ids = [path.split('/')[-1][:-4] for path in org_test_ids]


#%% def

# join_csv_files() deletes all temporary csv files and create one df
def join_csv_files(tmp_csv_path_all):
    df_new = pd.read_csv(tmp_csv_path_all[0], index_col=0)
    for i in range(1, len(tmp_csv_path_all)):
        df_new = pd.concat([df_new, pd.read_csv(tmp_csv_path_all[i], index_col=0)])
    # set new index
    df_new = df_new.set_index(np.array(range(len(df_new))))
    
    # now, delete all tmp_csv files
    for files_to_delete in tmp_csv_path_all:
        os.remove(files_to_delete)
    
    return df_new


# This preproces_func will save temporary .csv as "tmp_df_<num>.csv", where num is the work_idx
def preprocess_func(work_idx):
    test_id_indices_to_process = org_test_id_indices_to_process_all[work_idx]
    #test_ids = org_test_ids[test_id_indices_to_process]
    t=test_id_indices_to_process
    test_ids = org_test_ids[t[0]:t[-1]]
    
    # Prepare df to fill the new <preprocessed_train.csv> file...
    df = pd.DataFrame(columns=('test_id', 'npy_filepath'))
    total_test_ids = len(test_ids)
    
    for i in range(total_test_ids):
        mp3_path = fma_testset_filedir + '{}.mp3'.format(test_ids[i]) # get mp3 path
        
        # Load audio -> mono -> resample -> melspec -> slice    
        x_raw, sr = librosa.load(mp3_path, sr=args.sr, mono=True)
        x = librosa.feature.melspectrogram(y=x_raw, sr=sr, n_fft=1024, power=2.0, hop_length=512, n_mels=args.nmels)
        x = librosa.power_to_db(x, ref=np.max)
        #import librosa.display
        #import matplotlib as plt
        #librosa.display.specshow(x[:,128:128+86], y_axis='mel', fmax=sr, x_axis='time')
        
        total_samples = x.shape[1]
        n_segment = total_samples // n_samples
        
        # Split the signal into segments
        segments = np.asarray([x[:,k * n_samples:(k + 1) * n_samples] for k in range(n_segment)], dtype='float32')
        print('Preprocessing: song {}/{} into {} segments ____ cpu_work_id[{}]'.format(i, total_test_ids, n_segment, work_idx))
        
        for segment_idx in range(n_segment):   
            # new direcotry
            if not os.path.exists(out_filedir + '{}'.format(test_ids[i])):
                os.makedirs(out_filedir + '{}'.format(test_ids[i]))
        
            # Save segments
            save_filepath = out_filedir + '{}'.format(test_ids[i]) + '/{}.npy'.format(segment_idx)
            np.save(save_filepath, segments[segment_idx][:][:].reshape(1,segments.shape[1], n_samples))
            
            # prepare DataFrame to create .csv
            df.loc[len(df)] = [test_ids[i], save_filepath] # add a new row into DataFrame
        
    # save temporary tmp_df_<num>.csv
    df.to_csv(out_filedir + 'tmp_df_{:08d}.csv'.format(work_idx), encoding='utf-8')
    return 0



#%% Multi-processing
    
# Decide how many cpu workers to employ
if args.ncpu is not None:
    n_cpu_workers = args.ncpu
else:
    n_cpu_workers = int(joblib.cpu_count()*0.8)

print('Number of CPU workers to use: ', n_cpu_workers)

org_test_id_indices_to_process_all = np.array_split(range(len(org_test_ids)), 50)
work_indices = range(0, 50)

# Multithread processing
_ = joblib.Parallel(n_jobs=n_cpu_workers, verbose=2, backend="multiprocessing")(map(joblib.delayed(preprocess_func), work_indices))

# Collect tmp_df_<num>.csv files and save as 'preprocessed_train_all.csv'
tmp_csv_path_all = sorted(glob.glob(out_filedir + 'tmp_df_*.csv'))
df_composite = join_csv_files(tmp_csv_path_all) 
df_composite.to_csv(out_filedir + 'preprocessed_all.csv', encoding='utf-8')

