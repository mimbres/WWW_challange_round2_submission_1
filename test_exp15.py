#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 04:01:14 2018

@author: sungkyun Chang


train_exp15


preprocessing:
    - mel-spec {sr=22050, n_fft=1024, power=2.0, hop_length=512, n_mels=192}
    - zero-mean, unit standardization
    - (192 X 86) X 15 segments from 30s audio clip
    
Dual Path Networks:
    - DPN 92, num_init_features=64, k_R=96, G=32, k_sec=(3,4,20,3), inc_sec=(16,32,24,128)
    - Number of trainable parameters = 35,010,768
Batch-size Scheduler:
    - for each 12 epochs: 16,32,64,96,96,.. 
    - max 120 epochs

result:
    - logloss = 0.410 at epoch=115
"""

#%%
import torch
#import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from util.preparing_dataset_np_sep import FmaGenreDataset
import numpy as np
import os
import shutil
import argparse

parser = argparse.ArgumentParser(description='Musical Genre 2018')
parser.add_argument('--feature_dir', dest='feature_dir', action='store', required=True)
parser.add_argument('--output_path', dest='output_path', action='store', required=True)
parser.add_argument('-load', '--load', type=str, default=None, metavar='STR',
                    help='e.g. --load checkpoints/exp00/checkpoint_00', required=True)
args = parser.parse_args()


# Save options
FEATURE_PATH = args.feature_dir + '/preprocessed_all.csv' # 'data/preprocessed_data_spec/test/preprocessed_all.csv'
OUTPUT_PATH = args.output_path #'output.csv'
EXP_NAME = 'exp15'
CLASS_NAMES = ['Blues', 'Classical', 'Country', 'Easy Listening', 'Electronic',
           'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International',
           'Jazz', 'Old-Time / Historic', 'Pop', 'Rock', 'Soul-RnB', 'Spoken']

# Defiinition & settings
BATCH_SIZE = 256 # 42 # 84 #Since exp4, we schedule batch-size 
N_LABELS = 16
LOG_INTERVAL = 1

use_gpu = torch.cuda.is_available()

# Calculate mean and std of dataset
# run util/compute_scale_factors.py
scale_factor_filepath='data/scale_factor.pkl'



# Preparing Dataset
dset_test = FmaGenreDataset(csv_path=FEATURE_PATH, transform='normalize',
                           segment_collect=None, scale_factor_filepath=scale_factor_filepath)




# get data sample as:
# index = 0
# tmp_x, tmp_y = train_loader.dataset.__getitem__(index)
# input_x = Variable(torch.FloatTensor(3,1,192,86), volatile=True)
# batch X channels X length

#%% Architecture
from model.dpn_mod import dpn92

model = dpn92(num_classes=16)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#print('Number of trainable parameters = {}'.format(sum([np.prod(p.size()) for p in model_parameters])) )
## Number of trainable parameters = 35,007,632 (input_sz = 192 x 86)
## Number of trainable parameters = 10,589,488 (input_sz=59049) 
## Number of trainable parameters = 10,812,720 (input_sz=59049 * 3) 
## Number of trainable parameters =  (input_sz=59049 * 9) 
#4,995,376

if use_gpu:
    model = model.cuda()

#optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)

#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-6, nesterov=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5, nesterov=True)


#%%

def test():
    model.eval()
    
    segment_prob = np.array([], dtype='float32').reshape(0, N_LABELS) # An empty array: we first collect segment-wise prediction here.

    total_segments = len(test_loader.dataset) # Number of segments in whole dataset
    total_batch = len(test_loader) # Number of batch partitions

    for batch_idx, (_, X_test) in enumerate(test_loader):
        if use_gpu:
            X_test = X_test.cuda()

        X_test = Variable(X_test, volatile=True)
        y_output = model(X_test)

        # Concatanate y_outputs (segment-wise probabilities)
        segment_prob = np.concatenate((segment_prob, y_output.cpu().data.numpy()))
        
        if batch_idx % 100 == 0:
            print('Test: [{}/{} ({:.0f}%)]'.format(batch_idx * test_loader.batch_size, total_segments, 100. * batch_idx / total_batch))
    
    # Prepare .csv file for Submission: save as /checkpoints/EXP_NAME/submisssion_v{x}.csv
    # NOTE: Here we used train_loader.dataset.classnames because currently our test_loader doesn't include classnames 
    save_csv_pred_v1(probs=segment_prob, 
                     track_ids=test_loader.dataset.track_ids,
                     classnames=CLASS_NAMES,
                     expname=EXP_NAME)
#    save_csv_pred_v2(probs=segment_prob, 
#                     track_ids=test_loader.dataset.track_ids,
#                     classnames=train_loader.dataset.classnames,
#                     expname=EXP_NAME)




# Save model parameters
def save_checkpoint(state, is_best, accuracy):
    exp_weights_root_dir = SAVE_MODEL_DIR + EXP_NAME + '/'
    os.makedirs(exp_weights_root_dir, exist_ok=True)
    filename = exp_weights_root_dir + 'checkpoint.pth.tar'
    torch.save(state, filename)
    if is_best:
        print('best beaten')
        shutil.copyfile(filename, exp_weights_root_dir + 'model_best.pth.tar')

        f=open('best_result.txt','w')
        f.write('acc={}%'.format(acc))
        f.close()

# Load model parameters
def load_checkpoint(filepath):
    dt = torch.load(filepath)


    model.load_state_dict(dt['state_dict'])
    optimizer.load_state_dict(dt['optimizer'])
    

# Song-wise genre for submission with prediction method v1
def save_csv_pred_v1(probs, track_ids, classnames, expname):
    probs = F.softmax(Variable(torch.FloatTensor(probs), volatile=True)).data.numpy()
    
    #  Parts: first index of each song(= can be a track or partition),  n_seg: number of segments in each song
    _, parts, n_seg = np.unique(track_ids, return_index=True, return_counts=True)
    n_parts= len(parts)
    track_ids_unique = track_ids[parts] #---> save
    #pred_by_tracks = []
    
    # Get probs of each partition --> song-wise genre decision
    track_probs = np.zeros((n_parts, 16)) #--> save
    for i in range(n_parts):
        tmp = probs[ parts[i]:parts[i]+n_seg[i], : ]
        track_prob_sum = np.sum(tmp, axis=0)
        track_probs[i, :] = track_prob_sum
        #pred_by_tracks.append(np.argmax(track_prob_sum))
    
    # Normalize each rows of prob 
    row_sums = track_probs.sum(axis=1, keepdims=True)
    track_probs /= row_sums
    
    # Generate track probs using track_probs, track_ids_unique, classnames 
    import pandas as pd
    df_out = pd.DataFrame(track_probs, pd.Index(track_ids_unique, name='file_id'), classnames)
    
    # Save as checkpoints/EXP_NAME/submisssion_v1.csv
    #csv_filepath = SAVE_MODEL_DIR + expname + '/submission_v1.csv'
    csv_filepath = OUTPUT_PATH
    df_out.to_csv(csv_filepath, encoding='utf-8')
        
   
        

# Song-wise genre prediction V1: softmax --> group_sum --> argmax
def prediction_method_v1(probs, track_ids, labels, display=True):
    probs = F.softmax(torch.FloatTensor(probs)).data.numpy()
    
    #  Parts: first index of each song(= can be a track or partition),  n_seg: number of segments in each song
    _, parts, n_seg = np.unique(track_ids, return_index=True, return_counts=True)
    n_parts= len(parts)
    pred_by_tracks = []
    
    # Get probs of each partition --> song-wise genre decision
    track_probs = np.zeros((n_parts, 16))
    for i in range(n_parts):
        tmp = probs[ parts[i]:parts[i]+n_seg[i], : ]
        track_prob_sum = np.sum(tmp, axis=0)
        track_probs[i, :] = track_prob_sum
        pred_by_tracks.append(np.argmax(track_prob_sum))
    
    # Prepare labels by tracks
    label_by_tracks = labels[parts]
    
    # Eval1: Acc
    correct = np.sum(pred_by_tracks == label_by_tracks)
    song_acc = correct/np.float(n_parts)  
    
    # Eval2: MLL
    from sklearn.metrics import log_loss, f1_score
    logloss = log_loss(label_by_tracks, track_probs)
    f1 = f1_score(label_by_tracks, pred_by_tracks, average='macro')
    print('LogLoss: {}'.format(logloss))
    print('f1_score: {}'.format(f1))
    
    # Compute confusion matrix
    from sklearn.metrics import confusion_matrix
    cnf_mtx = confusion_matrix(label_by_tracks, pred_by_tracks)
    
    if display:
        print('Song-wise genre prediction V1: {}/{} = {:.6f}%'.format(correct, n_parts, 100.*song_acc))
    return correct, song_acc, pred_by_tracks, cnf_mtx, logloss, f1


# Song-wise genre prediction V2: no softmax --> group_sum --> argmax
def prediction_method_v2(probs, track_ids, labels, display=True):
    #probs = F.softmax(torch.FloatTensor(probs)).data.numpy()
    
    #  Parts: first index of each song(= can be a track or partition),  n_seg: number of segments in each song
    _, parts, n_seg = np.unique(track_ids, return_index=True, return_counts=True)
    n_parts= len(parts)
    pred_by_tracks = []
    
    # Get probs of each partition --> song-wise genre decision 
    for i in range(n_parts):
        tmp = probs[ parts[i]:parts[i]+n_seg[i], : ]
        track_prob_sum = np.sum(tmp, axis=0)
        pred_by_tracks.append(np.argmax(track_prob_sum))
    
    # Prepare labels by tracks
    label_by_tracks = labels[parts]
    
    # Eval
    correct = np.sum(pred_by_tracks == label_by_tracks)
    song_acc = correct/np.float(n_parts)  
    
    # Compute confusion matrix
    from sklearn.metrics import confusion_matrix
    cnf_mtx = confusion_matrix(label_by_tracks, pred_by_tracks)
    
    if display:
        print('Song-wise genre prediction V2: {}/{} = {:.6f}%'.format(correct, n_parts, 100.*song_acc))
    return correct, song_acc, pred_by_tracks, cnf_mtx




def history_recorder(exp_name, epoch,
                     tr_loss, tr_seg_acc, val_loss, val_seg_acc,
                     val_acc_v1, val_acc_v2,
                     cnf_mtx_v1, cnf_mtx_v2, logloss, f1, class_names=None, ):
    import pandas as pd
    
    # Path and directories
    exp_weights_root_dir = SAVE_MODEL_DIR + exp_name + '/'
    os.makedirs(exp_weights_root_dir, exist_ok=True)
    # History filepath
    filepath_hs = exp_weights_root_dir + '{}_history.csv'.format(exp_name)
    # Confusion matrix filepath
    filepath_cnf = exp_weights_root_dir + '{}_confusion.csv'.format(exp_name)
    filepath_cnf_png = exp_weights_root_dir + '{}_confusion.png'.format(exp_name)
    
    if os.path.isfile(filepath_hs) & (epoch>1):
        df = pd.read_csv(filepath_hs, index_col=0)  # load old history
        print('history_recorder: adding history to existing file, {}'.format(filepath_hs))
    else:
        df = pd.DataFrame(columns=('tr_loss', 'tr_seg_acc', 
                                   'val_loss', 'val_seg_acc', 'val_acc_v1', 'val_acc_v2',
                                   'val_logloss', 'val_f1','submission num','LL','F'))
        print('history_recorder: creating new history to {}'.format(filepath_hs))

    df.loc[len(df)] = [tr_loss, tr_seg_acc, val_loss, val_seg_acc, val_acc_v1, val_acc_v2, logloss, f1,'','','']


    # Save history as both .csv and image
    df.to_csv(filepath_hs, encoding='utf-8')
    save_img_from_history(df, exp_weights_root_dir, exp_name)
    
    # Save confusion matrix as both .csv
    norm_cnf_mtx_v2 = cnf_mtx_v2.astype('float') / cnf_mtx_v2.sum(axis=1)[:, np.newaxis] # normalized confusion
    
    if class_names is not None:
        cnf_df = pd.DataFrame(columns=class_names, index=class_names, data=cnf_mtx_v2)
        norm_cnf_df = pd.DataFrame(columns=class_names, index=class_names, data=norm_cnf_mtx_v2)
    else:
        cnf_df = pd.DataFrame(data=cnf_mtx_v2)
        norm_cnf_df = pd.DataFrame(data=norm_cnf_mtx_v2) 
    cnf_df.to_csv(filepath_cnf, encoding='utf-8')
    #norm_cnf_df.to_csv(filepath_cnf, encoding='utf-8')
    
    # Save confusion matrix as image
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams["figure.figsize"] = [10,4]
    sns.set(font_scale=0.4)
    plt.subplot(121)
    ax = sns.heatmap(cnf_df, annot=True, fmt="g")
    plt.subplot(122)
    ax = sns.heatmap(norm_cnf_df, annot=True, fmt=".2f")
    plt.title('y-axis = target, x-axis = prediction')
    plt.savefig(filepath_cnf_png, bbox_inches='tight', dpi=220)
    del(ax)
    plt.close('all')
    


def save_img_from_history(df, outdir, exp_name):
    import matplotlib.pyplot as plt
    plt.ioff()
    plt.rcParams.update(plt.rcParamsDefault)
    # plotting tr_loss with val_loss
    ax = df.iloc[:,[0,2]].plot(grid=True)
    ax.set_xlabel('epoch'); ax.set_ylabel('loss')
    ax.legend(loc='best')
    ax.get_figure().savefig(outdir + '/' + exp_name + '_loss.png', bbox_inches='tight', dpi=200)
    
    # plotting tr_acc with val_acc
    ax = df.iloc[:,[1,3,4,5,6,7]].plot()
    ax.set_xlabel('epoch'); ax.set_ylabel('accuracy')
    ax.legend(loc='best')
    ax.get_figure().savefig(outdir + '/' + exp_name + '_acc.png', bbox_inches='tight', dpi=200)
    plt.close('all')
    
    


    
    

#%% Run Test
    
if args.load is not None:
    load_checkpoint(args.load)
else:
    print('Please load model by --load command')
    
#%%   
test_loader = DataLoader(dset_test,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=8,
                         pin_memory=True
                         )
 
test() # test and save .csv file to checkcpoint directory

print('Test finished: output file is created in {}'.format(OUTPUT_PATH))
#%% 





