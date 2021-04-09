#https://github.com/priya-dwivedi/Music_Genre_Classification/blob/master/convert_to_npz.ipynb

import os
import numpy as np
import librosa
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt


AUDIO_DIR = '/home/sourjyadip/iitb/fma_small' #change
filepath = '/home/sourjyadip/iitb/genre_class/fma_metadata/tracks.csv' #change

def create_spectogram(track_id):
    filename = get_audio_path(AUDIO_DIR, track_id)
    y, sr = librosa.load(filename)
    spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=1024)
    spect = librosa.power_to_db(spect, ref=np.max)
    return spect.T

def get_tids_from_directory(audio_dir):
    """Get track IDs from the mp3s in a directory
    Parameters
    ----------
    audio_dir : str
        Path to the directory where the audio files are stored.
    Returns
    -------
        A list of track IDs.
    """
    tids = []
    for _, dirnames, files in os.walk(audio_dir):
        if dirnames == []:
            tids.extend(int(file[:-4]) for file in files)
    return tids

def get_audio_path(audio_dir, track_id):
    """
    Return the path to the mp3 given the directory where the audio is stored
    and the track ID.
    Examples
    --------
    >>> import utils
    >>> AUDIO_DIR = os.environ.get('AUDIO_DIR')
    >>> utils.get_audio_path(AUDIO_DIR, 2)
    '../data/fma_small/000/000002.mp3'
    """
    tid_str = '{:06d}'.format(track_id)
    print(tid_str)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')

tids = get_tids_from_directory(AUDIO_DIR)
print(len(tids))


tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])
keep_cols = [('set', 'split'),
('set', 'subset'),('track', 'genre_top')]

df_all = tracks[keep_cols]
df_all = df_all[df_all[('set', 'subset')] == 'small']

df_all['track_id'] = df_all.index

print(df_all.shape)

print(df_all[('track', 'genre_top')].unique())
dict_genres = {'Electronic':1, 'Experimental':2, 'Folk':3, 'Hip-Hop':4,'Instrumental':5,'International':6, 'Pop' :7, 'Rock': 8  }

def create_array(df):
    genres = []
    X_spect = np.empty((0, 640, 128))
    count = 0
    #Code skips records in case of errors
    for index, row in df.iterrows():
        try:
            count += 1
            track_id = int(row['track_id'])
            genre = str(row[('track', 'genre_top')]) 
            spect = create_spectogram(track_id)

            # Normalize for small shape differences
            spect = spect[:640, :]
            X_spect = np.append(X_spect, [spect], axis=0)
            genres.append(dict_genres[genre])
            if count % 100 == 0:
                print("Currently processing: ", count)
        except:
            print("Couldn't process: ", count)
            continue
    y_arr = np.array(genres)
    return X_spect, y_arr

print(df_all[('set', 'split')].unique())

df_train = df_all[df_all[('set', 'split')]=='training']
df_valid = df_all[df_all[('set', 'split')]=='validation']
df_test = df_all[df_all[('set', 'split')]=='test']

print(df_train.shape, df_valid.shape, df_test.shape)
print(df_train)


'''
X_test, y_test = create_array(df_test)

print(X_test.shape, y_test.shape)
print(y_test[:23])

np.savez('test_arr', X_test, y_test)

X_valid, y_valid = create_array(df_valid)

np.savez('valid_arr', X_valid, y_valid)


#Train data split into 4 chunks to do the slow pre-processing in phases

def splitDataFrameIntoSmaller(df, chunkSize = 1600): 
    listOfDf = list()
    numberChunks = len(df) // chunkSize + 1
    for i in range(numberChunks):
        listOfDf.append(df[i*chunkSize:(i+1)*chunkSize])
    return listOfDf

listDf = splitDataFrameIntoSmaller(df_train)
df1_train = listDf[0]
df2_train = listDf[1]
df3_train = listDf[2]
df4_train = listDf[3]
print(df1_train.shape, df2_train.shape, df3_train.shape, df4_train.shape)

X_train1, y_train1 = create_array(df1_train)
np.savez('train1_arr', X_train1, y_train1)

X_train2, y_train2 = create_array(df2_train)
np.savez('train2_arr', X_train2, y_train2)

X_train3, y_train3 = create_array(df3_train)
print(X_train3.shape, y_train3.shape)
np.savez('train3_arr', X_train3, y_train3)

X_train4, y_train4 = create_array(df4_train)
np.savez('train4_arr', X_train4, y_train4)
print(X_train4.shape, y_train4.shape)
'''
#Concatenate and shuffle data

npzfile = np.load('train1_arr.npz')
print(npzfile.files)
X_train1 = npzfile['arr_0']
y_train1 = npzfile['arr_1']
print(X_train1.shape, y_train1.shape)

npzfile = np.load('train2_arr.npz')
print(npzfile.files)
X_train2 = npzfile['arr_0']
y_train2 = npzfile['arr_1']
print(X_train2.shape, y_train2.shape)

npzfile = np.load('train3_arr.npz')
print(npzfile.files)
X_train3 = npzfile['arr_0']
y_train3 = npzfile['arr_1']
print(X_train3.shape, y_train3.shape)

npzfile = np.load('train4_arr.npz')
print(npzfile.files)
X_train4 = npzfile['arr_0']
y_train4 = npzfile['arr_1']
print(X_train4.shape, y_train4.shape)

#Concatenate train data

X_train = np.concatenate((X_train1, X_train2, X_train3, X_train4), axis = 0)
y_train = np.concatenate((y_train1, y_train2, y_train3, y_train4), axis = 0)
print(X_train.shape, y_train.shape)

## Convert y data from scale 0-7
print(np.amin(X_train), np.amax(X_train), np.mean(X_train))
y_train = y_train -1
y_valid = y_valid -1
print(np.amin(y_train), np.amax(y_train), np.mean(y_train))

### Convert the scale of training data
X_train_raw = librosa.core.db_to_power(X_train, ref=1.0)
print(np.amin(X_train_raw), np.amax(X_train_raw), np.mean(X_train_raw))

X_train_log = np.log(X_train_raw)
print(np.amin(X_train_log), np.amax(X_train_log), np.mean(X_train_log))

X_valid_raw = librosa.core.db_to_power(X_valid, ref=1.0)
X_valid_log = np.log(X_valid_raw)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

X_train, y_train = unison_shuffled_copies(X_train_log, y_train)
X_valid, y_valid = unison_shuffled_copies(X_valid_log, y_valid)

print("Shapes are: ", X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

np.savez('shuffled_train', X_train, y_train)
np.savez('shuffled_valid', X_valid, y_valid)
