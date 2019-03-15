import scipy.io.wavfile as wav
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plot
import librosa
import matplotlib.colors as colors
import os
from sklearn import preprocessing
import random
random.seed(12345)



def load_audio(_audio_filename):
    """
    Load audio file

    :param _audio_filename:
    :return: _y: audio samples
    :return: _fs: sampling rate
    """
    _fs, _y = wav.read(_audio_filename)
    if _y.dtype == np.int16:
        _y = _y / 32768.0  # short int to float
    return _y, _fs


def extract_feature(_audio_filename, nb_mel_bands, nb_frames, nfft):
    # User set parameters
    win_len = nfft
    hop_len = win_len / 2
    window = np.hamming(win_len)
    # nb_mel_bands = 40

    # load audio
    _y, _fs = load_audio(_audio_filename)

    # audio_length = len(_y)
    # nb_frames = int(np.floor((audio_length - win_len) / float(hop_len)))

    # Precompute FFT to mel band conversion matrix
    #fft_mel_bands = np.ones((1+nfft/2, nb_mel_bands)) ## TODO : Take help from your code in part 1
    fft_mel_bands = librosa.filters.mel(_fs,nfft,nb_mel_bands)

    _mbe = np.zeros((nb_frames, nb_mel_bands))
    _fft_en = np.zeros((nb_frames,hop_len+1))

    frame_cnt = 0
    for i in range(nb_frames):
        # framing and windowing
        y_win = _y[i * hop_len:i * hop_len + win_len] * window

        # calculate energy spectral density
        #_fft_en = np.abs(fft(y_win)[:1 + nfft / 2]) ** 2
        _fft_en[frame_cnt, :] = np.abs(fft(y_win,1025))**2

        # calculate mel band energy
        #_mbe[frame_cnt, :] = np.ones(nb_mel_bands) ## TODO: Take help from your code in part 1
        _mbe[frame_cnt, :] = np.dot(fft_mel_bands,_fft_en[frame_cnt, :])

        frame_cnt = frame_cnt + 1
    return _mbe

# -------------------------------------------------------------------
#              Main script starts here
# -------------------------------------------------------------------


# TODO: Tune the following three parameters only
window_length = 2048  # Window length in samples
nb_mel_bands = 32  # Number of Mel-bands to calculate Mel band energy feature in
nb_frames = 80  # Extracts max_nb_frames frames of features from the audio, and ignores the rest.
#  For example when max_mb_frames = 40, the script extracts features for the first 40 frames of audio
#  Where each frame is of length as specified by win_len variable in extract_feature() function


output_feat_name = 'speech_vs_music_{}_{}_{}.npz'.format(nb_frames, nb_mel_bands, window_length)
print('output_feat_name: {}'.format(output_feat_name))

# location of data. #TODO: UPDATE ACCORDING TO YOUR SYSTEM PATH
speech_audio_folder = 'data/speech_wav/'
music_audio_folder = 'data/music_wav/'

speech_files = os.listdir(speech_audio_folder)
music_files = os.listdir(music_audio_folder)

# Generate training and testing splits
training_ratio = 0.8  # 80% files for training
nb_train_files = int(len(speech_files) * training_ratio)    # The number of files for speech and music or the same in this dataset. Hence we do this only once.
nb_test_files = len(speech_files) - nb_train_files

random.shuffle(speech_files)
speech_train_files = speech_files[:nb_train_files]
speech_test_files = speech_files[nb_train_files:]

random.shuffle(music_files)
music_train_files = music_files[:nb_train_files]
music_test_files = music_files[nb_train_files:]

# Extract training features
speech_train_data = np.zeros((nb_train_files * nb_frames, nb_mel_bands))
music_train_data = np.zeros((nb_train_files * nb_frames, nb_mel_bands))
for ind in range(nb_train_files):
    music_train_data[ind * nb_frames:ind * nb_frames + nb_frames] = extract_feature(
        os.path.join(music_audio_folder, music_train_files[ind]),
        nb_mel_bands,
        nb_frames,
        window_length
    )
    speech_train_data[ind * nb_frames:ind * nb_frames + nb_frames] = extract_feature(
        os.path.join(speech_audio_folder, speech_train_files[ind]),
        nb_mel_bands,
        nb_frames,
        window_length
    )

# Extract testing features
speech_test_data = np.zeros((nb_test_files * nb_frames, nb_mel_bands))
music_test_data = np.zeros((nb_test_files * nb_frames, nb_mel_bands))
for ind in range(nb_test_files):
    music_test_data[ind * nb_frames:ind * nb_frames + nb_frames] = extract_feature(
        os.path.join(music_audio_folder, music_test_files[ind]),
        nb_mel_bands,
        nb_frames,
        window_length
    )
    speech_test_data[ind * nb_frames:ind * nb_frames + nb_frames] = extract_feature(
        os.path.join(speech_audio_folder, speech_test_files[ind]),
        nb_mel_bands,
        nb_frames,
        window_length
    )

# Plotting function to visualize training and testing data before normalization
plot.figure()
plot.subplot2grid((2, 3), (0, 0), colspan=2), plot.imshow(speech_train_data.T, aspect='auto', origin='lower', norm=colors.LogNorm(vmin=speech_train_data.min(), vmax=speech_train_data.max()))
plot.title('TRAINING DATA')
plot.xlabel('SPEECH - frames')
plot.ylabel('mel-bands')
plot.subplot2grid((2, 3), (0, 2)), plot.plot(np.mean(speech_train_data, 0))
plot.xlabel('mel-bands')
plot.ylabel('mean magnitude')
plot.subplot2grid((2, 3), (1, 0), colspan=2), plot.imshow(music_train_data.T, aspect='auto', origin='lower', norm=colors.LogNorm(vmin=music_train_data.min(), vmax=music_train_data.max()))
plot.xlabel('MUSIC - frames')
plot.ylabel('mel-bands')
plot.subplot2grid((2, 3), (1, 2)), plot.plot(np.mean(music_train_data, 0))
plot.xlabel('mel-bands')
plot.ylabel('mean magnitude')

plot.figure()
plot.subplot2grid((2, 3), (0, 0), colspan=2), plot.imshow(speech_test_data.T, aspect='auto', origin='lower', norm=colors.LogNorm(vmin=speech_test_data.min(), vmax=speech_test_data.max()))
plot.title('TESTING DATA')
plot.xlabel('SPEECH - frames')
plot.ylabel('mel-bands')
plot.subplot2grid((2, 3), (0, 2)), plot.plot(np.mean(speech_test_data, 0))
plot.xlabel('mel-bands')
plot.ylabel('mean magnitude')
plot.subplot2grid((2, 3), (1, 0), colspan=2), plot.imshow(music_test_data.T, aspect='auto', origin='lower', norm=colors.LogNorm(vmin=music_test_data.min(), vmax=music_test_data.max()))
plot.xlabel('MUSIC - frames')
plot.ylabel('mel-bands')
plot.subplot2grid((2, 3), (1, 2)), plot.plot(np.mean(music_test_data, 0))
plot.xlabel('mel-bands')
plot.ylabel('mean magnitude')


# Concatenate speech and music data into training and testing data
train_data = np.concatenate((music_train_data, speech_train_data), 0)
test_data = np.concatenate((music_test_data, speech_test_data), 0)

# Labels for training and testing data
train_labels = np.concatenate((np.ones(music_train_data.shape[0]), np.zeros(speech_train_data.shape[0])))
test_labels = np.concatenate((np.ones(music_test_data.shape[0]), np.zeros(speech_test_data.shape[0])))

# Normalize the training data, and scale the testing data using the training data weights
scaler = preprocessing.StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Save labels and the normalized features
np.savez(output_feat_name, train_data, train_labels, test_data, test_labels)
print('output_feat_name: {}'.format(output_feat_name))
plot.show()
