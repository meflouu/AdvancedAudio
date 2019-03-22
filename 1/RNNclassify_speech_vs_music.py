from __future__ import print_function
import os
import numpy as np
import librosa
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Input, GRU
from keras.layers.core import Dense, Activation, Dropout, Reshape, Permute
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
import sklearn.metrics as metrics
import matplotlib.pyplot as plot
import matplotlib.colors as colors
import matplotlib
matplotlib.rcParams.update({'font.size': 20})

from IPython import embed
K.set_image_data_format('channels_first')


def split_in_seqs(data, subdivs):
    """
    Splits a long sequence matrix into sub-sequences.
        Eg: input: data = MxN  sub-sequence length (subdivs) = 2
            output = M/2 x 2 x N

    :param data: Array of one or two dimensions
    :param subdivs: integer value representing a sub-sequence length
    :return: array of dimension = input array dimension + 1
    """
    if len(data.shape) == 1:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :]
        data = data.reshape((int(data.shape[0]/subdivs), subdivs, 1))
    elif len(data.shape) == 2:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :]
        data = data.reshape((int(data.shape[0]/subdivs), subdivs, data.shape[1]))
    return data

def split_multi_channels(data, num_channels):
    """
    Split features into multiple channels
        Eg: input: data = MxNxP  num_channels = 2
        output = M x 2 x N x P/2

    :param data: 3-D array
    :param num_channels: integer value representing the number of channels
    :return: array of dimension = input array dimension + 1
    """
    in_shape = data.shape
    if len(in_shape) == 3:
        hop = int(in_shape[2] / num_channels)
        tmp = np.zeros((in_shape[0], num_channels, in_shape[1], hop))
        for i in range(num_channels):
            tmp[:, i, :, :] = data[:, :, i*hop:(i+1)*hop]
        return tmp
    else:
        print("ERROR: The input should be a 3D matrix but it seems to have dimensions ", in_shape)
        exit()


def get_rnn_model(in_data, out_data):
    #TODO: implement your RNN model here

    mel_start = input(shape=( in_data.shape[-2] , in_data.shape[-1] ))
    out = None

    mel_x = GRU(32, Dropout=0.25, return_sequences = True)(mel_start)

    # leave the following unchanged
    _model = Model(inputs=mel_start, outputs=out)
    _model.compile(optimizer='Adam', loss='binary_crossentropy')
    _model.summary()
    return _model

def get_input_output_file_names(_window_length, _nb_mel_bands, _nb_frames):
    input_feat_name = 'speech_vs_music_{}_{}_{}.npz'.format(_nb_frames, _nb_mel_bands, _window_length)
    model_name = input_feat_name.replace('.npz', '_model.h5')
    test_filelist_name = input_feat_name.replace('.npz', '_filenames.npy')
    results_csv_name = input_feat_name.replace('npz', 'csv')
    print('input_feat_name: {}\n model_name: {}\nresults_csv_name: {}\ntest_filelist_name: {}'.format(
        input_feat_name, model_name, results_csv_name, test_filelist_name))
    return input_feat_name, model_name, results_csv_name, test_filelist_name


def load_feat(_input_feat_name, _nb_frames):
    # Load normalized features and pre-process them - splitting into sequence
    dmp = np.load(_input_feat_name)

    # TODO: Change the data pre-processing based on the GRU requirements. Check the definition in the website. See what should be the input and output format
    train_data = split_in_seqs(dmp['arr_0'], nb_frames)[:, 0]
    train_labels = split_in_seqs(dmp['arr_1'], nb_frames)[:, 0]
    test_data = split_in_seqs(dmp['arr_2'], nb_frames)[:, 0]
    test_labels = dmp['arr_3']
    # test_labels = split_in_seqs(dmp['arr_3'], nb_frames)[:, 0]

    test_labels_recording = split_in_seqs(dmp['arr_3'], nb_frames)[:, 0]

    return train_data, train_labels, test_data, test_labels, test_labels_recording


def train(_window_length, _nb_mel_bands, _nb_frames):
    # Initialize filenames
    input_feat_name, model_name, results_csv_name, test_filelist_name = \
        get_input_output_file_names(_window_length, _nb_mel_bands, _nb_frames)

    # Load data
    train_data, train_labels, test_data, test_labels, test_labels_recording = load_feat(input_feat_name, _nb_frames)

    # Load test data file names
    test_filelist = np.load(test_filelist_name)

    # Load the RNN model
    model = get_rnn_model(train_data, train_labels)

    nb_epoch = 300      # Maximum number of epochs for training
    batch_size = 16     # Batch size

    patience = int(0.25 * nb_epoch) # We stop training if the accuracy does not improve for 'patience' number of epochs
    patience_cnt = 0    # Variable to keep track of the patience

    best_accuracy = -999    # Variable to save the best accuracy of the model
    best_epoch = -1     # Variable to save the best epoch of the model
    train_loss = [0] * nb_epoch  # Variable to save the training loss of the model per epoch
    framewise_test_accuracy = [0] * nb_epoch  # Variable to save the training accuracy of the model per epoch
    recording_test_accuracy = [0] * nb_epoch  # Variable to save the training accuracy of the model per epoch

    # Training begins
    for i in range(nb_epoch):
        print('Epoch : {} '.format(i), end='')

        # Fit model for one epoch
        hist = model.fit(
            train_data,
            train_labels,
            batch_size=batch_size,
            epochs=1
        )

        # save the training loss for the epoch
        train_loss[i] = hist.history.get('loss')[-1]

        # Use the trained model on test data
        pred = model.predict(test_data, batch_size=batch_size)

        # Calculate the accuracy on the test data
        framewise_test_accuracy[i] = metrics.accuracy_score(test_labels, pred.reshape(-1) > 0.5)
        recording_pred = np.mean(pred, 1)
        recording_test_accuracy[i] = metrics.accuracy_score(test_labels_recording, recording_pred > 0.5)
        patience_cnt = patience_cnt + 1

        # Check if the test_accuracy for the epoch is better than the best_accuracy
        if framewise_test_accuracy[i] > best_accuracy:
            # Save the best accuracy and its respective epoch
            best_accuracy = framewise_test_accuracy[i]
            best_epoch = i
            patience_cnt = 0

            # Save the best model
            model.save('{}'.format(model_name))

            # Write the results of the best model to a file
            fid = open(results_csv_name, 'w')
            fid.write('{},{},{},{},{}\n'.format(
                'Index', 'Test file name', 'Groundtruth: music = 1 and speech=0', 'Predictions: music = 1 and speech=0',
                '% of music frames: closer to 1 means mostly music and 0 means mostly speech'))
            for cnt, test_file in enumerate(test_filelist):
                fid.write('{},{},{},{},{}\n'.format(cnt, test_file, int(test_labels_recording[cnt, 0]), int(recording_pred[cnt, 0] > 0.5), recording_pred[cnt, 0]))
            fid.close()

        print('framewise_accuracy: {}, recording_accuracy: {}, best framewise accuracy: {}, best epoch: {}'.format(framewise_test_accuracy[i], recording_test_accuracy[i], best_accuracy, best_epoch))

        # Early stopping, if the test_accuracy does not change for 'patience' number of epochs then we quit training
        if patience_cnt > patience:
            break

    print('The best_epoch: {} with best framewise accuracy: {}'.format(best_epoch, best_accuracy))


def test(_window_length, _nb_mel_bands, _nb_frames, test_file_index):
    # Initialize filenames
    input_feat_name, model_name, results_csv_name, test_filelist_name = \
        get_input_output_file_names(_window_length, _nb_mel_bands, _nb_frames)

    # Load data
    train_data, train_labels, test_data, test_labels, test_labels_recording = load_feat(input_feat_name, _nb_frames)

    # Load test data file names
    test_filelist = np.load(test_filelist_name)

    # Load trained model
    model = load_model(model_name)

    # Choose the feature for input file and format it to right dimension
    test_feat = test_data[test_file_index][np.newaxis]

    # predict the class using trainedmodel
    pred = model.predict(test_feat)
    pred = np.squeeze(pred)

    # Load audio file to extract spectrogram. This is done here only to visualize the audio.
    if test_file_index < len(test_filelist)/2:
        test_audio_filename = os.path.join(__music_audio_folder, test_filelist[test_file_index])
    else:
        test_audio_filename = os.path.join(__speech_audio_folder, test_filelist[test_file_index])

    y, sr = librosa.load(test_audio_filename)
    stft = librosa.stft(y, n_fft=_window_length, hop_length=int(_window_length/2), win_length=_window_length)
    stft = np.abs(stft[:500, :nb_frames])**2  # visualizing only the first 500 bins and nb_frames

    # Visualize the spectrogram and model outputs
    time_vec = np.arange(stft.shape[1])*_window_length/(2.0*sr)
    plot.figure()
    plot.subplot(211), plot.pcolormesh(time_vec, range(stft.shape[0]), stft, norm=colors.PowerNorm(0.25, vmin=stft.min(), vmax=stft.max())), plot.title('Spectrogram for {}'.format(test_filelist[test_file_index]))
    plot.xlabel('Time'), plot.ylabel('Spectral bins')
    plot.subplot(212), plot.plot(time_vec, pred, label='RNN output'), plot.plot(time_vec, 0.5*np.ones(pred.shape), label='Threshold'),
    plot.xlabel('Time'), plot.ylabel('RNN output magnitude')
    plot.grid(True), plot.ylim([-0.1, 1.2]), plot.title('RNN model output')
    plot.legend()
    plot.show()


# -------------------------------------------------------------------
#              Main script starts here
# -------------------------------------------------------------------

# location of data. #TODO: UPDATE ACCORDING TO YOUR SYSTEM PATH
__speech_audio_folder = '~/Desktop/4thPeriod/AdvancedAudio/Ex/1/data/speech_wav'
__music_audio_folder = '~/Desktop/4thPeriod/AdvancedAudio/Ex/1/data/music.wav'

window_length = 2048
nb_mel_bands = 32
nb_frames = 80

# Step 1: Run the following code once to train the model and save it. After training completes you can can comment it
train(window_length, nb_mel_bands, nb_frames)

# Step 2: After running step 1 and commenting it, check the CSV file. To analyse why you are getting the good or bad
# results for a recording, note the 'Index' number from the CSV file for the respective audio you want to analyze and
# use it below to visualize the outputs.

#file_index_in_csv_file = 12
#test(window_length, nb_mel_bands, nb_frames, file_index_in_csv_file)
