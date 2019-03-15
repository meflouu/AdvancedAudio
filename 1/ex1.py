import scipy.io.wavfile as wav
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plot
import librosa
import matplotlib.colors as colors


def get_ZCR(audio_window):
    """
    Number of zero crossings in a window of audio

    :param audio_window: window of audio samples
    :return: zcr: integer value representing number of zero crossings
    """
    zcr = 0

    # TODO: Implement ZCR
    for i in np.arange(audio_window.size-1):
        zcr=zcr+int(np.floor(np.abs(np.sign(audio_window[i+1])-np.sign(audio_window[i]))/2))
        if (audio_window[i+1]==1):
            zcr=zcr+1


    return zcr


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


def extract_feature(_audio_filename):

    # User set parameters
    nfft = 2048
    win_len = nfft
    hop_len = win_len / 2
    window = np.hamming(win_len)
    nb_mel_bands = 40

    # load audio
    _y, _fs = load_audio(_audio_filename)

    audio_length = len(_y)
    nb_frames = int(np.floor((audio_length - win_len) / float(hop_len)))

    # Precompute FFT to mel band conversion matrix
    #fft_mel_bands = np.ones((1+nfft/2, nb_mel_bands))  # TODO: Q1. Replace this line to get FFT to mel band conversion matrix # Hint: Check librosa.filters package
    fft_mel_bands = librosa.filters.mel(_fs,nfft,nb_mel_bands)


    #_fft_en = np.zeros((nb_frames, 1+nfft/2))
    _fft_en = np.zeros((nb_frames, int(1+nfft/2)))
    _zcr = np.zeros((nb_frames, 1))
    _mbe = np.zeros((nb_frames, nb_mel_bands))

    frame_cnt = 0
    for i in range(nb_frames):

        # framing and windowing
        #y_win = _y[i*hop_len:i*hop_len+win_len] * window
        y_win = _y[int(i*hop_len):int(i*hop_len+win_len)] * window

        # extract ZCR
        _zcr[frame_cnt] = get_ZCR(y_win)  # TODO: Q2. Implement ZCR, check the function

        # calculate energy spectral density (ESD)
        #_fft_en[frame_cnt, :] = np.ones((1 + nfft/2))  # TODO: Q3. Calculate energy spectral density, you can use scipy.fftpack
        _fft_en[frame_cnt,  :] = np.abs(fft(y_win,1025))**2

        # calculate mel band energy (MBE)
        #_mbe[frame_cnt, :] = np.ones(nb_mel_bands)  # TODO: Q4. Calculate mel band energy
        _mbe[frame_cnt,  :] = np.dot(fft_mel_bands,_fft_en[frame_cnt, :])

        frame_cnt = frame_cnt + 1
    return _y, _zcr, _fft_en, _mbe


audio_filename = 'ex1.wav'
y, zcr, fft_en, mbe = extract_feature(audio_filename)

# PLOTTING
plot.subplot(411), plot.plot(y), plot.xlim((0, len(y))), plot.xlabel('Audio')
plot.subplot(412), plot.imshow(fft_en[:, :100].T, aspect='auto', origin='lower', norm=colors.LogNorm(vmin=fft_en.min(), vmax=fft_en.max()))
plot.xlabel('Energy spectral density (ESD)')
plot.subplot(413), plot.imshow(mbe.T, aspect='auto', origin='lower', norm=colors.LogNorm(vmin=mbe.min(), vmax=mbe.max()))
plot.xlabel('Mel-band energy (MBE)')
plot.subplot(414), plot.plot(zcr), plot.xlim((0, len(zcr))), plot.grid(True)
plot.xlabel('ZCR')
plot.show()
