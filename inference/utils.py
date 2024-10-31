import numpy as np

def mono_check(audio):
    '''
    Check whether the audio is mono since our system only support mono audio
    :param audio: audio matrix with shape: channel x frames  (numpy array or tensor)
    :return: audio matrix with shape: 1 x frames
    '''
    if audio.shape[0] == 1:
        return audio
    # preprocess stereo
    elif audio.shape[0] == 2:
        audio = audio.sum(0)[None, :] / 2
    # preprocess 5.1
    elif audio.shape[0] == 6:
        L = audio[0]
        R = audio[1]
        C = audio[2]
        Ls = audio[4]
        Rs = audio[5]
        Lt = L + 0.707 * Ls + C * 0.707
        Rt = R + 0.707 * Rs + C * 0.707
        audio = np.divide(Lt + Rt, 2)[None, :]
    else:
        raise Exception("The input audio format is not currently support")
    # To Do: add more channel support
    return audio