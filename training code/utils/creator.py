from dataloader import avaspeech_dataset, muspeak_dataset, netflix_dataset, klaus_dataset, netflix_eval_dataset, netflix_whole_dataset, bmat_dataset
from model import tcn, RNN, CRNN
from model import preprocess
from utils import pcen

# choose which dataset to use
def dataset_creator(hparams, partition):
    dataset = hparams.dataset
    data_path = hparams.data_path
    duration = hparams.duration
    sr = hparams.sr
    hop_size = hparams.hop_size
    n_sample = hparams.n_sample
    features = hparams.input_features
    n_fft = hparams.n_fft
    if dataset == 'avaspeech':
        return avaspeech_dataset.AVASpeechDataset(data_path, partition, duration=duration, sr=sr, hop_size=hop_size, n_sample=n_sample)
    elif dataset == 'muspeak':
        return muspeak_dataset.MuspeakDataset(data_path, partition, duration=duration, sr=sr, hop_size=hop_size, n_sample=n_sample)
    elif dataset == 'netflix':
        return netflix_dataset.NetflixDataset(data_path, partition, duration=duration, sr=sr, hop_size=hop_size, n_sample=n_sample, features=features)
    elif dataset == 'klaus':
        return klaus_dataset.KlausDataset(data_path, partition, duration=duration, sr=sr, hop_size=hop_size, n_sample=n_sample)
    elif dataset == 'openbat':
        return bmat_dataset.BMATDataset(data_path, partition, duration=duration, sr=sr, hop_size=hop_size, n_sample=n_sample, n_fft=n_fft)
    elif dataset == 'netflix_eval':
        return netflix_eval_dataset.NetflixEvalDataset(data_path, partition, duration=duration, sr=sr, hop_size=hop_size, n_sample=n_sample)
    elif dataset == 'netflix_whole':
        return netflix_whole_dataset.NetflixWholeDataset(data_path, partition, duration=duration, sr=sr, hop_size=hop_size, n_sample=n_sample, features=features)
    else:
        raise ValueError('The dataset you choose is not supported!')

# choose which model to use
def model_creator(hparams, n_inp, n_oup, model_choose='tcn'):
    if 'rnn' == model_choose:
        model = RNN.RNN(n_inp, n_oup, hparams.rnn_layers, hparams.n_hidden)
    elif 'tcn' == model_choose:
        model = tcn.tcn(n_inp, hparams.n_hidden, n_oup, hparams.kernal_size, hparams.n_stacks, hparams.n_blocks)
    elif 'crnn' == model_choose:
        model = CRNN.CRNN()
    else:
        raise ValueError('The model you choose is not supported!')

    return model

# choose which pre-processing method to use
def preprocess_creator(hparams, feature):
    if feature == 'melspec':
        transform = preprocess.MelSpec(hparams.sr, hparams.n_fft, hparams.hop_size)
    elif feature == 'stft':
        transform = preprocess.STFT(hparams.n_fft, hparams.hop_size)
    elif feature == 'vgg':
        transform = preprocess.vgg_mel()
    elif feature == 'pcen':
        transform = pcen.PCENTransform(n_mels=hparams.n_features, n_fft=hparams.n_fft, hop_length=hparams.hop_size, trainable=True)
    else:
        raise ValueError('The preprocessing method you choose is not supported!')

    return transform

