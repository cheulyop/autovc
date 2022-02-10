import os
import random
import re
import string
import librosa
import numpy as np
from collections import Counter, deque, namedtuple
from itertools import chain
from tempfile import NamedTemporaryFile

import soundfile as sf
import torch
from librosa.filters import mel
from chamd import ChatReader
from p2fa import align
from pylangacq import Reader, read_chat
from scipy import signal
from scipy.signal import get_window
from numpy.random import RandomState



class AutovcCollator:
    """
    Collate dataset for both BERT & Wav2Vec2
    Final:
    (I) dict {attention_mask, input_ids, labels, token_type_ids, embeddings}
    (o) tuple (dict {attention_mask, input_ids, labels, token_type_ids}, embeddings, labels)
    (if only bert fine-tuning, only return ds)
    """

    def __init__(self, dim=0):
        self.dim = dim

    def collate(self, batch):
        spmels = [row.pop("spmel") for row in batch]

        max_len = max(map(lambda x: np.shape(x)[0], spmels))
        # freq 32
        max_len += (32 - max_len % 32)
        spmels = torch.FloatTensor(list(
            map(
                lambda x: self.pad_vector(x, pad=max_len),
                spmels,
            )
        ))
        batch = {
            col: self.merge([self.dict_to_tensor(dic[col], col) for dic in batch], col)
            for col in batch[0]
        }

        labels = batch["labels"]

        embeddings = batch.pop("sp_emb")
        # batch = list(zip(embeddings, labels))

        # TODO: pad if needed
        # max_len = max(map(lambda x: x.shape[self.dim], embeddings))  # find longest sequence
        # embeddings = list(
        #     map(
        #         lambda x: self.pad_tensor(x, pad=max_len, dim=self.dim),
        #         embeddings,
        #     )
        # )  # pad according to max_len
        # embeddings = torch.stack(
        #     [torch.squeeze(embedding) for embedding in embeddings], dim=0
        # )  # a tensor of all examples in 'batch' after padding
        
        labels = torch.FloatTensor(
            [label for label in labels]
        )  # a LongTensor of all abels in batch
        return spmels, embeddings, labels

    @staticmethod
    def dict_to_tensor(value, key):
        if key == "labels":
            return torch.LongTensor([value])
        else:
            return torch.LongTensor(value)

    @staticmethod
    def merge(tensor_list, key):
        if key == "labels":
            return torch.cat(tensor_list)
        else:
            return torch.stack(tensor_list)

    @staticmethod
    def pad_tensor(vec, pad, dim):
        """
        vec = tensor to pad
        pad = the size to pad to
        dim = dimension to pad
        """
        pad_size = list(vec.shape)
        pad_size[dim] = pad - vec.size(dim)
        return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)

    @staticmethod
    def pad_vector(vec, pad):
        """
        vec = tensor to pad
        pad = the size to pad to
        dim = dimension to pad
        """
        return np.pad(vec, ((0, pad - np.shape(vec)[0]),(0,0)))

    def __call__(self, batch):
        return self.collate(batch)



def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)
    # Numerator (b) and denominator (a)
    return b, a

def pySTFT(x, fft_length=1024, hop_length=256):

    x = np.pad(x, int(fft_length // 2), mode="reflect")

    noverlap = fft_length - hop_length
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // hop_length, fft_length)
    strides = x.strides[:-1] + (hop_length * x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    fft_window = get_window("hann", fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T

    return np.abs(result)

def speech_file_to_melspec(example, sr=16_000, pcm=False):
    """
    works with individual examples
    """
    if not pcm:
        speech, sr = librosa.load(example["file"], sr=sr, mono=True)
    else:
        with open(example["file"], "rb") as f:
            speech = np.fromfile(f, dtype="<i2")

    # # remove drifting noise
    # b, a = butter_highpass(30, 16000, order=5)
    # new_speech = signal.filtfilt(b, a, speech)

    # # Add a little random noise for model roubstness (remove??)
    # noised_speech = (
    #             new_speech * 0.96
    #             + (RandomState(int(example["file"].split("/")[-1].split("-")[0])).rand(new_speech.shape[0]) - 0.5)
    #             * 1e-06
    #         )
    
    # # make array to spect
    # spec = pySTFT(noised_speech).T

    # # convert spect to melspect / normalize
    # mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
    # min_level = np.exp(-100 / 20 * np.log(10))

    # mel_spec = np.dot(spec, mel_basis)
    # db_spec = 20 * np.log10(np.maximum(min_level, mel_spec)) - 16
    # normalized_spec = np.clip((db_spec + 100) / 100, 0, 1)


    mel_spec = librosa.feature.melspectrogram(
        speech,
        sr,
        n_fft=int(sr * 25 / 1000),
        hop_length=int(sr * 10 / 1000),
        n_mels=40
    )

    example['spmel'] = mel_spec.astype(np.float32).T

    return example

def get_metadata(example, speaker_encoder):
    speaker_mel = torch.tensor(example['spmel']).unsqueeze(0)
    with torch.no_grad():
        speaker_emb = speaker_encoder(speaker_mel)
    example['sp_emb'] = speaker_emb.detach().squeeze().cpu().numpy()
    example['sp_id'] = int(example["file"].split("/")[-1].split("-")[0])
    example['labels'] = example['control']
    return example


# not used
def get_transcript(batch, speaker="PAR"):
    """
    Speaker can be one of ['PAR', 'INV', None], where None will included both 'PAR' and 'INV'
    """
    chat_files = Reader.from_files(batch["text"])
    words_by_file = chat_files.words(by_files=True, participants=speaker)
    batch["transcript"] = list(map(" ".join, words_by_file))
    return batch


def force_align_speech(example):
    """
    Force align speech and transcript with Penn Phonetics Lab Force Aligned (P2FA)
    """
    with NamedTemporaryFile() as wavfile, NamedTemporaryFile() as trsfile:
        # temporarily write speech array as a wav file
        sf.write(wavfile.name, example["speech"], 16000, format="WAV", subtype="PCM_16")

        # temporarily write transcript to a file
        with open(trsfile.name, "w") as f:
            f.write(example["transcript"])

        _, word_alignments, _ = align.align(
            wavfile=wavfile.name,
            trsfile=trsfile.name,
        )
    example["aligned_words"] = [list(map(str, word_info)) for word_info in word_alignments]
    return example