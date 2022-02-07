from argparse import Namespace
from collections import OrderedDict
from math import ceil

import numpy as np
import soundfile as sf
import torch
from librosa.filters import mel
from numpy.random import RandomState
from scipy import signal
from scipy.signal import get_window

from model_bl import D_VECTOR
from model_vc import Generator
from synthesis import build_model, wavegen


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)
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


def get_melspec(wav_path, seed=0):
    mel_basis = mel(16000, 1024, fmin=80, fmax=7600, n_mels=80).T
    min_level = np.exp(-100 / 20 * np.log(10))
    b, a = butter_highpass(30, 16000, order=5)

    # Read audio file
    x, fs = sf.read(wav_path)

    # Remove drifting noise
    y = signal.filtfilt(b, a, x)

    # Add a little random noise for model roubstness
    wav = y * 0.96 + (RandomState(seed).rand(y.shape[0]) - 0.5) * 1e-06

    # Compute spect
    D = pySTFT(wav).T

    # Convert to mel and normalize
    D_mel = np.dot(D, mel_basis)
    D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
    S = np.clip((D_db + 100) / 100, 0, 1)

    return S.astype(np.float32)


def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0]) / base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0, len_pad), (0, 0)), "constant"), len_pad


def main():
    dm_wav = "/home/cheul/data/DementiaBank/utt/005-0/005-0_par_14086-20422.wav"
    cn_wav = "/home/cheul/data/DementiaBank/utt/002-0/002-0_par_5703-11898.wav"

    S_cn, S_dm = get_melspec(cn_wav), get_melspec(dm_wav)
    print(f"Extracted mel-spectrograms:\n....CN: {S_cn.shape}\n....DM: {S_dm.shape}")

    # load pretrained speaker encoder,
    # with LSTM layers of size 768 and bottleneck fc layer of size 256
    E_S = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
    ckpt = torch.load("assets/3000000-BL.ckpt")

    new_state_dict = OrderedDict()
    for key, val in ckpt["model_b"].items():
        new_key = key[7:]
        new_state_dict[new_key] = val
    E_S.load_state_dict(new_state_dict)
    print(f"Loaded speaker encoder:\n{E_S}")

    # get speaker embeddings from mel-spectrograms
    S_cn_tensor = torch.from_numpy(S_cn).unsqueeze(0)
    S_dm_tensor = torch.from_numpy(S_dm).unsqueeze(0)

    I_cn = E_S(S_cn_tensor.cuda()).detach().cpu()
    I_dm = E_S(S_dm_tensor.cuda()).detach().cpu()
    print(
        f"Extracted speaker identity embeddings:\n....CN: {I_cn.shape}\n....DM: {I_dm.shape}"
    )

    config = Namespace(
        batch_size=2,
        data_dir="./spmel",
        dim_neck=32,
        dim_emb=256,
        dim_pre=512,
        freq=32,
        lambda_cd=1,
        len_crop=128,
        log_step=10,
        num_iters=1000000,
    )

    device = "cuda"
    G = (
        Generator(config.dim_neck, config.dim_emb, config.dim_pre, config.freq)
        .eval()
        .to(device)
    )
    G.load_state_dict(torch.load("assets/autovc.ckpt", map_location=device)["model"])
    print(f"Loaded Generator:\n{G}")

    S_cn_padded, len_pad_cn = pad_seq(S_cn)
    print(f"Padded CN by {len_pad_cn} to {S_cn_padded.shape}")

    S_dm_padded, len_pad_dm = pad_seq(S_dm)
    print(f"Padded DM by {len_pad_dm} to {S_dm_padded.shape}")

    S_cn_cuda = torch.from_numpy(S_cn_padded).unsqueeze(0).to(device)
    S_dm_cuda = torch.from_numpy(S_dm_padded).unsqueeze(0).to(device)
    I_cn_cuda = I_cn.to(device)
    I_dm_cuda = I_dm.to(device)
    print(
        f"Prepared inputs to the network:\n....CN: S={S_cn_cuda.shape}, I={I_cn_cuda.shape}\n....DM: {S_dm_cuda.shape}, I={I_dm_cuda.shape}"
    )

    with torch.no_grad():
        _, X_cn_to_dm, _ = G(S_cn_cuda, I_cn_cuda, I_dm_cuda)
        _, X_dm_to_cn, _ = G(S_dm_cuda, I_dm_cuda, I_cn_cuda)

    X_cn_to_dm = X_cn_to_dm[0, 0, :-len_pad_cn, :].cpu().numpy()
    X_dm_to_cn = X_dm_to_cn[0, 0, :-len_pad_dm, :].cpu().numpy()
    print(X_cn_to_dm.shape, X_dm_to_cn.shape)

    np.save("test/cn2dm_utt.npy", X_cn_to_dm)
    np.save("test/dm2cn_utt.npy", X_dm_to_cn)

    vocoder = build_model().to(device)
    vocoder_ckpt = torch.load("assets/checkpoint_step001000000_ema.pth")
    vocoder.load_state_dict(vocoder_ckpt["state_dict"])
    print(vocoder)

    wav_cn_to_dm = wavegen(vocoder, c=X_cn_to_dm)
    sf.write(f"test/outputs/wavenet/cn2dm_utt.wav", wav_cn_to_dm, samplerate=16000)

    # wav_dm_to_cn = wavegen(vocoder, c=X_dm_to_cn)
    # sf.write(f"test/outputs/wavenet/dm2cn.wav", wav_dm_to_cn, samplerate=16000)


if __name__ == "__main__":
    main()
