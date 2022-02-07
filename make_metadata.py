"""
Generate speaker embeddings and metadata for training
"""
import glob
import os
import pickle
from collections import OrderedDict

import numpy as np
import torch

from model_bl import D_VECTOR


def main():
    # load pretrained speaker encoder,
    # with LSTM layers of size 768 and bottleneck fc layer of size 256
    C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
    c_checkpoint = torch.load("3000000-BL.ckpt")

    new_state_dict = OrderedDict()
    for key, val in c_checkpoint["model_b"].items():
        new_key = key[7:]
        new_state_dict[new_key] = val
    C.load_state_dict(new_state_dict)

    num_uttrs = 10
    len_crop = 128

    # Directory containing mel-spectrograms
    root = "./spmel/dementiabank"
    filelist = glob.glob(os.path.join(root, "*/*.npy"))

    metadata = []
    # for each speaker
    for fpath in filelist:
        speaker = fpath.split("/")[-1].split(".")[0]
        print(f"Processing: {fpath}")

        # load saved mel spectrogram
        melsp = torch.from_numpy(np.load(fpath)).unsqueeze(0).cuda()

        # get speaker embedding
        # in the original code, speaker embedding is computed from a randomly chosen portion of speech
        # cropped to match the length len_crop, but here we use a full speech instead.
        E = C(melsp).detach().squeeze().cpu().numpy()

        # at this point the content of utterances list becomes
        # [speaker, averaged embedding, files from the speaker ...]

        metadata.append((speaker, E, fpath))

    with open(os.path.join(root, "metadata.pkl"), "wb") as handle:
        pickle.dump(metadata, handle)


if __name__ == "__main__":
    main()
