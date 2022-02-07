from torch.utils import data
import torch
import numpy as np
import pickle
import os

from multiprocessing import Process, Manager


class Speeches(data.Dataset):
    def __init__(self, data_path, len_crop):
        """Initialize and preprocess the dataset."""
        self.len_crop = len_crop
        self.step = 10

        # metadata = [[speaker id, embedding, filepath to saved mel-spectrogram], ...]
        metadata = pickle.load(open(data_path, "rb"))

        """Load data using multiprocessing"""
        manager = Manager()
        metadata = manager.list(metadata)
        dataset = manager.list(len(metadata) * [None])

        processes = []
        for i in range(0, len(metadata), self.step):
            p = Process(
                target=self.load_data, args=(metadata[i : i + self.step], dataset, i)
            )
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()

        self.train_dataset = list(dataset)
        self.num_tokens = len(self.train_dataset)

        print("Finished loading the dataset...")
        return

    def load_data(self, submeta, dataset, offset):
        # item = [speaker id, embedding, filepath]
        for i, item in enumerate(submeta):
            dataset[offset + i] = [
                item[0], item[1], np.load(item[2])
            ]
        return

    def __getitem__(self, index):
        # pick a random speaker
        dataset = self.train_dataset
        speech = dataset[index]
        E, S = speech[1], speech[-1]
        return S, E

    def __len__(self):
        """Return the number of spkrs."""
        return self.num_tokens


def get_loader(data_path, batch_size=16, len_crop=128, num_workers=0):
    """Build and return a data loader."""

    dataset = Speeches(data_path, len_crop)

    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    return data_loader
