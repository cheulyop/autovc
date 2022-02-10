import os
import pytorch_lightning as pl
import torch
import pickle
from datasets import load_dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from typing import Dict, List, Optional, Union
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.utils.dementiabank_utils import speech_file_to_melspec, get_metadata, AutovcCollator
from src.models.speaker_encoder import SpeakerEncoder
from src.utils import utils

LOG = utils.get_logger(__name__)

class DementiabankDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, cache_dir, split_dir, num_folds=5, batch_size=1,
        num_workers=1,test_split_size=0.2, run_prepare_data=False):
        super().__init__()
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.split_dir = split_dir

        self.num_folds = num_folds
        self.batch_size = batch_size
        self.collate_fn = AutovcCollator(0)
        self.num_workers = num_workers
        self.test_split_size = test_split_size

        self.run_prepare_data = run_prepare_data

        tqdm.pandas()


    def prepare_data(self):
        if not self.run_prepare_data:
            return
        # get metadata dataset
        # file, text, id, control
        # filepath, textpath, id, control(label)
        dataset = load_dataset(
                "/home/dongseok/projects/autovc/src/utils/datautils/dementiabank.py",
                data_dir=self.data_dir,
                split="full",
            )
        dataset = dataset.to_pandas()
        print(dataset)
        
        # # 1. make mp3 to melspec
        dataset = dataset.progress_apply(speech_file_to_melspec, axis=1)
        
        # 2. make metadata to train autovc
        # get speaker encoder
        print(dataset)
        speaker_encoder = SpeakerEncoder(dim_input=80, dim_cell=768, dim_emb=256).eval()
        c_checkpoint = torch.load("assets/3000000-BL.ckpt")

        new_state_dict = OrderedDict()
        for key, val in c_checkpoint["model_b"].items():
            new_key = key[7:]
            new_state_dict[new_key] = val
        speaker_encoder.load_state_dict(new_state_dict)
        dataset = dataset.progress_apply(get_metadata, speaker_encoder=speaker_encoder, axis=1)
                                         
        
        # 3. save dataset
        print(dataset)
        dataset.to_pickle(self.cache_dir)
        return

    def setup(self, stage: Optional[str] = None) -> None:

        # skip setup(stratifying) when testing      
        if stage == "test":
            return

        if self.run_prepare_data:
            self.prepare_data()

        self.dataset = pd.read_pickle(self.cache_dir)
        print([np.shape(mel) for mel in self.dataset['spmel']])
        self.dataset = self.dataset.drop(
            [
                "file",
                "control",
                "text",
            ],
            axis=1,
        )
        self.dataset["labels"] = self.dataset["labels"].astype(int)
        # input_values, gen_transcript

        # idx_to_id = {i: key for i, key in enumerate(dataset_dict.keys())}
        # indexed_dataset_dict = {idx: dataset_dict[idx_to_id[idx]S] for idx in idx_to_id}
        # 0: {id: 123e, label: 1}
        # id: {label: 1}
        # self.dataset['labels'] = self.dataset['labels'].apply(lambda x: x if x == 0 else x-1)
        # (lambda x: 1 if x == 2 else x) normal vs abnormal
        # (lambda x: x if x == 0 else x-1) ad vs non ad
        
        if not os.path.isfile(self.split_dir):
            # make 0 : {id:{}, labels:1....}
            dataset = self.dataset.reset_index()
            dataset = dataset.set_index(pd.Index(list(range(len(self.dataset))), name="index"))
            dataset_dict = dataset.to_dict("index")
            # train-test-split
            # TODO: seed
            labels = np.array(dataset.labels)
            train_dataset, test_dataset, _, _ = train_test_split(
                dataset_dict, labels, test_size=self.test_split_size, stratify=labels
            )
            test_eids = [ds["id"] for ds in test_dataset]

            # get 5 fold ids
            kfold = StratifiedKFold(n_splits=5)
            fold_eids = []
            labels = [ds["labels"] for ds in train_dataset]
            for _, fold_ids in kfold.split(train_dataset, labels):
                # print(trains, tests)
                fold_eids.append([train_dataset[fold_id]["id"] for fold_id in fold_ids])
            with open(self.split_dir, "wb") as f:
                pickle.dump((fold_eids, test_eids), f)
        else:
            LOG.info("Loading previously saved stratified splits")
            with open(self.split_dir, "rb") as f:
                fold_eids, test_eids = pickle.load(f)

        LOG.info("Filtering dataset into splits")
        self.dataset.index = self.dataset.id
        print(self.dataset)
        self.fold_set = [self.dataset.loc[eids].drop(['id', 'sp_id'], axis=1) for eids in fold_eids]
        self.test_set = self.dataset.loc[test_eids].drop(['id', 'sp_id'], axis=1)


        self.test_set = self.test_set.set_index(pd.Index(list(range(len(self.test_set))))).to_dict(
            "index"
        )

    def set_folds(self, fold_index=0):
        self.train_set = self.fold_set.copy()
        self.val_set = self.train_set.pop(fold_index)
        self.train_set = pd.concat(self.train_set)

        self.train_set = self.train_set.set_index(
            pd.Index(list(range(len(self.train_set))))
        ).to_dict("index")
        self.val_set = self.val_set.set_index(pd.Index(list(range(len(self.val_set))))).to_dict(
            "index"
        )
        print(self.train_set)
        print(self.test_set)
        return

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        # self.train_set.set_format(type='torch', columns=['embeddings', 'control'])
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=False,
            pin_memory=False,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        # self.validation_set.set_format(type='torch', columns=['embeddings', 'control'])
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=False,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        # self.test_set.set_format(type='torch', columns=['embeddings', 'control'])
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=False,
        )


if __name__ == '__main__':
    dm = DementiabankDataModule(
        data_dir="/home/dongseok/projects/autovc/wavs/dementiabank/",
        cache_dir="/home/dongseok/data/cached/dementiabank/full_speech/autovc.pkl",
        split_dir="/home/dongseok/data/cached/dementiabank/full_speech/autovc_split_keys.pkl",
        batch_size=32,
        run_prepare_data=True,
        num_workers=len(os.sched_getaffinity(0)),
    )
    dm.setup()
    dm.set_folds(0)