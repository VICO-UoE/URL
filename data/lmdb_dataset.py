import os
import torch
import random
import numpy as np
from tqdm import tqdm

import lmdb
import pickle as pkl
from utils import SerializableArray, device

from paths import META_DATA_ROOT

class LMDBDataset:
    """
    Opens several LMDB readers and loads data from there
    """
    def __init__(self, extractor_domains, datasets, backbone, mode, dump_name, limit_len=None):
        self.mode = mode
        self.datasets = datasets

        self.dataset_readers = dict()
        for evalset in datasets:
            all_names = os.listdir(os.path.join(META_DATA_ROOT, 'Dumps',
                                                backbone, mode, evalset))
            self.dataset_readers[evalset] = [
                DatasetReader(extractor_domains, evalset, backbone, mode, name)
                for name in all_names if dump_name in name]
        self._current_sampling_dataset = datasets[0]
        self.full_len = sum([len(ds) for ds in self.dataset_readers[self._current_sampling_dataset]])
        if limit_len is not None:
            self.full_len = min(self.full_len, limit_len)

    def __len__(self):
        return self.full_len

    def __getitem__(self, idx):
        if self.mode == 'train':
            random_lmdb_subset = random.sample(self.dataset_readers[self._current_sampling_dataset], 1)[0]
            idx = random.sample(range(len(random_lmdb_subset)), 1)[0]
            sample = random_lmdb_subset[idx]
        else:
            sample = self.dataset_readers[self._current_sampling_dataset][0][idx]

        for key, val in sample.items():
            if isinstance(val, str):
                pass
            if 'label' in key:
                sample[key] = torch.from_numpy(val).long()
            elif 'feature_dict' in key:
                for fkey, fval in sample[key].items():
                    sample[key][fkey] = torch.from_numpy(fval)

        return sample

    def set_sampling_dataset(self, sampling_dataset):
        self._current_sampling_dataset = sampling_dataset

    # open lmdb environment and transaction
    # load keys from cache
    def _load_db(self, info, class_id):
        path = self._path

        self._env = lmdb.open(
            self._path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        self._txn = self._env.begin(write=False)

        if class_id is None:
            cache_file = os.path.join(path, 'keys')
            if os.path.isfile(cache_file):
                self.keys = pkl.load(open(cache_file, 'rb'))
            else:
                print('Loading dataset keys...')
                with self._env.begin(write=False) as txn:
                    self.keys = [key.decode('ascii')
                                for key, _ in tqdm(txn.cursor())]
                pkl.dump(self.keys, open(cache_file, 'wb'))
        else:
            self.keys = [str(k).encode() for k in info['labels2keys'][str(class_id)]]

        if not self.keys:
            raise ValueError('Empty dataset.')

    def eval(self):
        self.mode = 'eval'

    def train(self):
        self.mode = 'train'

    def transform(self, x):
        if self.mode == 'train':
            out = self.train_transform(x) if self.train_transform else x
            return out
        else:
            out = self.test_transform(x) if self.test_transform else x
            return out


class DatasetReader(object):
    """
    Opens a single LMDB file, containing dumped activations for a dataset,
    and samples data from it.
    """
    def __init__(self, extractor_domains, evalset, backbone, mode, name):
        self._mode = mode
        self._env = None
        self._txn = None
        self.keys = None

        self.trainsets = extractor_domains
        path = os.path.join(META_DATA_ROOT, 'Dumps', backbone, mode, evalset, name)
        self._path = path

        self._load_db()

    def __len__(self):
        return self.full_len

    def _load_db(self):
        path = self._path

        self._env = lmdb.open(
            self._path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        self._txn = self._env.begin(write=False)

        cache_file = os.path.join(path, 'keys')
        if os.path.isfile(cache_file):
            self.keys = pkl.load(open(cache_file, 'rb'))
        else:
            print('Loading dataset keys...')
            with self._env.begin(write=False) as txn:
                self.keys = [key.decode('ascii')
                            for key, _ in tqdm(txn.cursor())]
            pkl.dump(self.keys, open(cache_file, 'wb'))
        self.full_len = len(self.keys) // 18

    def __getitem__(self, idx):
        sample = dict()
        support_labels = pkl.loads(self._txn.get(f"{idx}_labels_support".encode("ascii")))
        query_labels = pkl.loads(self._txn.get(f"{idx}_labels_query".encode("ascii")))
        sample['context_labels'] = support_labels.get()
        sample['target_labels'] = query_labels.get()

        sample['context_feature_dict'] = dict()
        sample['target_feature_dict'] = dict()
        for dataset in self.trainsets:
            support_batch = pkl.loads(self._txn.get(f"{idx}_{dataset}_support".encode("ascii")))
            query_batch = pkl.loads(self._txn.get(f"{idx}_{dataset}_query".encode("ascii")))
            sample['context_feature_dict'][dataset] = support_batch.get()
            sample['target_feature_dict'][dataset] = query_batch.get()
        return sample
