import os
import sys
import torch
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import json
import lmdb
import pickle as pkl

sys.path.insert(0, '/'.join(os.path.realpath(__file__).split('/')[:-2]))

from data.meta_dataset_reader import MetaDatasetEpisodeReader, MetaDatasetBatchReader
from models.model_utils import  CheckPointer
from models.models_dict import DATASET_MODELS_DICT
from models.model_helpers import get_domain_extractors, get_model
from config import args
from paths import META_DATA_ROOT
from utils import check_dir, SerializableArray



class DatasetWriter(object):
    def __init__(self, args, rewrite=True, write_frequency=10):
        self._mode = args['dump.mode']
        self._write_frequency = write_frequency
        self._db = None
        self.args = args
        self.dataset_models = DATASET_MODELS_DICT[args['model.backbone']]
        print(self.dataset_models)

        trainsets, valsets, testsets = args['data.train'], args['data.val'], args['data.test']
        loader = MetaDatasetEpisodeReader(self._mode, trainsets, valsets, testsets)
        self._map_size = 50000 * 100 ** 2 * 512 * 8
        self.trainsets = trainsets

        if self._mode == 'train':
            evalset = "allcat"
            self.load_sample = lambda sess: loader.get_train_task(sess)
        elif self._mode == 'test':
            evalset = testsets[0]
            self.load_sample = lambda sess: loader.get_test_task(sess, evalset)
        elif self._mode == 'val':
            evalset = valsets[0]
            self.load_sample = lambda sess: loader.get_validation_task(sess, evalset)

        dump_name = mode + '_dump' if not args['dump.name'] else args['dump.name']
        path = check_dir(os.path.join(META_DATA_ROOT, 'Dumps', self.args['model.backbone'],
                                      self._mode, evalset, dump_name))
        self._path = path
        if not (os.path.exists(path)):
            os.mkdir(path)
        self._keys_file = os.path.join(path, 'keys')
        self._keys = []
        if os.path.exists(path) and not rewrite:
            raise NameError("Dataset {} already exists.".format(self._path))

    # do not initialize during __init__ to avoid pickling error when using MPI
    def init(self):
        self._db = lmdb.open(self._path, map_size=self._map_size, map_async=True)
        self.embed_many = get_domain_extractors(self.trainsets, self.dataset_models, self.args)

    def close(self):
        keys = tuple(self._keys)
        pkl.dump(keys, open(self._keys_file, 'wb'))

        if self._db is not None:
            self._db.sync()
            self._db.close()
            self._db = None

    def encode_dataset(self, n_tasks=1000):
        if self._db is None:
            self.init()

        txn = self._db.begin(write=True)
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=config) as session:
            for idx in tqdm(range(n_tasks)):
                # compressing image
                sample = self.load_sample(session)
                support_embed_dict = self.embed_many(sample['context_images'])
                query_embed_dict = self.embed_many(sample['target_images'])
                support_labels = SerializableArray(sample['context_labels'].detach().cpu().numpy())
                query_labels = SerializableArray(sample['target_labels'].detach().cpu().numpy())
                SerializableArray.__module__ = 'utils'

                # writing
                for dataset in support_embed_dict.keys():
                    support_batch = SerializableArray(
                        support_embed_dict[dataset].detach().cpu().numpy())
                    query_batch = SerializableArray(
                        query_embed_dict[dataset].detach().cpu().numpy())
                    SerializableArray.__module__ = 'utils'
                    txn.put(f"{idx}_{dataset}_support".encode("ascii"), pkl.dumps(support_batch))
                    txn.put(f"{idx}_{dataset}_query".encode("ascii"), pkl.dumps(query_batch))
                    self._keys.extend([f"{idx}_{dataset}_support", f"{idx}_{dataset}_query"])
                txn.put(f"{idx}_labels_support".encode("ascii"), pkl.dumps(support_labels))
                txn.put(f"{idx}_labels_query".encode("ascii"), pkl.dumps(query_labels))
                self._keys.extend([f"{idx}_labels_support", f"{idx}_labels_query"])

                # flushing into lmdb
                if idx > 0 and idx % self._write_frequency == 0:
                    txn.commit()
                    txn = self._db.begin(write=True)
            txn.commit()


if __name__ == '__main__':
    dr = DatasetWriter(args)
    dr.init()
    dr.encode_dataset(args['dump.size'])
    dr.close()
    print('Done')
