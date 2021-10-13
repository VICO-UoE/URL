import os
import gin
import sys
import torch
import numpy as np
import tensorflow as tf

from utils import device
from paths import META_DATASET_ROOT, META_RECORDS_ROOT, PROJECT_ROOT
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Quiet the TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Quiet the TensorFlow warnings

sys.path.append(os.path.abspath(META_DATASET_ROOT))
from meta_dataset.data import dataset_spec as dataset_spec_lib
from meta_dataset.data import learning_spec
from meta_dataset.data import pipeline
from meta_dataset.data import config


ALL_METADATASET_NAMES = "ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower traffic_sign mscoco mnist cifar10 cifar100".split(' ')
TRAIN_METADATASET_NAMES = ALL_METADATASET_NAMES[:8]
TEST_METADATASET_NAMES = ALL_METADATASET_NAMES[-5:]

SPLIT_NAME_TO_SPLIT = {'train': learning_spec.Split.TRAIN,
                       'val': learning_spec.Split.VALID,
                       'test': learning_spec.Split.TEST}


class MetaDatasetReader(object):
    def __init__(self, mode, train_set, validation_set, test_set):
        assert (train_set is not None or validation_set is not None or test_set is not None)

        self.data_path = META_RECORDS_ROOT
        self.train_dataset_next_task = None
        self.validation_set_dict = {}
        self.test_set_dict = {}
        self.specs_dict = {}
        gin.parse_config_file(f"{PROJECT_ROOT}/data/meta_dataset_config.gin")

    def _get_dataset_spec(self, items):
        if isinstance(items, list):
            dataset_specs = []
            for dataset_name in items:
                dataset_records_path = os.path.join(self.data_path, dataset_name)
                dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
                dataset_specs.append(dataset_spec)
            return dataset_specs
        else:
            dataset_name = items
            dataset_records_path = os.path.join(self.data_path, dataset_name)
            dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
            return dataset_spec

    def _to_torch(self, sample):
        for key, val in sample.items():
            if isinstance(val, str):
                continue
            val = torch.from_numpy(val)
            if 'image' in key:
                val = val.permute(0, 3, 1, 2)
            else:
                val = val.long()
            sample[key] = val.to(device)
        return sample

    def num_classes(self, split_name):
        split = SPLIT_NAME_TO_SPLIT[split_name]
        all_split_specs = self.specs_dict[SPLIT_NAME_TO_SPLIT['train']]

        if not isinstance(all_split_specs, list):
            all_split_specs = [all_split_specs]

        total_n_classes = 0
        for specs in all_split_specs:
            total_n_classes += len(specs.get_classes(split))
        return total_n_classes

    def build_class_to_identity(self):
        split = SPLIT_NAME_TO_SPLIT['train']
        all_split_specs = self.specs_dict[SPLIT_NAME_TO_SPLIT['train']]

        if not isinstance(all_split_specs, list):
            all_split_specs = [all_split_specs]

        self.cls_to_identity = dict()
        self.dataset_id_to_dataset_name = dict()
        self.dataset_to_n_cats = dict()
        offset = 0
        for dataset_id, specs in enumerate(all_split_specs):
            dataset_name = specs.name
            self.dataset_id_to_dataset_name[dataset_id] = dataset_name
            n_cats = len(specs.get_classes(split))
            self.dataset_to_n_cats[dataset_name] = n_cats
            for cat in range(n_cats):
                self.cls_to_identity[offset + cat] = (cat, dataset_id)
            offset += n_cats

        self.dataset_name_to_dataset_id = {v: k for k, v in
                                           self.dataset_id_to_dataset_name.items()}


class MetaDatasetEpisodeReader(MetaDatasetReader):
    """
    Class that wraps the Meta-Dataset episode readers.
    """
    def __init__(self, mode, train_set=None, validation_set=None, test_set=None, test_type='standard'):
        super(MetaDatasetEpisodeReader, self).__init__(mode, train_set, validation_set, test_set)
        self.mode = mode

        # standard episode reader (varying-way-varying-shot)
        if test_type == 'standard':
            if mode == 'train':
                train_episode_desscription = config.EpisodeDescriptionConfig(None, None, None)
                self.train_dataset_next_task = self._init_multi_source_dataset(
                    train_set, SPLIT_NAME_TO_SPLIT['train'], train_episode_desscription)
                self.build_class_to_identity()

            if mode == 'val':
                test_episode_desscription = config.EpisodeDescriptionConfig(None, None, None)
                for item in validation_set:
                    next_task = self._init_single_source_dataset(
                        item, SPLIT_NAME_TO_SPLIT['val'], test_episode_desscription)
                    self.validation_set_dict[item] = next_task

            if mode == 'test':
                test_episode_desscription = config.EpisodeDescriptionConfig(None, None, None)
                for item in test_set:
                    next_task = self._init_single_source_dataset(
                        item, SPLIT_NAME_TO_SPLIT['test'], test_episode_desscription)
                    self.test_set_dict[item] = next_task

        # five-way-one-shot
        elif test_type == '1shot':
            if mode == 'train':
                train_episode_desscription = config.EpisodeDescriptionConfig(None, None, None, min_ways=5, max_ways_upper_bound=5, max_num_query=10, max_support_size_contrib_per_class=1)
                self.train_dataset_next_task = self._init_multi_source_dataset(
                    train_set, SPLIT_NAME_TO_SPLIT['train'], train_episode_desscription)

            if mode == 'val':
                test_episode_desscription = config.EpisodeDescriptionConfig(None, None, None, min_ways=5, max_ways_upper_bound=5, max_num_query=10, max_support_size_contrib_per_class=1)
                for item in validation_set:
                    next_task = self._init_single_source_dataset(
                        item, SPLIT_NAME_TO_SPLIT['val'], test_episode_desscription)
                    self.validation_set_dict[item] = next_task

            if mode == 'test':
                test_episode_desscription = config.EpisodeDescriptionConfig(None, None, None, min_ways=5, max_ways_upper_bound=5, max_num_query=10, max_support_size_contrib_per_class=1)
                for item in test_set:
                    next_task = self._init_single_source_dataset(
                        item, SPLIT_NAME_TO_SPLIT['test'], test_episode_desscription)
                    self.test_set_dict[item] = next_task

        # varying-way-five-shot
        elif test_type == '5shot':
            if mode == 'train':
                train_episode_desscription = config.EpisodeDescriptionConfig(None, 5, None)
                self.train_dataset_next_task = self._init_multi_source_dataset(
                    train_set, SPLIT_NAME_TO_SPLIT['train'], train_episode_desscription)

            if mode == 'val':
                test_episode_desscription = config.EpisodeDescriptionConfig(None, 5, None)
                for item in validation_set:
                    next_task = self._init_single_source_dataset(
                        item, SPLIT_NAME_TO_SPLIT['val'], test_episode_desscription)
                    self.validation_set_dict[item] = next_task

            if mode == 'test':
                test_episode_desscription = config.EpisodeDescriptionConfig(None, 5, None)
                for item in test_set:
                    next_task = self._init_single_source_dataset(
                        item, SPLIT_NAME_TO_SPLIT['test'], test_episode_desscription)
                    self.test_set_dict[item] = next_task


    def _init_multi_source_dataset(self, items, split, episode_description):
        dataset_specs = self._get_dataset_spec(items)
        self.specs_dict[split] = dataset_specs

        use_bilevel_ontology_list = [False] * len(items)
        use_dag_ontology_list = [False] * len(items)
        # Enable ontology aware sampling for Omniglot and ImageNet.
        if 'omniglot' in items:
            use_bilevel_ontology_list[items.index('omniglot')] = True
        if 'ilsvrc_2012' in items:
            use_dag_ontology_list[items.index('ilsvrc_2012')] = True

        multi_source_pipeline = pipeline.make_multisource_episode_pipeline(
            dataset_spec_list=dataset_specs,
            use_dag_ontology_list=use_dag_ontology_list,
            use_bilevel_ontology_list=use_bilevel_ontology_list,
            split=split,
            episode_descr_config = episode_description,
            image_size=84,
            shuffle_buffer_size=1000)

        iterator = multi_source_pipeline.make_one_shot_iterator()
        return iterator.get_next()

    def _init_single_source_dataset(self, dataset_name, split, episode_description):
        dataset_spec = self._get_dataset_spec(dataset_name)
        self.specs_dict[split] = dataset_spec

        # Enable ontology aware sampling for Omniglot and ImageNet.
        use_bilevel_ontology = False
        if 'omniglot' in dataset_name:
            use_bilevel_ontology = True

        use_dag_ontology = False
        if 'ilsvrc_2012' in dataset_name:
            use_dag_ontology = True

        single_source_pipeline = pipeline.make_one_source_episode_pipeline(
            dataset_spec=dataset_spec,
            use_dag_ontology=use_dag_ontology,
            use_bilevel_ontology=use_bilevel_ontology,
            split=split,
            episode_descr_config=episode_description,
            image_size=84,
            shuffle_buffer_size=1000)

        iterator = single_source_pipeline.make_one_shot_iterator()
        return iterator.get_next()

    def _get_task(self, next_task, session):
        episode = session.run(next_task)[0]
        task_dict = {
            'context_images': episode[0],
            'context_labels': episode[1],
            'context_gt': episode[2],
            'target_images': episode[3],
            'target_labels': episode[4],
            'target_gt': episode[5]
            }

        return self._to_torch(task_dict)

    def get_train_task(self, session):
        return self._get_task(self.train_dataset_next_task, session)

    def get_validation_task(self, session, item=None):
        item = item if item else list(self.validation_set_dict.keys())[0]
        return self._get_task(self.validation_set_dict[item], session)

    def get_test_task(self, session, item=None):
        item = item if item else list(self.test_set_dict.keys())[0]
        return self._get_task(self.test_set_dict[item], session)


class MetaDatasetBatchReader(MetaDatasetReader):
    """
    Class that wraps the Meta-Dataset episode readers.
    """
    def __init__(self, mode, train_set, validation_set, test_set, batch_size):
        super(MetaDatasetBatchReader, self).__init__(mode, train_set, validation_set, test_set)
        self.batch_size = batch_size

        if mode == 'train':
            self.train_dataset_next_task = self._init_multi_source_dataset(
                train_set, SPLIT_NAME_TO_SPLIT['train'])

        if mode == 'val':
            for item in validation_set:
                next_task = self.validation_dataset = self._init_single_source_dataset(
                    item, SPLIT_NAME_TO_SPLIT['val'])
                self.validation_set_dict[item] = next_task

        if mode == 'test':
            for item in test_set:
                next_task = self._init_single_source_dataset(
                    item, SPLIT_NAME_TO_SPLIT['test'])
                self.test_set_dict[item] = next_task

        self.build_class_to_identity()

    def _init_multi_source_dataset(self, items, split):
        dataset_specs = self._get_dataset_spec(items)
        self.specs_dict[split] = dataset_specs
        multi_source_pipeline = pipeline.make_multisource_batch_pipeline(
            dataset_spec_list=dataset_specs, batch_size=self.batch_size,
            split=split, image_size=84, add_dataset_offset=True, shuffle_buffer_size=1000)

        iterator = multi_source_pipeline.make_one_shot_iterator()
        return iterator.get_next()

    def _init_single_source_dataset(self, dataset_name, split):
        dataset_specs = self._get_dataset_spec(dataset_name)
        self.specs_dict[split] = dataset_specs
        multi_source_pipeline = pipeline.make_one_source_batch_pipeline(
            dataset_spec=dataset_specs, batch_size=self.batch_size,
            split=split, image_size=84)

        iterator = multi_source_pipeline.make_one_shot_iterator()
        return iterator.get_next()

    def _get_batch(self, next_task, session):
        episode = session.run(next_task)[0]
        images, labels = episode[0], episode[1]
        local_classes, dataset_ids = [], []
        for label in labels:
            local_class, dataset_id = self.cls_to_identity[label]
            local_classes.append(local_class)
            dataset_ids.append(dataset_id)
        task_dict = {
            'images': images,
            'labels': labels,
            'local_classes': np.array(local_classes),
            'dataset_ids': np.array(dataset_ids),
            'dataset_name': self.dataset_id_to_dataset_name[dataset_ids[-1]]
            }
        return self._to_torch(task_dict)

    def get_train_batch(self, session):
        return self._get_batch(self.train_dataset_next_task, session)

    def get_validation_batch(self, item, session):
        return self._get_batch(self.validation_set_dict[item], session)

    def get_test_batch(self, item, session):
        return self._get_batch(self.test_set_dict[item], session)
