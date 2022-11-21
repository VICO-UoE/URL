"""
This code allows you to evaluate performance of a single feature extractor + tsa with NCC
on the test splits of all datasets (ilsvrc_2012, omniglot, aircraft, cu_birds, dtd, quickdraw, fungi, 
vgg_flower, traffic_sign, mscoco, mnist, cifar10, cifar100). 

To test the url model with residual adapters in matrix form and pre-classifier alignment
on the test splits of all datasets, run:
python test_extractor_tsa.py --model.name=url --model.dir ./saved_results/url -test.tsa-ad-type residual \
--test.tsa-ad-form matrix --test.tsa-opt alpha+beta --test.tsa-init eye

To test the url model with residual adapters in matrix form and pre-classifier alignment
on the test splits of ilsrvc_2012, dtd, vgg_flower, quickdraw,
comment the line 'testsets = ALL_METADATASET_NAMES' and run:
python test_extractor_tsa.py --model.name=url --model.dir ./saved_results/url -test.tsa-ad-type residual \
--test.tsa-ad-form matrix --test.tsa-opt alpha+beta --test.tsa-init eye \
-data.test ilsrvc_2012 dtd vgg_flower quickdraw
"""

import os
import torch
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from utils import check_dir

from models.losses import prototype_loss, knn_loss, lr_loss, scm_loss, svm_loss
from models.model_utils import CheckPointer
from models.model_helpers import get_model
from models.tsa import resnet_tsa, tsa
from data.meta_dataset_reader import (MetaDatasetEpisodeReader, MetaDatasetBatchReader, TRAIN_METADATASET_NAMES,
                                      ALL_METADATASET_NAMES)
from config import args


def main():
    TEST_SIZE = 600

    # Setting up datasets
    trainsets, valsets, testsets = args['data.train'], args['data.val'], args['data.test']
    testsets = ALL_METADATASET_NAMES # comment this line to test the model on args['data.test']
    if args['test.mode'] == 'mdl':
        # multi-domain learning setting, meta-train on 8 training sets
        trainsets = TRAIN_METADATASET_NAMES
    elif args['test.mode'] == 'sdl':
        # single-domain learning setting, meta-train on ImageNet
        trainsets = ['ilsvrc_2012']
    test_loader = MetaDatasetEpisodeReader('test', trainsets, trainsets, testsets, test_type=args['test.type'])
    model = get_model(None, args)
    checkpointer = CheckPointer(args, model, optimizer=None)
    checkpointer.restore_model(ckpt='best', strict=False)
    model.eval()
    model = resnet_tsa(model)
    model.reset()
    model.cuda()

    accs_names = ['NCC']
    var_accs = dict()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False
    with tf.compat.v1.Session(config=config) as session:
        # go over each test domain
        for dataset in testsets:
            if dataset in trainsets:
                if args['test.tsa_ad_type'] == 'serial' and args['test.tsa_ad_form'] == 'matrix':
                    lr = 0.001
                else:
                    lr = 0.05
                lr_beta = 0.1
            else:
                if args['test.tsa_ad_type'] == 'serial' and args['test.tsa_ad_form'] == 'matrix':
                    lr = 0.01
                elif args['test.tsa_ad_form'] == 'vector':
                    lr = 1
                else:
                    lr = 0.5
                lr_beta = 1
            print(dataset)
            var_accs[dataset] = {name: [] for name in accs_names}

            for i in tqdm(range(TEST_SIZE)):
                # initialize task-specific adapters and pre-classifier alignment for each task
                model.reset()
                # loading a task containing a support set and a query set
                sample = test_loader.get_test_task(session, dataset)
                context_images = sample['context_images']
                target_images = sample['target_images']
                context_labels = sample['context_labels']
                target_labels = sample['target_labels']

                # optimize task-specific adapters and/or pre-classifier alignment
                tsa(context_images, context_labels, model, max_iter=40, lr=lr, lr_beta=lr_beta, distance=args['test.distance'])
                with torch.no_grad():
                    context_features = model.embed(sample['context_images'])
                    target_features = model.embed(sample['target_images'])
                    if 'beta' in args['test.tsa_opt']:
                        context_features = model.beta(context_features)
                        target_features = model.beta(target_features)
                _, stats_dict, _ = prototype_loss(
                    context_features, context_labels,
                    target_features, target_labels, distance=args['test.distance'])

                var_accs[dataset]['NCC'].append(stats_dict['acc'])
            dataset_acc = np.array(var_accs[dataset]['NCC']) * 100
            print(f"{dataset}: test_acc {dataset_acc.mean():.2f}%")
    # Print nice results table
    print('results of {} with {}'.format(args['model.name'], args['test.tsa_opt']))
    rows = []
    for dataset_name in testsets:
        row = [dataset_name]
        for model_name in accs_names:
            acc = np.array(var_accs[dataset_name][model_name]) * 100
            mean_acc = acc.mean()
            conf = (1.96 * acc.std()) / np.sqrt(len(acc))
            row.append(f"{mean_acc:0.2f} +- {conf:0.2f}")
        rows.append(row)
    out_path = os.path.join(args['out.dir'], 'weights')
    out_path = check_dir(out_path, True)
    out_path = os.path.join(out_path, '{}-tsa-{}-{}-{}-{}-test-results-.npy'.format(args['model.name'], args['test.tsa_opt'], args['test.tsa_ad_type'], args['test.tsa_ad_form'], args['test.tsa_init']))
    np.save(out_path, {'rows': rows})

    table = tabulate(rows, headers=['model \\ data'] + accs_names, floatfmt=".2f")
    print(table)
    print("\n")


if __name__ == '__main__':
    main()



