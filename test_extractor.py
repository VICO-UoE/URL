"""
This code allows you to evaluate performance of a single feature extractor + a classifier
on the test splits of all datasets (ilsvrc_2012, omniglot, aircraft, cu_birds, dtd, quickdraw, fungi, 
vgg_flower, traffic_sign, mscoco, mnist, cifar10, cifar100). 

The default classifier used in this code is the NCC with cosine similarity. 
One can use other classifiers for meta-testing, 
e.g. use ```--test.loss-opt``` to select nearest centroid classifier (ncc, default), 
support vector machine (svm), logistic regression (lr), Mahalanobis distance from 
Simple CNAPS (scm), or k-nearest neighbor (knn); 
use ```--test.feature-norm``` to normalize feature (l2) or not for svm and lr; 
use ```--test.distance``` to specify the feature similarity function (l2 or cos) for NCC. 

To evaluate the feature extractor with NCC and cosine similarity on test splits of all datasets, run:
python test_extractor.py --test.loss-opt ncc --test.feature-norm none --test.distance cos --model.name=<model name> --model.dir <directory of url> 

To test the feature extractor one the test splits of ilsrvc_2012, dtd, vgg_flower, quickdraw,
comment the line 'testsets = ALL_METADATASET_NAMES' and run:
python test_extractor.py --test.loss-opt ncc --test.feature-norm none --test.distance cos --data.test ilsrvc_2012 dtd vgg_flower quickdraw --model.name=<model name> --model.dir <directory of url> 
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
from data.meta_dataset_reader import (MetaDatasetEpisodeReader, MetaDatasetBatchReader, TRAIN_METADATASET_NAMES,
                                      ALL_METADATASET_NAMES)
from config import args


def main():
    TEST_SIZE = 600

    # Setting up datasets
    trainsets, valsets, testsets = args['data.train'], args['data.val'], args['data.test']
    testsets = ALL_METADATASET_NAMES # comment this line to test the model on args['data.test']
    trainsets = TRAIN_METADATASET_NAMES
    test_loader = MetaDatasetEpisodeReader('test', trainsets, trainsets, testsets, test_type=args['test.type'])
    model = get_model(None, args)
    checkpointer = CheckPointer(args, model, optimizer=None)
    checkpointer.restore_model(ckpt='best', strict=False)
    model.eval()
    accs_names = [args['test.loss_opt']]
    var_accs = dict()

    config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.allow_growth = False
    with tf.compat.v1.Session(config=config) as session:
        # go over each test domain
        for dataset in testsets:
            print(dataset)
            var_accs[dataset] = {name: [] for name in accs_names}

            for i in tqdm(range(TEST_SIZE)):
                with torch.no_grad():
                    sample = test_loader.get_test_task(session, dataset)
                    context_features = model.embed(sample['context_images'])
                    target_features = model.embed(sample['target_images'])
                    context_labels = sample['context_labels']
                    target_labels = sample['target_labels']
                    if args['test.loss_opt'] == 'ncc':
                        _, stats_dict, _ = prototype_loss(
                            context_features, context_labels,
                            target_features, target_labels, distance=args['test.distance'])
                    elif args['test.loss_opt'] == 'knn':
                        _, stats_dict, _ = knn_loss(
                            context_features, context_labels,
                            target_features, target_labels)
                    elif args['test.loss_opt'] == 'lr':
                        _, stats_dict, _ = lr_loss(
                            context_features, context_labels,
                            target_features, target_labels, normalize=(args['test.feature_norm'] == 'l2'))
                    elif args['test.loss_opt'] == 'svm':
                        _, stats_dict, _ = svm_loss(
                            context_features, context_labels,
                            target_features, target_labels, normalize=(args['test.feature_norm'] == 'l2'))
                    elif args['test.loss_opt'] == 'scm':
                        _, stats_dict, _ = scm_loss(
                            context_features, context_labels,
                            target_features, target_labels, normalize=False)
                var_accs[dataset][args['test.loss_opt']].append(stats_dict['acc'])
            dataset_acc = np.array(var_accs[dataset][args['test.loss_opt']]) * 100
            print(f"{dataset}: test_acc {dataset_acc.mean():.2f}%")
    # Print nice results table
    print('results of {}'.format(args['model.name']))
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
    out_path = os.path.join(out_path, '{}-{}-{}-{}-{}-test-results.npy'.format(args['model.name'], args['test.type'], args['test.loss_opt'], args['test.feature_norm'], args['test.distance']))
    np.save(out_path, {'rows': rows})

    table = tabulate(rows, headers=['model \\ data'] + accs_names, floatfmt=".2f")
    print(table)
    print("\n")


if __name__ == '__main__':
    main()



