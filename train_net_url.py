"""
This code allows you to train the URL model proposed in 
'Universal Representation Learning from Multiple Domains for Few-shot Classification'
(https://arxiv.org/pdf/2103.13841.pdf).
"""

import os
import sys
import torch
import numpy as np
import tensorflow as tf
from time import sleep

from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter

from data.meta_dataset_reader import (MetaDatasetBatchReader,
                                      MetaDatasetEpisodeReader)
from models.losses import cross_entropy_loss, prototype_loss, distillation_loss, DistillKL
from models.model_utils import (CheckPointer, UniformStepLR,
                                CosineAnnealRestartLR, ExpDecayLR, WeightAnnealing)
from models.models_dict import DATASET_MODELS_DICT
from models.model_helpers import get_model, get_optimizer, get_domain_extractors
from models.adaptors import adaptor
from utils import Accumulator, device
from config import args, BATCHSIZES, LOSSWEIGHTS, KDFLOSSWEIGHTS, KDPLOSSWEIGHTS, KDANNEALING


def train():
    # initialize datasets and loaders
    trainsets, valsets, testsets = args['data.train'], args['data.val'], args['data.test']

    train_loaders = []
    num_train_classes = dict()
    kd_weight_annealing = dict()
    for t_indx, trainset in enumerate(trainsets):
        train_loaders.append(MetaDatasetBatchReader('train', [trainset], valsets, testsets,
                                          batch_size=BATCHSIZES[trainset]))
        num_train_classes[trainset] = train_loaders[t_indx].num_classes('train')
        # setting up knowledge distillation losses weights annealing
        kd_weight_annealing[trainset] = WeightAnnealing(T=int(args['train.cosine_anneal_freq'] * KDANNEALING[trainset]))
    val_loader = MetaDatasetEpisodeReader('val', trainsets, valsets, testsets)

    # initialize model and optimizer
    model = get_model(list(num_train_classes.values()), args)
    model_name_temp = args['model.name']
    # KL-divergence loss
    criterion_div = DistillKL(T=4)
    # get a MTL model initialized by ImageNet pretrained model and deactivate the pretrained flag
    args['model.pretrained']=False
    optimizer = get_optimizer(model, args, params=model.get_parameters())
    # adaptors for aligning features between MDL and SDL models
    adaptors = adaptor(num_datasets=len(trainsets), dim_in=512, opt=args['adaptor.opt']).to(device)
    optimizer_adaptor = torch.optim.Adam(adaptors.parameters(), lr=0.1, weight_decay=5e-4)

    # loading single domain learning networks
    extractor_domains = trainsets
    dataset_models = DATASET_MODELS_DICT[args['model.backbone']]
    embed_many = get_domain_extractors(extractor_domains, dataset_models, args, num_train_classes)

    # restoring the last checkpoint
    args['model.name'] = model_name_temp
    checkpointer = CheckPointer(args, model, optimizer=optimizer)
    if os.path.isfile(checkpointer.out_last_ckpt) and args['train.resume']:
        start_iter, best_val_loss, best_val_acc =\
            checkpointer.restore_out_model(ckpt='last')
    else:
        print('No checkpoint restoration')
        best_val_loss = 999999999
        best_val_acc = start_iter = 0

    # define learning rate policy
    if args['train.lr_policy'] == "step":
        lr_manager = UniformStepLR(optimizer, args, start_iter)
        lr_manager_ad = UniformStepLR(optimizer_adaptor, args, start_iter)
    elif "exp_decay" in args['train.lr_policy']:
        lr_manager = ExpDecayLR(optimizer, args, start_iter)
        lr_manager_ad = ExpDecayLR(optimizer_adaptor, args, start_iter)
    elif "cosine" in args['train.lr_policy']:
        lr_manager = CosineAnnealRestartLR(optimizer, args, start_iter)
        lr_manager_ad = CosineAnnealRestartLR(optimizer_adaptor, args, start_iter)

    # defining the summary writer
    writer = SummaryWriter(checkpointer.out_path)

    # Training loop
    max_iter = args['train.max_iter']
    epoch_loss = {name: [] for name in trainsets}
    epoch_kd_f_loss = {name: [] for name in trainsets}
    epoch_kd_p_loss = {name: [] for name in trainsets}
    epoch_acc = {name: [] for name in trainsets}
    epoch_val_loss = {name: [] for name in valsets}
    epoch_val_acc = {name: [] for name in valsets}
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False
    with tf.compat.v1.Session(config=config) as session:
        for i in tqdm(range(max_iter)):
            if i < start_iter:
                continue

            optimizer.zero_grad()
            optimizer_adaptor.zero_grad()

            samples = []
            images = dict()
            num_samples = []
            # loading images and labels
            for t_indx, (name, train_loader) in enumerate(zip(trainsets, train_loaders)):
                sample = train_loader.get_train_batch(session)
                samples.append(sample)
                images[name] = sample['images']
                num_samples.append(sample['images'].size(0))

            logits, mtl_features = model.forward(torch.cat(list(images.values()), dim=0), num_samples, kd=True)
            stl_features, stl_logits = embed_many(images, return_type='list', kd=True, logits=True)
            mtl_features = adaptors(mtl_features)

            batch_losses, stats_dicts = [], []
            kd_f_losses = 0
            kd_p_losses = 0
            for t_indx, trainset in enumerate(trainsets):
                batch_loss, stats_dict, _ = cross_entropy_loss(logits[t_indx], samples[t_indx]['labels'])
                batch_losses.append(batch_loss*LOSSWEIGHTS[trainset])
                stats_dicts.append(stats_dict)
                batch_dataset = samples[t_indx]['dataset_name']
                epoch_loss[batch_dataset].append(stats_dict['loss'])
                epoch_acc[batch_dataset].append(stats_dict['acc'])
                ft, fs = torch.nn.functional.normalize(stl_features[t_indx], p=2, dim=1, eps=1e-12), torch.nn.functional.normalize(mtl_features[t_indx], p=2, dim=1, eps=1e-12)
                kd_f_losses_ = distillation_loss(fs, ft.detach(), opt='kernelcka')
                kd_p_losses_ = criterion_div(logits[t_indx], stl_logits[t_indx])
                kd_weight = kd_weight_annealing[trainset](t=i, opt='linear') * KDFLOSSWEIGHTS[trainset]
                bam_weight = kd_weight_annealing[trainset](t=i, opt='linear') * KDPLOSSWEIGHTS[trainset]
                if kd_weight > 0:
                    kd_f_losses = kd_f_losses + kd_f_losses_ * kd_weight
                if bam_weight > 0:
                    kd_p_losses = kd_p_losses + kd_p_losses_ * bam_weight
                epoch_kd_f_loss[batch_dataset].append(kd_f_losses_.item())
                epoch_kd_p_loss[batch_dataset].append(kd_p_losses_.item())

            batch_loss = torch.stack(batch_losses).sum()
            kd_f_loss = kd_f_losses * args['train.sigma']
            kd_p_loss = kd_p_losses * args['train.beta']
            batch_loss = batch_loss + kd_f_loss + kd_p_loss


            batch_loss.backward()
            optimizer.step()
            optimizer_adaptor.step()
            lr_manager.step(i)
            lr_manager_ad.step(i)

            if (i + 1) % 200 == 0:
                for dataset_name in trainsets:
                    writer.add_scalar(f"loss/{dataset_name}-train_loss",
                                      np.mean(epoch_loss[dataset_name]), i)
                    writer.add_scalar(f"accuracy/{dataset_name}-train_acc",
                                      np.mean(epoch_acc[dataset_name]), i)
                    writer.add_scalar(f"kd_f_loss/{dataset_name}-train_kd_f_loss",
                                      np.mean(epoch_kd_f_loss[dataset_name]), i)
                    writer.add_scalar(f"kd_p_loss/{dataset_name}-train_kd_p_loss",
                                      np.mean(epoch_kd_p_loss[dataset_name]), i)
                    epoch_loss[dataset_name], epoch_acc[dataset_name], epoch_kd_f_loss[dataset_name], epoch_kd_p_loss[dataset_name] = [], [], [], []

                writer.add_scalar('learning_rate',
                                  optimizer.param_groups[0]['lr'], i)

            # Evaluation inside the training loop
            if (i + 1) % args['train.eval_freq'] == 0:
                model.eval()
                dataset_accs, dataset_losses = [], []
                for valset in valsets:
                    val_losses, val_accs = [], []
                    for j in tqdm(range(args['train.eval_size'])):
                        with torch.no_grad():
                            sample = val_loader.get_validation_task(session, valset)
                            context_features = model.embed(sample['context_images'])
                            target_features = model.embed(sample['target_images'])
                            context_labels = sample['context_labels']
                            target_labels = sample['target_labels']
                            _, stats_dict, _ = prototype_loss(context_features, context_labels,
                                                              target_features, target_labels)
                        val_losses.append(stats_dict['loss'])
                        val_accs.append(stats_dict['acc'])

                    # write summaries per validation set
                    dataset_acc, dataset_loss = np.mean(val_accs) * 100, np.mean(val_losses)
                    dataset_accs.append(dataset_acc)
                    dataset_losses.append(dataset_loss)
                    epoch_val_loss[valset].append(dataset_loss)
                    epoch_val_acc[valset].append(dataset_acc)
                    writer.add_scalar(f"loss/{valset}/val_loss", dataset_loss, i)
                    writer.add_scalar(f"accuracy/{valset}/val_acc", dataset_acc, i)
                    print(f"{valset}: val_acc {dataset_acc:.2f}%, val_loss {dataset_loss:.3f}")

                # write summaries averaged over datasets
                avg_val_loss, avg_val_acc = np.mean(dataset_losses), np.mean(dataset_accs)
                writer.add_scalar(f"loss/avg_val_loss", avg_val_loss, i)
                writer.add_scalar(f"accuracy/avg_val_acc", avg_val_acc, i)

                # saving checkpoints
                if avg_val_acc > best_val_acc:
                    best_val_loss, best_val_acc = avg_val_loss, avg_val_acc
                    is_best = True
                    print('Best model so far!')
                else:
                    is_best = False
                extra_dict = {'epoch_loss': epoch_loss, 'epoch_acc': epoch_acc, 'epoch_val_loss': epoch_val_loss, 'epoch_val_acc': epoch_val_acc, 'adaptors': adaptors.state_dict(), 'optimizer_adaptor':optimizer_adaptor.state_dict()}
                checkpointer.save_checkpoint(i, best_val_acc, best_val_loss,
                                             is_best, optimizer=optimizer,
                                             state_dict=model.get_state_dict(), extra=extra_dict)

                model.train()
                print(f"Trained and evaluated at {i}")

    writer.close()
    if start_iter < max_iter:
        print(f"""Done training with best_mean_val_loss: {best_val_loss:.3f}, best_avg_val_acc: {best_val_acc:.2f}%""")
    else:
        print(f"""No training happened. Loaded checkpoint at {start_iter}, while max_iter was {max_iter}""")


if __name__ == '__main__':
    train()
