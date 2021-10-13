import os
import torch
import shutil
import numpy as np
from torch import nn
from torch.optim.lr_scheduler import (MultiStepLR, ExponentialLR,
                                      CosineAnnealingWarmRestarts,
                                      CosineAnnealingLR)
from utils import check_dir, device


sigmoid = nn.Sigmoid()


def cosine_sim(embeds, prots):
    prots = prots.unsqueeze(0)
    embeds = embeds.unsqueeze(1)
    return F.cosine_similarity(embeds, prots, dim=-1, eps=1e-30)



class CosineClassifier(nn.Module):
    def __init__(self, n_feat, num_classes):
        super(CosineClassifier, self).__init__()
        self.num_classes = num_classes
        self.scale = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        weight = torch.FloatTensor(n_feat, num_classes).normal_(
                    0.0, np.sqrt(2.0 / num_classes))
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x):
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1, eps=1e-12)
        weight = torch.nn.functional.normalize(self.weight, p=2, dim=0, eps=1e-12)
        cos_dist = x_norm @ weight
        scores = self.scale * cos_dist
        return scores

    def extra_repr(self):
        s = 'CosineClassifier: input_channels={}, num_classes={}; learned_scale: {}'.format(
            self.weight.shape[0], self.weight.shape[1], self.scale.item())
        return s


class CosineConv(nn.Module):
    def __init__(self, n_feat, num_classes, kernel_size=1):
        super(CosineConv, self).__init__()
        self.scale = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        weight = torch.FloatTensor(num_classes, n_feat, 1, 1).normal_(
                    0.0, np.sqrt(2.0/num_classes))
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x):
        x_normalized = torch.nn.functional.normalize(
            x, p=2, dim=1, eps=1e-12)
        weight = torch.nn.functional.normalize(
            self.weight, p=2, dim=1, eps=1e-12)

        cos_dist = torch.nn.functional.conv2d(x_normalized, weight)
        scores = self.scale * cos_dist
        return scores

    def extra_repr(self):
        s = 'CosineConv: num_inputs={}, num_classes={}, kernel_size=1; scale_value: {}'.format(
            self.weight.shape[0], self.weight.shape[1], self.scale.item())
        return s


class CheckPointer(object):
    def __init__(self, args, model=None, optimizer=None):
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.model_path = os.path.join(args['model.dir'], 'weights', args['model.name'])
        self.out_path = os.path.join(args['out.dir'], 'weights', args['model.name'])
        self.last_ckpt = os.path.join(self.model_path, 'checkpoint.pth.tar')
        self.best_ckpt = os.path.join(self.model_path, 'model_best.pth.tar')
        self.out_last_ckpt = os.path.join(self.out_path, 'checkpoint.pth.tar')
        self.out_best_ckpt = os.path.join(self.out_path, 'model_best.pth.tar')

    def restore_model(self, ckpt='last', model=True,
                      optimizer=True, strict=True):
        if not os.path.exists(self.model_path):
            assert False, "Model is not found at {}".format(self.model_path)
        self.last_ckpt = os.path.join(self.model_path, 'checkpoint.pth.tar')
        self.best_ckpt = os.path.join(self.model_path, 'model_best.pth.tar')
        ckpt_path = self.last_ckpt if ckpt == 'last' else self.best_ckpt
        if os.path.isfile(ckpt_path):
            print("=> loading {} checkpoint '{}'".format(ckpt, ckpt_path))
            ch = torch.load(ckpt_path, map_location=device)
            if self.model is not None and model:
                self.model.load_state_dict(ch['state_dict'], strict=strict)
            if self.optimizer is not None and optimizer:
                self.optimizer.load_state_dict(ch['optimizer'])
        else:
            assert False, "No checkpoint! %s" % ckpt_path

        return ch.get('epoch', None), ch.get('best_val_loss', None), ch.get('best_val_acc', None)

    def save_checkpoint(self, epoch, best_val_acc, best_val_loss,
                        is_best, filename='checkpoint.pth.tar',
                        optimizer=None, state_dict=None, extra=None):
        state_dict = self.model.state_dict() if state_dict is None else state_dict
        state = {'epoch': epoch + 1,
                 'args': self.args,
                 'state_dict': state_dict,
                 'best_val_acc': best_val_acc,
                 'best_val_loss': best_val_loss}

        if extra is not None:
            state.update(extra)
        if optimizer is not None:
            state['optimizer'] = optimizer.state_dict()

        model_path = check_dir(self.out_path, True)
        torch.save(state, os.path.join(model_path, filename))
        if is_best:
            shutil.copyfile(os.path.join(model_path, filename),
                            os.path.join(model_path, 'model_best.pth.tar'))
    def restore_out_model(self, ckpt='last', model=True,
                      optimizer=True, strict=True):
        if not os.path.exists(self.out_path):
            assert False, "Model is not found at {}".format(self.model_path)
        self.out_last_ckpt = os.path.join(self.out_path, 'checkpoint.pth.tar')
        self.out_best_ckpt = os.path.join(self.out_path, 'model_best.pth.tar')
        ckpt_path = self.out_last_ckpt if ckpt == 'last' else self.out_best_ckpt

        if os.path.isfile(ckpt_path):
            print("=> loading {} checkpoint '{}'".format(ckpt, ckpt_path))
            ch = torch.load(ckpt_path, map_location=device)
            if self.model is not None and model:
                self.model.load_state_dict(ch['state_dict'], strict=strict)
            if self.optimizer is not None and optimizer:
                self.optimizer.load_state_dict(ch['optimizer'])
        else:
            assert False, "No checkpoint! %s" % ckpt_path

        return ch.get('epoch', None), ch.get('best_val_loss', None), ch.get('best_val_acc', None)


class UniformStepLR(object):
    def __init__(self, optimizer, args, start_iter):
        self.iter = start_iter
        self.max_iter = args['train.max_iter']
        step_iters = self.compute_milestones(args)
        self.lr_scheduler = MultiStepLR(
            optimizer, milestones=step_iters, last_epoch=start_iter-1,
            gamma=args['train.lr_decay_step_gamma'])

    def step(self, _iter):
        self.iter += 1
        self.lr_scheduler.step()
        stop_training = self.iter >= self.max_iter
        return stop_training

    def compute_milestones(self, args):
        # pdb.set_trace()
        max_iter = args['train.max_iter']
        step_size = max_iter / args['train.lr_decay_step_freq']
        step_iters = [0]
        while step_iters[-1] < max_iter:
            step_iters.append(step_iters[-1] + step_size)
        return step_iters[1:]


class ExpDecayLR(object):
    def __init__(self, optimizer, args, start_iter):
        self.iter = start_iter
        self.max_iter = args['train.max_iter']
        self.start_decay_iter = args['train.exp_decay_start_iter']
        gamma = self.compute_gamma(args)
        schedule_start = max(start_iter - self.start_decay_iter, 0) - 1
        self.lr_scheduler = ExponentialLR(optimizer, gamma=gamma,
                                          last_epoch=schedule_start)

    def step(self, _iter):
        self.iter += 1
        if _iter > self.start_decay_iter:
            self.lr_scheduler.step()
        stop_training = self.iter >= self.max_iter
        return stop_training

    def compute_gamma(self, args):
        last_iter, start_iter = args['train.max_iter'], args['train.exp_decay_start_iter']
        start_lr, last_lr = args['train.learning_rate'], args['train.exp_decay_final_lr']
        return np.power(last_lr / start_lr, 1 / (last_iter - start_iter))


class CosineAnnealRestartLR(object):
    def __init__(self, optimizer, args, start_iter):
        self.iter = start_iter
        self.max_iter = args['train.max_iter']
        self.lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer, args['train.cosine_anneal_freq'], last_epoch=start_iter-1)
        # self.lr_scheduler = CosineAnnealingLR(
        #     optimizer, args['train.cosine_anneal_freq'], last_epoch=start_iter-1)

    def step(self, _iter):
        self.iter += 1
        self.lr_scheduler.step(_iter)
        stop_training = self.iter >= self.max_iter
        return stop_training
def labels_to_episode_labels(labels):
    uni_labels = labels.unique()
    eposide_labels = torch.zeros(list(labels.size())).to(labels.device)
    for i in range(len(uni_labels)):
        eposide_labels[labels == uni_labels[i]] = i
    return eposide_labels

class WeightAnnealing(nn.Module):
    """WeightAnnealing"""
    def __init__(self, T, alpha=10):
        super(WeightAnnealing, self).__init__()
        self.T = T
        self.alpha = alpha

    def forward(self, t, opt='exp'):
        if t > self.T:
            return 0
        if opt == 'exp':
            return 1-np.exp(self.alpha*((t)/self.T-1))
        if opt == 'log':
            return np.exp(-(t)/self.T*self.alpha)
        if opt == 'linear':
            return 1-(t)/self.T


