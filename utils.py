import os
import random
import torch
import numpy as np
from time import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class ConfusionMatrix():
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.mat = np.zeros([n_classes, n_classes])

    def update_mat(self, preds, labels, idxs):
        idxs = np.array(idxs)
        real_pred = idxs[preds]
        real_labels = idxs[labels]
        self.mat[real_pred, real_labels] += 1

    def get_mat(self):
        return self.mat


class Accumulator():
    def __init__(self, max_size=2000):
        self.max_size = max_size
        self.ac = np.empty(0)

    def append(self, v):
        self.ac = np.append(self.ac[-self.max_size:], v)

    def reset(self):
        self.ac = np.empty(0)

    def mean(self, last=None):
        last = last if last else self.max_size
        return self.ac[-last:].mean()


class IterBeat():
    def __init__(self, freq, length=None):
        self.length = length
        self.freq = freq

    def step(self, i):
        if i == 0:
            self.t = time()
            self.lastcall = 0
        else:
            if ((i % self.freq) == 0) or ((i + 1) == self.length):
                t = time()
                print('{0} / {1} ---- {2:.2f} it/sec'.format(
                    i, self.length, (i - self.lastcall) / (t - self.t)))
                self.lastcall = i
                self.t = t


class SerializableArray(object):
    def __init__(self, array):
        self.shape = array.shape
        self.data = array.tobytes()
        self.dtype = array.dtype

    def get(self):
        array = np.frombuffer(self.data, self.dtype)
        return np.reshape(array, self.shape)


def print_res(array, name, file=None, prec=4, mult=1):
    array = np.array(array) * mult
    mean, std = np.mean(array), np.std(array)
    conf = 1.96 * std / np.sqrt(len(array))
    stat_string = ("test {:s}: {:0.%df} +/- {:0.%df}"
                   % (prec, prec)).format(name, mean, conf)
    print(stat_string)
    if file is not None:
        with open(file, 'a+') as f:
            f.write(stat_string + '\n')


def process_copies(embeddings, labels, args):
    n_copy = args['test.n_copy']
    test_embeddings = embeddings.view(
        args['data.test_query'] * args['data.test_way'],
        n_copy, -1).mean(dim=1)
    return test_embeddings, labels[0::n_copy]


def set_determ(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def merge_dicts(dicts, torch_stack=True):
    def stack_fn(l):
        if isinstance(l[0], torch.Tensor):
            return torch.stack(l)
        elif isinstance(l[0], str):
            return l
        else:
            return torch.tensor(l)

    keys = dicts[0].keys()
    new_dict = {key: [] for key in keys}
    for key in keys:
        for d in dicts:
            new_dict[key].append(d[key])
    if torch_stack:
        for key in keys:
            new_dict[key] = stack_fn(new_dict[key])
    return new_dict


def voting(preds, pref_ind=0):
    n_models = len(preds)
    n_test = len(preds[0])
    final_preds = []
    for i in range(n_test):
        cur_preds = [preds[k][i] for k in range(n_models)]
        classes, counts = np.unique(cur_preds, return_counts=True)
        if (counts == max(counts)).sum() > 1:
            final_preds.append(preds[pref_ind][i])
        else:
            final_preds.append(classes[np.argmax(counts)])
    return final_preds


def agreement(preds):
    n_preds = preds.shape[0]
    mat = np.zeros((n_preds, n_preds))
    for i in range(n_preds):
        for j in range(i, n_preds):
            mat[i, j] = mat[j, i] = (
                preds[i] == preds[j]).astype('float').mean()
    return mat


def read_textfile(filename, skip_last_line=True):
    with open(filename, 'r') as f:
        container = f.read().split('\n')
        if skip_last_line:
            container = container[:-1]
    return container


def check_dir(dirname, verbose=True):
    """This function creates a directory
    in case it doesn't exist"""
    try:
        # Create target Directory
        os.makedirs(dirname)
        if verbose:
            print("Directory ", dirname, " was created")
    except FileExistsError:
        if verbose:
            print("Directory ", dirname, " already exists")
    return dirname
