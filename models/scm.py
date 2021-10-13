# This code is adapted from https://github.com/peymanbateni/simple-cnaps
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np

NUM_SAMPLES=1


def scm(context_features, context_labels, target_features):
    class_representations = OrderedDict()  # Dictionary mapping class label (integer) to encoded representation
        
    """
    SCM: in addition to saving class representations, Simple CNAPS uses a separate
    ordered dictionary for saving the class percision matrices for use when infering on
    query examples.
    """
    class_precision_matrices = OrderedDict() # Dictionary mapping class label (integer) to regularized precision matrices estimated

    class_representations, class_precision_matrices = build_class_reps_and_covariance_estimates(context_features, context_labels)
    class_means = torch.stack(list(class_representations.values())).squeeze(1)
    class_precision_matrices = torch.stack(list(class_precision_matrices.values()))
    number_of_classes = class_means.size(0)
    number_of_targets = target_features.size(0)
    repeated_target = target_features.repeat(1, number_of_classes).view(-1, class_means.size(1))
    repeated_class_means = class_means.repeat(number_of_targets, 1)
    repeated_difference = (repeated_class_means - repeated_target)
    repeated_difference = repeated_difference.view(number_of_targets, number_of_classes, repeated_difference.size(1)).permute(1, 0, 2)
    first_half = torch.matmul(repeated_difference, class_precision_matrices)
    sample_logits = torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1,0) * -1
    class_representations.clear()
    # class_precision_matrices.clear()
    return split_first_dim_linear(sample_logits, [NUM_SAMPLES, target_features.size(0)])

def build_class_reps_and_covariance_estimates(context_features, context_labels):
    """
    Construct and return class level representations and class covariance estimattes for each class in task.
    :param context_features: (torch.tensor) Adapted feature representation for each image in the context set.
    :param context_labels: (torch.tensor) Label for each image in the context set.
    :return: (void) Updates the internal class representation and class covariance estimates dictionary.
    """
    """
    SCM: calculating a task level covariance estimate using the provided function.
    """
    class_representations = OrderedDict()  # Dictionary mapping class label (integer) to encoded representation
    """
    SCM: in addition to saving class representations, Simple CNAPS uses a separate
    ordered dictionary for saving the class percision matrices for use when infering on
    query examples.
    """
    class_precision_matrices = OrderedDict() # Dictionary mapping class label (integer) to regularized precision matrices estimated
    task_covariance_estimate = estimate_cov(context_features)
    for c in torch.unique(context_labels):
        # filter out feature vectors which have class c
        class_features = torch.index_select(context_features, 0, extract_class_indices(context_labels, c))
        # mean pooling examples to form class means
        class_rep = mean_pooling(class_features)
        # updating the class representations dictionary with the mean pooled representation
        class_representations[c.item()] = class_rep
        """
        Calculating the mixing ratio lambda_k_tau for regularizing the class level estimate with the task level estimate."
        Then using this ratio, to mix the two estimate; further regularizing with the identity matrix to assure invertability, and then
        inverting the resulting matrix, to obtain the regularized precision matrix. This tensor is then saved in the corresponding
        dictionary for use later in infering of the query data points.
        """
        lambda_k_tau = (class_features.size(0) / (class_features.size(0) + 1))
        class_precision_matrices[c.item()] = torch.inverse((lambda_k_tau * estimate_cov(class_features)) + ((1 - lambda_k_tau) * task_covariance_estimate) \
                + torch.eye(class_features.size(1), class_features.size(1)).cuda(0))
    return class_representations, class_precision_matrices


    
def estimate_cov(examples, rowvar=False, inplace=False):
    """
    SCM: unction based on the suggested implementation of Modar Tensai
    and his answer as noted in: https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/5

    Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        examples: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    """
    if examples.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if examples.dim() < 2:
        examples = examples.view(1, -1)
    if not rowvar and examples.size(0) != 1:
        examples = examples.t()
    factor = 1.0 / (examples.size(1) - 1)
    if inplace:
        examples -= torch.mean(examples, dim=1, keepdim=True)
    else:
        examples = examples - torch.mean(examples, dim=1, keepdim=True)
    examples_t = examples.t()
    return factor * examples.matmul(examples_t).squeeze()

def split_first_dim_linear(x, first_two_dims):
    """
    Undo the stacking operation
    """
    x_shape = x.size()
    new_shape = first_two_dims
    if len(x_shape) > 1:
        new_shape += [x_shape[-1]]
    return x.view(new_shape)

def mean_pooling(x):
    return torch.mean(x, dim=0, keepdim=True)

# @staticmethod
def extract_class_indices(labels, which_class):
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector



