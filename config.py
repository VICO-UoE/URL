import argparse
from paths import PROJECT_ROOT

parser = argparse.ArgumentParser(description='Train prototypical networks')

# data args
parser.add_argument('--data.train', type=str, default='cu_birds', metavar='TRAINSETS', nargs='+', help="Datasets for training extractors")
parser.add_argument('--data.val', type=str, default='cu_birds', metavar='VALSETS', nargs='+',
                    help="Datasets used for validation")
parser.add_argument('--data.test', type=str, default='cu_birds', metavar='TESTSETS', nargs='+',
                    help="Datasets used for testing")
parser.add_argument('--data.num_workers', type=int, default=32, metavar='NEPOCHS',
                    help="Number of workers that pre-process images in parallel")

# model args
default_model_name = 'noname'
parser.add_argument('--model.name', type=str, default=default_model_name, metavar='MODELNAME',
                    help="A name you give to the extractor".format(default_model_name))
parser.add_argument('--model.backbone', default='resnet18', help="Use ResNet18 for experiments (default: False)")
parser.add_argument('--model.classifier', type=str, default='linear', choices=['none', 'linear', 'cosine'], help="Do classification using cosine similatity between activations and weights")
parser.add_argument('--model.dropout', type=float, default=0, help="Adding dropout inside a basic block of widenet")
parser.add_argument('--model.pretrained', action='store_true', help="Using pretrained model for learning or not")
# adaptor args
parser.add_argument('--adaptor.opt', type=str, default='linear', help="type of adaptor, linear or nonlinear")

# train args
parser.add_argument('--train.batch_size', type=int, default=16, metavar='BS',
                    help='number of images in a batch')
parser.add_argument('--train.max_iter', type=int, default=500000, metavar='NEPOCHS',
                    help='number of epochs to train (default: 10000)')
parser.add_argument('--train.weight_decay', type=float, default=7e-4, metavar='WD',
                    help="weight decay coef")
parser.add_argument('--train.optimizer', type=str, default='momentum', metavar='OPTIM',
                    help='optimization method (default: momentum)')
parser.add_argument('--train.learning_rate', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--train.sigma', type=float, default=1, metavar='SIGMA',
                    help='weight of CKA loss on features')
parser.add_argument('--train.beta', type=float, default=1, metavar='BETA',
                    help='weight of KL-divergence loss on logits')
parser.add_argument('--train.lr_policy', type=str, default='cosine', metavar='LR_policy',
                    help='learning rate decay policy')
parser.add_argument('--train.lr_decay_step_gamma', type=int, default=1e-1, metavar='DECAY_GAMMA',
                    help='the value to divide learning rate by when decayin lr')
parser.add_argument('--train.lr_decay_step_freq', type=int, default=10000, metavar='DECAY_FREQ',
                    help='the value to divide learning rate by when decayin lr')
parser.add_argument('--train.exp_decay_final_lr', type=float, default=8e-5, metavar='FINAL_LR',
                    help='the value to divide learning rate by when decayin lr')
parser.add_argument('--train.exp_decay_start_iter', type=int, default=30000, metavar='START_ITER',
                    help='the value to divide learning rate by when decayin lr')
parser.add_argument('--train.cosine_anneal_freq', type=int, default=4000, metavar='ANNEAL_FREQ',
                    help='the value to divide learning rate by when decayin lr')
parser.add_argument('--train.nesterov_momentum', action='store_true', help="If to augment query images in order to avearge the embeddings")

# evaluation during training
parser.add_argument('--train.eval_freq', type=int, default=5000, metavar='EVAL_FREQ',
                    help='How often to evaluate model during training')
parser.add_argument('--train.eval_size', type=int, default=300, metavar='EVAL_SIZE',
                    help='How many episodes to sample for validation')
parser.add_argument('--train.resume', type=int, default=1, metavar='RESUME_TRAIN',
                    help="Resume training starting from the last checkpoint (default: True)")


# creating a database of features
parser.add_argument('--dump.name', type=str, default='', metavar='DUMP_NAME',
                    help='Name for dumped dataset of features')
parser.add_argument('--dump.mode', type=str, default='test', metavar='DUMP_MODE',
                    help='What split of the original dataset to dump')
parser.add_argument('--dump.size', type=int, default=600, metavar='DUMP_SIZE',
                    help='Howe many episodes to dump')


# test args
parser.add_argument('--test.size', type=int, default=600, metavar='TEST_SIZE',
                    help='The number of test episodes sampled')
parser.add_argument('--test.mode', type=str, choices=['mdl', 'sdl'], default='mdl', metavar='TEST_MODE',
                    help="Test mode: multi-domain learning (mdl) or single-domain learning (sdl) settings")
parser.add_argument('--test.type', type=str, choices=['standard', '1shot', '5shot'], default='standard', metavar='LOSS_FN',
                    help="meta-test type, standard varying number of ways and shots as in Meta-Dataset, 1shot for five-way-one-shot and 5shot for varying-way-five-shot evaluation.")
parser.add_argument('--test.distance', type=str, choices=['cos', 'l2'], default='cos', metavar='DISTANCE_FN',
                    help="feature similarity function")
parser.add_argument('--test.loss-opt', type=str, choices=['ncc', 'knn', 'lr', 'svm', 'scm'], default='ncc', metavar='LOSS_FN',
                    help="Loss function for meta-testing, knn or prototype loss (ncc), Support Vector Machine (svm), Logistic Regression (lr) or Mahalanobis Distance (scm)")
parser.add_argument('--test.feature-norm', type=str, choices=['l2', 'none'], default='none', metavar='LOSS_FN',
                    help="normalization options")

# task-specific adapters
parser.add_argument('--test.tsa-ad-type', type=str, choices=['residual', 'serial', 'none'], default='none', metavar='TSA_AD_TYPE',
                    help="adapter type")
parser.add_argument('--test.tsa-ad-form', type=str, choices=['matrix', 'vector', 'none'], default='matrix', metavar='TSA_AD_FORM',
                    help="adapter form")
parser.add_argument('--test.tsa-opt', type=str, choices=['alpha', 'beta', 'alpha+beta'], default='alpha+beta', metavar='TSA_OPT',
                    help="task adaptation option")
parser.add_argument('--test.tsa-init', type=str, choices=['random', 'eye'], default='eye', metavar='TSA_INIT',
                    help="initialization for adapter")

# path args
parser.add_argument('--model.dir', default='', type=str, metavar='PATH',
                    help='path of single domain learning models')
parser.add_argument('--out.dir', default='', type=str, metavar='PATH',
                    help='directory to output the result and checkpoints')
parser.add_argument('--source', default='', type=str, metavar='PATH',
                    help='path of pretrained model')


# log args
args = vars(parser.parse_args())
if not args['model.dir']:
    args['model.dir'] = PROJECT_ROOT
if not args['out.dir']:
    args['out.dir'] = args['model.dir']

BATCHSIZES = {
                "ilsvrc_2012": 448,
                "omniglot": 64,
                "aircraft": 64,
                "cu_birds": 64,
                "dtd": 64,
                "quickdraw": 64,
                "fungi": 64,
                "vgg_flower": 64
                }

LOSSWEIGHTS = {
                "ilsvrc_2012": 1,
                "omniglot": 1,
                "aircraft": 1,
                "cu_birds": 1,
                "dtd": 1,
                "quickdraw": 1,
                "fungi": 1,
                "vgg_flower": 1
                }

# lambda^f in our paper
KDFLOSSWEIGHTS = {
                    "ilsvrc_2012": 4,
                    "omniglot": 1,
                    "aircraft": 1,
                    "cu_birds": 1,
                    "dtd": 1,
                    "quickdraw": 1,
                    "fungi": 1,
                    "vgg_flower": 1
                }
# lambda^p in our paper
KDPLOSSWEIGHTS = {
                    "ilsvrc_2012": 4,
                    "omniglot": 1,
                    "aircraft": 1,
                    "cu_birds": 1,
                    "dtd": 1,
                    "quickdraw": 1,
                    "fungi": 1,
                    "vgg_flower": 1
                }
# k in our paper
KDANNEALING = {
                    "ilsvrc_2012": 5,
                    "omniglot": 2,
                    "aircraft": 1,
                    "cu_birds": 1,
                    "dtd": 1,
                    "quickdraw": 2,
                    "fungi": 2,
                    "vgg_flower": 1
                }