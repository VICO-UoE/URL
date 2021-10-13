DATASET_MODELS_RESNET18 = {
    'ilsvrc_2012': 'imagenet-net',
    'omniglot': 'omniglot-net',
    'aircraft': 'aircraft-net',
    'cu_birds': 'birds-net',
    'dtd': 'textures-net',
    'quickdraw': 'quickdraw-net',
    'fungi': 'fungi-net',
    'vgg_flower': 'vgg_flower-net'
}


DATASET_MODELS_RESNET18_PNF = {
    'omniglot': 'omniglot-film',
    'aircraft': 'aircraft-film',
    'cu_birds': 'birds-film',
    'dtd': 'textures-film',
    'quickdraw': 'quickdraw-film',
    'fungi': 'fungi-film',
    'vgg_flower': 'vgg_flower-film'
}

DATASET_MODELS_DICT = {'resnet18': DATASET_MODELS_RESNET18,
                       'resnet18_pnf': DATASET_MODELS_RESNET18_PNF}
