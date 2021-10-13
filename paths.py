import os
import sys

PROJECT_ROOT = '/'.join(os.path.realpath(__file__).split('/')[:-1])
META_DATASET_ROOT = os.environ['META_DATASET_ROOT']
META_RECORDS_ROOT = os.environ['RECORDS']
META_DATA_ROOT = '/'.join(META_RECORDS_ROOT.split('/')[:-1])
