# %matplotlib inline
import matplotlib.pyplot as plt

import sys
import torch
import numpy as np

sys.path.insert(0, './')

from isegm.utils import vis, exp
from isegm.inference import utils
from isegm.inference.evaluation import evaluate_dataset, evaluate_sample

DATASET = 'EPIC-Kitchen'
cfg = exp.load_config_file('./config.yml', return_edict=True)
dataset = utils.get_dataset(DATASET, cfg)