from utils.functions import get_configs
from Detectors.DeepFake.train import Train

import torch
import gc

# print(os.path.abspath(__file__))
if __name__ == '__main__':
    torch.cuda.empty_cache()
    gc.collect()

    cfg = get_configs('./configs/train.yml')
    cls_train = Train(cfg)
    cls_train.train()
