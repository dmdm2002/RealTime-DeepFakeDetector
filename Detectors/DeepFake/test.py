import time
import os
import torch
import tqdm
import pandas as pd

import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from Detectors.DeepFake.utils.dataset import get_loader
from Detectors.DeepFake.utils.functions import get_configs, cal_metrics
from Detectors.DeepFake.repvgg_qimtozed import RepVGGWholeQuant
from torchmetrics.classification import ConfusionMatrix


class Test:
    def __init__(self, cfg: dict, ep: int):
        self.cfg = cfg
        torch.manual_seed(self.cfg['seed'])
        torch.cuda.manual_seed_all(self.cfg['seed'])

        if self.cfg['do_logging']:
            os.makedirs(f"{self.cfg['log_path']}/tensorboard", exist_ok=True)
            self.summary = SummaryWriter(f"{self.cfg['log_path']}/tensorboard")

        self.model = torch.load(f"{self.cfg['ckp_path']}/best_{ep}.pt")
        # self.model = RepVGGWholeQuant(self.model, 'all')
        self.conf_mat = ConfusionMatrix(task="binary", num_classes=2)

    def test(self):
        fps_full = 0
        test_score = {'acc': 0, 'apcer': 0, 'bpcer': 0, 'acer': 0}
        tp, tn, fp, fn = 0, 0, 0, 0
        test_loader = get_loader(train=False,
                             image_size=self.cfg['image_size'],
                             batch_size=1,
                             fake_path=self.cfg['te_fake_dataset_path'],
                             live_path=self.cfg['te_live_dataset_path'])
        pred_label_list = []

        with torch.no_grad():
            self.model.eval()
            for _, (image, label) in enumerate(tqdm.tqdm(test_loader, desc=f"[Test]")):
                image = image.to(self.cfg['device'])
                label = label.to(self.cfg['device'])

                prevTime = time.time()

                logit = self.model(image)

                # 프레임 수 계산
                curTime = time.time()  # current time
                fps = 1 / (curTime - prevTime)
                fps_full += fps
                prevTime = curTime

                [tn_batch, fp_batch], [fn_batch, tp_batch] = self.conf_mat(logit.argmax(1).cpu(), label.cpu())
                tp += tp_batch
                tn += tn_batch
                fp += fp_batch
                fn += fn_batch

                prov = nn.functional.softmax(logit)
                # print(f'Label: {label.cpu().numpy()[0]}, Pred: {prov.cpu().numpy()[0][1]}')
                pred_label_list.append([label.cpu().numpy()[0], prov.cpu().numpy()[0][1]])

        test_score['acc'] = (tp + tn) / (tp + fn + fp + tn) if (tp + fn + fp + tn) != 0 else 0
        test_score['apcer'] = fp / (tn + fp) if (tn + fp) != 0 else 0
        test_score['bpcer'] = fn / (fn + tp) if (fn + tp) != 0 else 0
        test_score['acer'] = (test_score['apcer'] + test_score['bpcer']) / 2


        print(f'-----------------------------[TEST]----------------------------')
        print(f'FPS: {fps_full / 10905}')
        print(f"Attack APCER: {test_score['apcer'] * 100}  |  Attack BPCER: {test_score['bpcer'] * 100}  |  Attack ACER: {test_score['acer'] * 100}")
        print('-------------------------------------------------------------------')

        df = pd.DataFrame(data=pred_label_list, columns=['label', 'predict'])
        df.to_csv('./prediction.csv', index=False)


if __name__ == '__main__':
    cfg = get_configs('D:/Side/CodeGate/backup/DeepFake/RepVGG/try_3/log/train_parameters.yml')
    te = Test(cfg, 13)
    te.test()
