import timm
import os
import torch
import tqdm

import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from Detectors.DeepFake.utils.dataset import get_loader
from Detectors.DeepFake.utils.functions import save_configs, cal_metrics
from torchmetrics.classification import ConfusionMatrix


class Train:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        torch.manual_seed(self.cfg['seed'])
        torch.cuda.manual_seed_all(self.cfg['seed'])

        if self.cfg['do_logging']:
            os.makedirs(f"{self.cfg['log_path']}/tensorboard", exist_ok=True)
            self.summary = SummaryWriter(f"{self.cfg['log_path']}/tensorboard")
            save_configs(self.cfg)
        if self.cfg['do_ckp_save']:
            os.makedirs(f"{self.cfg['ckp_path']}", exist_ok=True)

        self.model = timm.create_model('repvgg_a2.rvgg_in1k', pretrained=True, num_classes=2).to(self.cfg['device'])
        self.optimizer = optim.Adam(self.model.parameters(), self.cfg['lr'], (self.cfg['b1'], self.cfg['b2']))

        self.tr_loader = get_loader(train=True,
                                    image_size=self.cfg['image_size'],
                                    crop=self.cfg['crop'],
                                    jitter=self.cfg['jitter'],
                                    noise=self.cfg['noise'],
                                    batch_size=self.cfg['batch_size'],
                                    fake_path=self.cfg['tr_fake_dataset_path'],
                                    live_path=self.cfg['tr_live_dataset_path'])

        self.val_loader = get_loader(train=False,
                                     image_size=self.cfg['image_size'],
                                     batch_size=self.cfg['batch_size'],
                                     fake_path=self.cfg['val_fake_dataset_path'],
                                     live_path=self.cfg['val_live_dataset_path'])

        self.criterion = nn.CrossEntropyLoss()
        self.conf_mat = ConfusionMatrix(task="binary", num_classes=2)

    def train(self):
        best_score = {'epoch': 0, 'acc': 0, 'apcer': 0, 'bpcer': 0, 'acer': 0}
        tr_score = {'acc': 0, 'apcer': 0, 'bpcer': 0, 'acer': 0}
        val_score = {'acc': 0, 'apcer': 0, 'bpcer': 0, 'acer': 0}

        for ep in range(self.cfg['epoch']):
            self.model.train()

            losses = 0

            tr_tp, tr_tn, tr_fp, tr_fn = 0, 0, 0, 0
            val_tp, val_tn, val_fp, val_fn = 0, 0, 0, 0

            for _, (image, label) in enumerate(tqdm.tqdm(self.tr_loader, desc=f"[Train-->{ep}/{self.cfg['epoch']}]")):
                image = image.to(self.cfg['device'])
                label = label.to(self.cfg['device'])

                logit = self.model(image)
                loss = self.criterion(logit, label)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                [tn_batch, fp_batch], [fn_batch, tp_batch] = self.conf_mat(logit.argmax(1).cpu(), label.cpu())
                tr_tp += tp_batch
                tr_tn += tn_batch
                tr_fp += fp_batch
                tr_fn += fn_batch

                losses += loss.item()

            with torch.no_grad():
                self.model.eval()
                for _, (x_o, label) in enumerate(tqdm.tqdm(self.val_loader, desc=f"[Test-->{ep}/{self.cfg['epoch']}]")):
                    x_o = x_o.to(self.cfg['device'])
                    label = label.to(self.cfg['device'])

                    logit = self.model(x_o)
                    [tn_batch, fp_batch], [fn_batch, tp_batch] = self.conf_mat(logit.argmax(1).cpu(), label.cpu())
                    val_tp += tp_batch
                    val_tn += tn_batch
                    val_fp += fp_batch
                    val_fn += fn_batch

            losses = losses / len(self.tr_loader)
            tr_score['acc'], tr_score['apcer'], tr_score['bpcer'], tr_score['acer'] = cal_metrics(tr_tp, tr_tn, tr_fp, tr_fn)
            val_score['acc'], val_score['apcer'], val_score['bpcer'], val_score['acer'] = cal_metrics(val_tp, val_tn, val_fp, val_fn)

            if best_score['acc'] <= val_score['acc']:
                best_score['acc'] = val_score['acc']
                best_score['apcer'] = val_score['apcer']
                best_score['bpcer'] = val_score['bpcer']
                best_score['acer'] = val_score['acer']
                best_score['epoch'] = ep

                torch.save(self.model, f"{self.cfg['ckp_path']}/best_{ep}.pt")

            if self.cfg['do_logging']:
                self.summary.add_scalar('Train/loss', losses, ep)
                self.summary.add_scalar('Train/Accuracy', tr_score['acc'], ep)
                self.summary.add_scalar('Train/APCER', tr_score['apcer'], ep)
                self.summary.add_scalar('Train/BPCER', tr_score['bpcer'], ep)
                self.summary.add_scalar('Train/ACER', tr_score['acer'], ep)

                self.summary.add_scalar('Test/Accuracy', val_score['acc'], ep)
                self.summary.add_scalar('Test/APCER', val_score['apcer'], ep)
                self.summary.add_scalar('Test/BPCER', val_score['bpcer'], ep)
                self.summary.add_scalar('Test/ACER', val_score['acer'], ep)

                f = open(f"{self.cfg['log_path']}/ACC_LOSS_LOG.txt", 'a', encoding='utf-8')
                f.write(
                    f"-------------------------------------------------------------------\n"
                )
                f.write(
                    f"Epoch {ep}\n"
                )
                f.write(
                    f"Train [Accuracy: {tr_score['acc'] * 100}\tAPCER: {tr_score['apcer'] * 100}\tBPCER: {tr_score['bpcer'] * 100}\tACER: {tr_score['acer'] * 100}]\n")
                f.write(
                    f"Test [Accuracy: {val_score['acc'] * 100}\tAPCER: {val_score['apcer'] * 100}\tBPCER: {val_score['bpcer'] * 100}\tACER: {val_score['acer'] * 100}]\n")
                f.write(
                    f"-------------------------------------------------------------------\n"
                )
                f.close()

            if self.cfg['do_print']:
                print('\n')
                print('-------------------------------------------------------------------')
                print(f"Now Epoch: {ep}/{self.cfg['epoch']}")
                print(
                    f"Train [APCER: {tr_score['apcer'] * 100}  |  BPCER: {tr_score['bpcer'] * 100}  |  ACER: {tr_score['acer'] * 100}]")
                print(
                    f"Validation [APCER: {val_score['apcer'] * 100}  |  BPCER: {val_score['bpcer'] * 100}  |  ACER: {val_score['acer'] * 100}]")
                print('-------------------------------------------------------------------')
                print(f"Best acc epoch: {best_score['epoch']}")
                print(f"Best acc: {best_score['acc']}")
                print(
                    f"APCER: {best_score['apcer'] * 100}  |  BPCER: {best_score['bpcer'] * 100}  |  ACER: {best_score['acer'] * 100}")
                print('-------------------------------------------------------------------')

            if self.cfg['do_ckp_save']:
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "Adam_state_dict": self.optimizer.state_dict(),
                        "epoch": ep,
                    },
                    os.path.join(f"{self.cfg['ckp_path']}/", f"{ep}.pth.tar"),
                )

        if self.cfg['do_logging']:
            f = open(f"{self.cfg['log_path']}/best_score.txt", 'w', encoding='utf-8')
            f.write(f'[Best Score!!]\n')
            f.write(
                f'epoch : {best_score["epoch"]}\nAcc : {best_score["acc"] * 100}\nAPCER: {best_score["apcer"] * 100}\nBPCER: {best_score["bpcer"] * 100}\nACER: {best_score["acer"] * 100}\n')
            f.close()