import pandas as pd
import numpy as np

# Attack APCER: 20.025863647460938  |  Attack BPCER: 1.8754551410675049  |  Attack ACER: 10.95065975189209

thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
info = pd.read_csv('./prediction.csv')
info = info.to_numpy()

for thr in thresholds:
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for _, (label, pred) in enumerate(info):
        if pred >= thr:
            pred_label = 1
        else:
            pred_label = 0

        if pred_label == 1 and label == 0:
            fp = fp + 1
        elif pred_label == 1 and label == 1:
            tp = tp + 1
        elif pred_label == 0 and label == 1:
            fn = fn + 1
        elif pred_label == 0 and label == 0:
            tn = tn + 1

    apcer = fn / (tp + fn)
    bpcer = fp / (fp + tn)

    print('\n')
    print('---------------------------------------------------')
    print(f"Threshold : {thr}")
    print(f"APCER : {apcer * 100}")
    print(f"BPCER : {bpcer * 100}")
    print(f"ACER : {((apcer + bpcer)/2)*100}")
    print('---------------------------------------------------')
