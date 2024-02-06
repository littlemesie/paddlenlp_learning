# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from utils.metric import MetricReport
from tqdm import tqdm
from core.model_config import get_model_config
from utils.dataloader import label2ids

config = get_model_config('retrieval_match')


def evaluate(label2id):
    metric = MetricReport()
    text2similar = {}
    # Encoding labels as one hot
    dev_df = pd.read_csv(f"{config.data_path}/dev.csv")
    for index, row in dev_df.iterrows():
        text2similar[row['text']] = np.zeros(len(label2id))
        # One hot Encoding
        for label in row['label'].strip().split(","):
            text2similar[row['text']][label2id[label]] = 1
    pred_labels = {}
    # Convert predicted labels into one hot encoding
    with open(config.recall_result_file, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            text_arr = line.rstrip().split("\t")
            text, labels, cosine_sim = text_arr
            # One hot Encoding
            if text not in pred_labels:
                pred_labels[text] = np.zeros(len(label2id))
            if float(cosine_sim) > config.threshold:
                for label in labels.split(","):
                    pred_labels[text][label2id[label]] = float(cosine_sim)

        for text, probs in tqdm(pred_labels.items()):
            metric.update(probs, text2similar[text])

        micro_f1_score, macro_f1_score = metric.accumulate()
        print("Micro fl score: {}".format(micro_f1_score * 100))
        print("Macro f1 score: {}".format(macro_f1_score * 100))


if __name__ == "__main__":
    label2id = label2ids(f"{config.data_path}/label.csv")
    evaluate(label2id)
