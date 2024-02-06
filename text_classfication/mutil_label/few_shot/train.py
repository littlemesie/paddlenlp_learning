# -*- coding:utf-8 -*-

"""
@date: 2024/2/4 下午6:00
@summary:
"""
import functools
import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_path)
import random
import time
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.io import BatchSampler, DataLoader, DistributedBatchSampler

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LinearDecayWithWarmup,
)
from paddlenlp.utils.log import logger
from core.model_config import get_model_config
from utils.dataloader import label2ids, read_dataset
from utils.metric import MetricReport

config = get_model_config('few_shot')

def set_seed(seed):
    """
    Sets random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    """
    Given a dataset, it evaluates model and computes the metric.
    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
    """

    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        labels = batch.pop("labels")
        logits = model(**batch)
        loss = criterion(logits, labels)
        probs = F.sigmoid(logits)
        losses.append(loss.numpy())
        metric.update(probs, labels)

    micro_f1_score, macro_f1_score = metric.accumulate()
    logger.info(
        "eval loss: %.5f, micro f1 score: %.5f, macro f1 score: %.5f"
        % (np.mean(losses), micro_f1_score, macro_f1_score)
    )
    model.train()
    metric.reset()

    return micro_f1_score, macro_f1_score


def preprocess_function(examples, tokenizer, max_seq_length, label_nums, is_test=False):
    """
    Builds model inputs from a sequence for sequence classification tasks
    by concatenating and adding special tokens.

    Args:
        examples(obj:`list[str]`): List of input data, containing text and label if it have label.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer`
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_length(obj:`int`): The maximum total input sequence length after tokenization.
            Sequences longer than this will be truncated, sequences shorter will be padded.
        label_nums(obj:`int`): The number of the labels.
    Returns:
        result(obj:`dict`): The preprocessed data including input_ids, token_type_ids, labels.
    """
    result = tokenizer(text=examples["text"], max_seq_len=max_seq_length)
    # One-Hot label
    if not is_test:
        result["labels"] = [float(1) if i in examples["label"] else float(0) for i in range(label_nums)]
    return result

def train():
    """"""
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    set_seed(config.seed)
    paddle.set_device(config.device)

    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    # load and preprocess dataset
    label2id = label2ids(f"{config.data_path}/label.csv")
    # id2corpus = {v: k for k, v in label2id.items()}

    train_ds = load_dataset(
        read_dataset, data_path=f"{config.data_path}/train_1.csv", label2id=label2id, lazy=False
    )
    dev_ds = load_dataset(
        read_dataset, data_path=f"{config.data_path}/dev.csv", label2id=label2id, lazy=False
    )
    print(train_ds[0])
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=config.model_path)
    trans_func = functools.partial(
        preprocess_function, tokenizer=tokenizer, max_seq_length=config.max_seq_length, label_nums=len(label2id)
    )
    train_ds = train_ds.map(trans_func)
    dev_ds = dev_ds.map(trans_func)
    # batchify dataset
    collate_fn = DataCollatorWithPadding(tokenizer)
    if paddle.distributed.get_world_size() > 1:
        train_batch_sampler = DistributedBatchSampler(train_ds, batch_size=config.batch_size, shuffle=True)
    else:
        train_batch_sampler = BatchSampler(train_ds, batch_size=config.batch_size, shuffle=True)
    dev_batch_sampler = BatchSampler(dev_ds, batch_size=config.batch_size, shuffle=False)
    train_data_loader = DataLoader(dataset=train_ds, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
    dev_data_loader = DataLoader(dataset=dev_ds, batch_sampler=dev_batch_sampler, collate_fn=collate_fn)

    # define model
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_classes=len(label2id),
                                                               cache_dir=config.model_path)
    if config.init_from_ckpt and os.path.isfile(config.init_from_ckpt):
        state_dict = paddle.load(config.init_from_ckpt)
        model.set_dict(state_dict)
    model = paddle.DataParallel(model)

    num_training_steps = len(train_data_loader) * config.epochs
    lr_scheduler = LinearDecayWithWarmup(float(config.learning_rate), num_training_steps, config.warmup_steps)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=config.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
    )

    criterion = paddle.nn.BCEWithLogitsLoss()
    metric = MetricReport()

    global_step = 0
    best_f1_score = 0.8
    early_stop_count = 0
    tic_train = time.time()

    for epoch in range(1, config.epochs + 1):

        if config.early_stop and early_stop_count >= config.early_stop_nums:
            logger.info("Early stop!")
            break

        for step, batch in enumerate(train_data_loader, start=1):
            # print(batch)
            labels = batch.pop("labels")
            print(type(batch))
            logits = model(**batch)
            loss = criterion(logits, labels)
            # print(loss)
            loss.backward()
            optimizer.step()
            if config.warmup:
                lr_scheduler.step()
            optimizer.clear_grad()

            global_step += 1
            if global_step % config.logging_steps == 0 and rank == 0:
                logger.info(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, config.logging_steps / (time.time() - tic_train))
                )
                tic_train = time.time()

        early_stop_count += 1
        micro_f1_score, macro_f1_score = evaluate(model, criterion, metric, dev_data_loader)

        save_best_path = config.save_dir
        if not os.path.exists(save_best_path):
            os.makedirs(save_best_path)

        # save models
        if macro_f1_score > best_f1_score:
            early_stop_count = 0
            best_f1_score = macro_f1_score
            model._layers.save_pretrained(save_best_path)
            tokenizer.save_pretrained(save_best_path)

        if epoch == config.epochs:
            # save_param_path = os.path.join(save_best_path, "model_state.pdparams")
            # paddle.save(model.state_dict(), save_param_path)
            # tokenizer.save_pretrained(save_best_path)
            model._layers.save_pretrained(save_best_path)
            tokenizer.save_pretrained(save_best_path)

        logger.info("Current best macro f1 score: %.5f" % (best_f1_score))
    logger.info("Final best macro f1 score: %.5f" % (best_f1_score))
    logger.info("Save best macro f1 text classification model in %s" % (config.save_dir))

def predict(text):
    """predict"""
    paddle.set_device(config.device)
    model = AutoModelForSequenceClassification.from_pretrained(config.save_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.save_dir)
    label2id = label2ids(f"{config.data_path}/label.csv")
    id2label = {v: k for k, v in label2id.items()}
    t1 = time.time()
    result = tokenizer(text=text, max_seq_len=config.max_seq_length, padding='max_length')
    result['input_ids'] = paddle.to_tensor([result['input_ids']])
    result['token_type_ids'] = paddle.to_tensor([result['token_type_ids']])
    logits = model(**result)
    probs = F.sigmoid(logits).numpy()
    labels = []
    for prob in probs:
        for i, p in enumerate(prob):
            if p > 0.5:
                label = id2label[i]
                labels.append((label, round(p, 4)))
    print(labels)
    print(time.time() - t1)




if __name__ == "__main__":
    """"""
    # train()
    text = '成人大学的毕业证和普通的一样吗？'
    predict(text)
