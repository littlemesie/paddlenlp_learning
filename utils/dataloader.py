# -*- coding:utf-8 -*-

"""
@date: 2024/1/24 下午4:30
@summary: 数据加载
"""
import paddle
import pandas as pd

def read_text_pair(data_path):
    """Reads data."""
    data_df = pd.read_csv(data_path)
    data_df = data_df.dropna()
    for _, row in data_df.iterrows():
        yield {"text": row['text'], "label": row['label']}

def label2ids(label_path):
    label2id = {}
    data_df = pd.read_csv(label_path)
    data_df = data_df.dropna()
    for index, row in data_df.iterrows():
        label = row['label']
        label2id[label] = index
    return label2id

def get_dev_pair(dev_text_pair_file):
    data_df = pd.read_csv(dev_text_pair_file)
    text2similar_text = {}
    texts = []
    for index, row in data_df.iterrows():
        text2similar_text[row['text']] = row['label']
        texts.append({"text": row['text']})
    return texts, text2similar_text

def convert_token(example, tokenizer, max_seq_length=512, pad_to_max_seq_len=False):
    """
    Builds model inputs from a sequence.
    A BERT sequence has the following format:
    - single sequence: ``[CLS] X [SEP]``
    Args:
        example(obj:`list(str)`): The list of text to be converted to ids.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer`
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization.
            Sequences longer than this will be truncated, sequences shorter will be padded.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.
    Returns:
        input_ids(obj:`list[int]`): The list of query token ids.
        token_type_ids(obj: `list[int]`): List of query sequence pair mask.
    """

    result = []
    for key, text in example.items():
        encoded_inputs = tokenizer(text=text, max_seq_len=max_seq_length, pad_to_max_seq_len=pad_to_max_seq_len)
        input_ids = encoded_inputs["input_ids"]
        token_type_ids = encoded_inputs["token_type_ids"]
        result += [input_ids, token_type_ids]
    return result

def create_dataloader(dataset, mode="train", batch_size=8, batchify_fn=None, trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)
    shuffle = True if mode == "train" else False
    if mode == "train":
        batch_sampler = paddle.io.DistributedBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(dataset=dataset, batch_sampler=batch_sampler, collate_fn=batchify_fn, return_list=True)