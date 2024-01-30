# -*- coding:utf-8 -*-

"""
@date: 2024/1/24 下午4:20
@summary:
"""
import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_path)
import time
import paddle
import hnswlib
import numpy as np
import pandas as pd
import paddle.nn as nn
from functools import partial
from paddlenlp.data import Pad, Tuple
from paddlenlp.datasets import MapDataset, load_dataset
from paddlenlp.transformers import AutoModel, AutoTokenizer, LinearDecayWithWarmup
from core.model_config import get_model_config
from text_classfication.mutil_label.retrieval_match.metric import MetricReport
from text_classfication.mutil_label.retrieval_match.model import SemanticIndexBatchNeg, SemanticIndexBase
from utils.dataloader import read_text_pair, label2ids, get_dev_pair, convert_token, create_dataloader

config = get_model_config('retrieval_match')
# data_path = '/media/mesie/F0E66F06E66ECC82/数据/paddle/baike_qa_multilabel'
# model_path = "/media/mesie/F0E66F06E66ECC82/model/paddle"

def corpus_embedding(corpus_data_loader, model):
    """"""
    all_embeddings = []
    for text_embeddings in model.get_semantic_embedding(corpus_data_loader):
        all_embeddings.append(text_embeddings.numpy())
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings

def build_index(all_embeddings, output_emb_size, hnsw_max_elements, hnsw_ef, hnsw_m):
    """"""
    index = hnswlib.Index(space="cosine", dim=output_emb_size if output_emb_size > 0 else 768)
    # Initializing index
    # max_elements - the maximum number of elements (capacity). Will throw an exception if exceeded
    # during insertion of an element.
    # The capacity can be increased by saving/loading the index, see below.
    #
    # ef_construction - controls index search speed/build speed tradeoff
    #
    # M - is tightly connected with internal dimensionality of the data. Strongly affects memory consumption (~M)
    # Higher M leads to higher accuracy/run_time at fixed ef/efConstruction
    index.init_index(max_elements=hnsw_max_elements, ef_construction=hnsw_ef, M=hnsw_m)

    # Controlling the recall by setting ef:
    # higher ef leads to better accuracy, but slower search
    index.set_ef(hnsw_ef)

    # Set number of threads used during batch search/construction
    # By default using all available cores
    index.set_num_threads(16)
    print("start build index..........")
    index.add_items(all_embeddings)
    print("Total index number:{}".format(index.get_current_count()))
    return index


@paddle.no_grad()
def evaluate(model, corpus_data_loader, query_data_loader, recall_result_file, text_list, id2corpus, label2id):
    metric = MetricReport()
    # Load pretrained semantic model
    inner_model = model._layers
    all_embeddings = corpus_embedding(corpus_data_loader, model)
    final_index = build_index(
        all_embeddings,
        output_emb_size=config.output_emb_size,
        hnsw_max_elements=config.hnsw_max_elements,
        hnsw_ef=config.hnsw_ef,
        hnsw_m=config.hnsw_m,
    )
    query_embedding = inner_model.get_semantic_embedding(query_data_loader)
    with open(recall_result_file, "w", encoding="utf-8") as f:
        for batch_index, batch_query_embedding in enumerate(query_embedding):
            recalled_idx, cosine_sims = final_index.knn_query(batch_query_embedding.numpy(), config.recall_num)
            batch_size = len(cosine_sims)
            for row_index in range(batch_size):
                text_index = config.batch_size * batch_index + row_index
                for idx, doc_idx in enumerate(recalled_idx[row_index]):
                    f.write(
                        "{}\t{}\t{}\n".format(
                            text_list[text_index]["text"], id2corpus[doc_idx], 1.0 - cosine_sims[row_index][idx]
                        )
                    )
    text2similar = {}
    dev_df = pd.read_csv(f"{config.data_path}/dev.csv")
    dev_df = dev_df.dropna()
    for index, row in dev_df.iterrows():
        text2similar[row['text']] = np.zeros(len(label2id))
        # One hot Encoding
        for label in row['label'].strip().split(","):
            text2similar[row['text']][label2id[label]] = 1
    # Convert predicted labels into one hot encoding
    pred_labels = {}
    with open(recall_result_file, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            text_arr = line.rstrip().split("\t")
            text, labels, cosine_sim = text_arr
            # One hot Encoding
            if text not in pred_labels:
                pred_labels[text] = np.zeros(len(label2id))
            if float(cosine_sim) > config.threshold:
                for label in labels.split(","):
                    pred_labels[text][label2id[label]] = float(cosine_sim)

        for text, probs in pred_labels.items():
            metric.update(probs, text2similar[text])
        micro_f1_score, macro_f1_score = metric.accumulate()
    return macro_f1_score

def train():
    """"""
    paddle.set_device(config.device)
    rank = paddle.distributed.get_rank()
    # load model
    pretrained_model = AutoModel.from_pretrained('rocketqa-zh-dureader-query-encoder', cache_dir=config.model_path)
    model = SemanticIndexBatchNeg(
        pretrained_model, margin=config.margin, scale=config.scale, output_emb_size=config.output_emb_size
    )
    tokenizer = AutoTokenizer.from_pretrained('rocketqa-zh-dureader-query-encoder', cache_dir=config.model_path)
    # train data
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # text_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),  # text_segment
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # label_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),  # label_segment
    ): [data for data in fn(samples)]
    trans_func = partial(convert_token, tokenizer=tokenizer, max_seq_length=config.max_seq_length)
    train_ds = load_dataset(read_text_pair, data_path=f"{config.data_path}/train.csv", lazy=False)
    train_data_loader = create_dataloader(
        train_ds, mode="train", batch_size=config.batch_size, batchify_fn=batchify_fn, trans_fn=trans_func
    )
    # dev data
    batchify_fn_dev = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # text_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),  # text_segment
    ): [data for data in fn(samples)]
    if config.evaluate:
        eval_func = partial(convert_token, tokenizer=tokenizer, max_seq_length=config.max_seq_length)
        label2id = label2ids(f"{config.data_path}/label.csv")
        id2corpus = {v: k for k, v in label2id.items()}
        # convert_example function's input must be dict
        corpus_list = [{idx: text} for idx, text in id2corpus.items()]
        corpus_ds = MapDataset(corpus_list)
        corpus_data_loader = create_dataloader(
            corpus_ds, mode="predict", batch_size=config.batch_size, batchify_fn=batchify_fn_dev, trans_fn=eval_func
        )
        query_func = partial(convert_token, tokenizer=tokenizer, max_seq_length=config.max_seq_length)
        text_list, _ = get_dev_pair(f"{config.data_path}/dev.csv")
        query_ds = MapDataset(text_list)
        query_data_loader = create_dataloader(
            query_ds, mode="predict", batch_size=config.batch_size, batchify_fn=batchify_fn_dev, trans_fn=query_func
        )
        if not os.path.exists(config.recall_result_dir):
            os.mkdir(config.recall_result_dir)
        recall_result_file = os.path.join(config.recall_result_dir, config.recall_result_file)


    # train model
    model = paddle.DataParallel(model)

    num_training_steps = len(train_data_loader) * config.epochs
    lr_scheduler = LinearDecayWithWarmup(float(config.learning_rate), num_training_steps, config.warmup_proportion)
    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=config.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
        grad_clip=nn.ClipGradByNorm(clip_norm=1.0),
    )
    global_step = 0
    best_score = 0.0
    tic_train = time.time()
    for epoch in range(1, config.epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids = batch
            loss = model(
                query_input_ids=query_input_ids,
                title_input_ids=title_input_ids,
                query_token_type_ids=query_token_type_ids,
                title_token_type_ids=title_token_type_ids,
            )
            global_step += 1
            if global_step % config.log_steps == 0 and rank == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, 10 / (time.time() - tic_train))
                )
                tic_train = time.time()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            if not config.evaluate and rank == 0:
                if global_step % config.save_steps == 0 and rank == 0:
                    save_dir = os.path.join(config.model_save_dir, "model_%d" % global_step)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_param_path = os.path.join(save_dir, "model_state.pdparams")
                    paddle.save(model.state_dict(), save_param_path)
                    tokenizer.save_pretrained(save_dir)
        if config.evaluate and rank == 0:
            print("evaluating")
            macro_f1_score = evaluate(
                model, corpus_data_loader, query_data_loader, recall_result_file, text_list, id2corpus, label2id
            )
            if macro_f1_score > best_score:
                best_score = macro_f1_score
                save_dir = os.path.join(config.model_save_dir, "model_best")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_param_path = os.path.join(save_dir, "model_state.pdparams")
                paddle.save(model.state_dict(), save_param_path)
                tokenizer.save_pretrained(save_dir)
                with open(os.path.join(save_dir, "train_result.txt"), "a", encoding="utf-8") as fp:
                    fp.write("epoch=%d, global_step: %d, Macro f1: %s\n" % (epoch, global_step, macro_f1_score))

    # print(train_ds)

def corpus_index(tokenizer):
    """"""
    batchify_fn_dev = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # text_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),  # text_segment
    ): [data for data in fn(samples)]
    eval_func = partial(convert_token, tokenizer=tokenizer, max_seq_length=config.max_seq_length)
    label2id = label2ids(f"{config.data_path}/label.csv")
    id2corpus = {v: k for k, v in label2id.items()}
    # convert_example function's input must be dict
    corpus_list = [{idx: text} for idx, text in id2corpus.items()]
    corpus_ds = MapDataset(corpus_list)
    corpus_data_loader = create_dataloader(
        corpus_ds, mode="predict", batch_size=config.batch_size, batchify_fn=batchify_fn_dev, trans_fn=eval_func
    )
    # load model
    pretrained_model = AutoModel.from_pretrained(config.model_name_or_path,  cache_dir=config.model_path)
    model = SemanticIndexBase(pretrained_model, output_emb_size=config.output_emb_size)
    params_path = f"{config.model_save_dir}/model_best/model_state.pdparams"
    if params_path:
        state_dict = paddle.load(params_path)
        model.set_dict(state_dict)
    model.eval()
    all_embeddings = corpus_embedding(corpus_data_loader, model)
    final_index = build_index(
        all_embeddings,
        output_emb_size=config.output_emb_size,
        hnsw_max_elements=config.hnsw_max_elements,
        hnsw_ef=config.hnsw_ef,
        hnsw_m=config.hnsw_m,
    )
    final_index.save_index("./index/model_index.bin")
    del final_index

def predict(text, tokenizer):
    """predict"""
    # load model
    pretrained_model = AutoModel.from_pretrained(config.model_name_or_path, cache_dir=config.model_path)
    model = SemanticIndexBase(pretrained_model, output_emb_size=config.output_emb_size)
    params_path = f"{config.model_save_dir}/model_best/model_state.pdparams"
    if params_path:
        state_dict = paddle.load(params_path)
        model.set_dict(state_dict)
    model.eval()
    # load index
    index = hnswlib.Index(space="cosine", dim=config.output_emb_size if config.output_emb_size > 0 else 768)
    index.load_index("./index/model_index.bin", max_elements=config.hnsw_max_elements)
    # print(index.get_current_count())
    # load label
    label2id = label2ids(f"{config.data_path}/label.csv")
    id2corpus = {v: k for k, v in label2id.items()}
    t1 = time.time()
    token_list = convert_token({'1': text}, tokenizer)

    input_ids = paddle.to_tensor([token_list[0]])
    token_type_ids = paddle.to_tensor([token_list[1]])
    # print(input_ids)
    query_embedding = model.get_pooled_embedding(input_ids, token_type_ids)

    idx, distances = index.knn_query(query_embedding.numpy(), 10)
    for i, ind in enumerate(idx[0]):
        label = id2corpus[ind]
        distance = distances[0][i]
        print(label, distance)
    print(time.time() - t1)


if __name__ == '__main__':
    """"""
    # train()
    tokenizer = AutoTokenizer.from_pretrained('rocketqa-zh-dureader-query-encoder', cache_dir=config.model_path)
    # corpus_index(tokenizer)
    text = '快要到期的“我的资料”怎么续日期？'
    predict(text, tokenizer)



