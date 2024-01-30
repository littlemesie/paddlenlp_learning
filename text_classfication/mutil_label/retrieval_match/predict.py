# -*- coding:utf-8 -*-

"""
@date: 2024/1/29 下午3:48
@summary: 预测
"""
import os
import time
import paddle
import hnswlib
from paddle import inference
from paddlenlp.data import Pad, Tuple
from paddlenlp.transformers import AutoTokenizer
from core.model_config import get_model_config
from utils.dataloader import convert_token, label2ids

config = get_model_config('retrieval_match')


class Predictor(object):
    def __init__(self):
        """"""
        model_file = config.output_path + "/output/inference.get_pooled_embedding.pdmodel"
        params_file = config.output_path + "/output/inference.get_pooled_embedding.pdiparams"
        if not os.path.exists(model_file):
            raise ValueError("not find model file path {}".format(model_file))
        if not os.path.exists(params_file):
            raise ValueError("not find params file path {}".format(params_file))
        model_config = paddle.inference.Config(model_file, params_file)

        if config.device == "gpu":
            # set GPU configs accordingly
            # such as initialize the gpu memory, enable tensorrt
            model_config.enable_use_gpu(100, 0)
            precision_map = {
                "fp16": inference.PrecisionType.Half,
                "fp32": inference.PrecisionType.Float32,
                "int8": inference.PrecisionType.Int8,
            }
            precision_mode = precision_map[config.precision]

            if config.use_tensorrt:
                model_config.enable_tensorrt_engine(
                    max_batch_size=config.batch_size, min_subgraph_size=30, precision_mode=precision_mode
                )
        elif config.device == "cpu":
            # set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            model_config.disable_gpu()
            if config.enable_mkldnn:
                # cache 10 different shapes for mkldnn to avoid memory leak
                model_config.set_mkldnn_cache_capacity(10)
                model_config.enable_mkldnn()
            model_config.set_cpu_math_library_num_threads(config.cpu_threads)
        elif config.device == "xpu":
            # set XPU configs accordingly
            model_config.enable_xpu(100)

        model_config.switch_use_feed_fetch_ops(False)
        self.predictor = paddle.inference.create_predictor(model_config)

        self.input_handles = [self.predictor.get_input_handle(name) for name in self.predictor.get_input_names()]
        self.output_handle = self.predictor.get_output_handle(self.predictor.get_output_names()[0])
        # preload
        self.preload()

    def preload(self):
        """preload"""
        # load index
        self.index = hnswlib.Index(space="cosine", dim=config.output_emb_size if config.output_emb_size > 0 else 768)
        self.index.load_index("./index/model_index.bin", max_elements=config.hnsw_max_elements)
        # load label
        label2id = label2ids(f"{config.data_path}/label.csv")
        self.id2corpus = {v: k for k, v in label2id.items()}


    def extract_embedding(self, data, tokenizer):
        """"""
        examples = []
        for idx, text in data.items():
            input_ids, segment_ids = convert_token(text, tokenizer)
            examples.append((input_ids, segment_ids))

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # input
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # segment
        ): fn(samples)

        input_ids, segment_ids = batchify_fn(examples)
        print(input_ids)
        self.input_handles[0].copy_from_cpu(input_ids)
        self.input_handles[1].copy_from_cpu(segment_ids)
        self.predictor.run()
        embed = self.output_handle.copy_to_cpu()
        return embed

    def predict(self, text, tokenizer):
        """"""
        token_list = convert_token({'1': text}, tokenizer)
        input_ids = paddle.to_tensor([token_list[0]])
        token_type_ids = paddle.to_tensor([token_list[1]])
        input_ids = Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64")(input_ids)
        token_type_ids = Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64")(token_type_ids)
        # batchify_fn = lambda samples, fn=Tuple(
        #     Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # input
        #     Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # segment
        # ): fn(samples)
        # input_ids, token_type_ids = batchify_fn((input_ids, token_type_ids))
        # print(input_ids)

        self.input_handles[0].copy_from_cpu(input_ids)
        self.input_handles[1].copy_from_cpu(token_type_ids)
        self.predictor.run()
        embed = self.output_handle.copy_to_cpu()
        idx, distances = self.index.knn_query(embed, 10)
        for i, ind in enumerate(idx[0]):
            label = self.id2corpus[ind]
            distance = distances[0][i]
            print(label, distance)



if __name__ == "__main__":
    """"""
    predictor = Predictor()
    tokenizer = AutoTokenizer.from_pretrained("rocketqa-zh-dureader-query-encoder", cache_dir=config.model_path)
    # id2corpus = {0: {"sentence": "快要到期的“我的资料”怎么续日期？"}}
    # res = predictor.extract_embedding(id2corpus, tokenizer)
    # print(res.shape)
    # print(res)
    t1 = time.time()
    text = "快要到期的“我的资料”怎么续日期？"
    predictor.predict(text, tokenizer)
    print(time.time() - t1)

