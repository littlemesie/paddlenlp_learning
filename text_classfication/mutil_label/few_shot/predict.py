# -*- coding:utf-8 -*-

"""
@date: 2024/2/6 上午10:04
@summary: 模型预测
"""
import os
import time
import paddle
import paddle.nn.functional as F
from paddle import inference
from paddlenlp.data import Pad
from paddlenlp.transformers import AutoTokenizer
from core.model_config import get_model_config
from utils.dataloader import convert_token, label2ids

config = get_model_config('few_shot')


class Predictor(object):
    def __init__(self):
        """"""
        model_file = config.output_path + "/inference.pdmodel"
        params_file = config.output_path + "/inference.pdiparams"
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

    def predict(self, text, tokenizer, id2label):
        """"""
        token_list = convert_token({'1': text}, tokenizer)
        input_ids = paddle.to_tensor([token_list[0]])
        token_type_ids = paddle.to_tensor([token_list[1]])
        input_ids = Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64")(input_ids)
        token_type_ids = Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64")(token_type_ids)

        self.input_handles[0].copy_from_cpu(input_ids)
        self.input_handles[1].copy_from_cpu(token_type_ids)
        self.predictor.run()
        logits = self.output_handle.copy_to_cpu()
        probs = F.sigmoid(paddle.to_tensor(logits)).numpy()
        labels = []
        for prob in probs:
            for i, p in enumerate(prob):
                if p > 0.5:
                    label = id2label[i]
                    labels.append((label, round(p, 4)))
        print(labels)




if __name__ == "__main__":
    """"""
    predictor = Predictor()
    tokenizer = AutoTokenizer.from_pretrained(config.save_dir)
    label2id = label2ids(f"{config.data_path}/label.csv")
    id2label = {v: k for k, v in label2id.items()}
    t1 = time.time()
    text = "快要到期的“我的资料”怎么续日期？"
    predictor.predict(text, tokenizer, id2label)
    print(time.time() - t1)

