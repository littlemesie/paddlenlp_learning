# -*- coding:utf-8 -*-

"""
@date: 2024/2/6 上午9:25
@summary:
"""
import os
import paddle
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from core.model_config import get_model_config

config = get_model_config('few_shot')


def convert_model():
    """model convert"""
    model = AutoModelForSequenceClassification.from_pretrained(config.save_dir)
    model.eval()
    input_spec = [
        paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
        paddle.static.InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
    ]
    # Convert to static graph with specific input description
    model = paddle.jit.to_static(model, input_spec=input_spec)

    # Save in static graph model.
    save_path = os.path.join(config.output_path, "inference")
    paddle.jit.save(model, save_path)

if __name__ == '__main__':
    """"""
    convert_model()