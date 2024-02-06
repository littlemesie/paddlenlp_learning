# -*- coding:utf-8 -*-

"""
@date: 2024/1/30 下午3:17
@summary:
"""
import os
import paddle
from paddlenlp.transformers import AutoTokenizer, AutoModel
from core.model_config import get_model_config
from text_classfication.mutil_label.retrieval_match.base_model import SemanticIndexBaseStatic

config = get_model_config('retrieval_match')


def convert_model():
    """model convert"""
    pretrained_model = AutoModel.from_pretrained(config.model_name_or_path, cache_dir=config.model_path)
    model = SemanticIndexBaseStatic(pretrained_model, output_emb_size=config.output_emb_size)
    params_path = f"{config.model_save_dir}/model_best/model_state.pdparams"
    if params_path:
        state_dict = paddle.load(params_path)
        model.set_dict(state_dict)
    model.eval()
    # Convert to static graph with specific input description
    model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # input_ids
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # segment_ids
        ],
    )
    # Save in static graph model.
    save_path = os.path.join(config.output_path, "inference")
    paddle.jit.save(model, save_path)

if __name__ == '__main__':
    """"""
    convert_model()