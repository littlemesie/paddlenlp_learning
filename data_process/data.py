# -*- coding:utf-8 -*-

"""
@date: 2024/1/24 下午5:18
@summary:
"""
import pandas as pd

data_path = '/media/mesie/F0E66F06E66ECC82/数据/paddle/baike_qa_multilabel'

def text_to_csv():
    data_dict = {'text': [],'label': []}
    with open(f"{data_path}/dev.txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.replace('\n', '').split('\t')
            data_dict['text'].append(line_list[0])
            data_dict['label'].append(line_list[1])
    data_df = pd.DataFrame(data_dict).reset_index(drop=True)
    data_df.to_csv(f"{data_path}/dev.csv",  index=False)
if __name__ == '__main__':
    """"""
    # text_to_csv()