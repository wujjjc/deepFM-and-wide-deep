import os
import pandas as pd
import numpy as np
"""
data数据构成
总共有 39 个特征（1 个标签 + 13 个数值特征 + 26 个类别特征）。

类型	数量	列号	特点
数值特征	13	I1 – I13	整数，多数是计数类特征（如曝光次数、历史点击等），可能包含缺失值（留空）。
类别特征	26	C1 – C26	已被 哈希到 32 位（0 ~ 2³²-1 之间），语义未知；存在缺失值。
"""

def encode(raw_x, val_x, min_freq=5):
    """
    特征编码
    input: raw_x, val_x  未编码之前
    return code_x, length 编码之后
    """
    voc = {-1: 0, '<UNK>': 0}  # -1表示缺失值，编码为0
    idx = 0
    freq = raw_x.value_counts()
    for x in freq.index:
        if x not in voc and freq[x] >= min_freq:
            idx += 1
            voc[x] = idx
    code_x = raw_x.map(voc).fillna(0).astype('int32')  # 将缺失值编码为0
    val_code_x = val_x.map(voc).fillna(0).astype('int32')  # 验证集同样处理
    return code_x, val_code_x, idx + 1  # vocab_size = 最大索引 + 1




def read_data(path):
    """
    读取数据
    Args:
        path (_type_): train.txt路径
    Returns:
        train_cat_list: 训练集类别特征列表，每个元素是一个类别特征的编码结果
        train_label_list: 训练集标签列表
        cat_vocab_sizes: 每个类别特征的词表大小列表
        val_cat_list: 验证集类别特征列表，每个元素是一个类别特征的编码结果
        val_label_list: 验证集标签列表
        train_num_list: 训练集数值特征列表，每个元素是一个数值特征的数组
        val_num_list: 验证集数值特征列表，每个元素是一个数值特征的数组
        train_x_val: 训练集分类特征列表，每个元素是一个分类特征的数组（用于FM的输入）
        val_x_val: 验证集分类特征列表，每个元素是一个分类特征的数组（用于FM的输入）
    """
    print("正在读取数据...")
    # 列名定义
    num_cols = [f'I{i}' for i in range(1, 14)]          # I1 ~ I13
    cat_cols = [f'C{i}' for i in range(1, 27)]          # C1 ~ C26
    all_cols = ['label'] + num_cols + cat_cols

    # 读取训练集
    dtype_map = {col: 'float32' for col in ['label'] + num_cols}
    dtype_map.update({col: 'string' for col in cat_cols})
    df = pd.read_csv(
        path,
        sep='\t',
        names=all_cols,
        na_values=[''],           # 将空字符串视为缺失值
        keep_default_na=False,
        dtype=dtype_map
    )
    # 填充数值特征的缺失值（用0填充）
    df[num_cols] = df[num_cols].fillna(0)
    df[cat_cols] = df[cat_cols].fillna(-1)
    df[num_cols] =  np.log1p(df[num_cols].clip(lower=0)) # 数值特征取对数，clip防止负数
    split = int(len(df) * 0.9)  # 90%训练，10%验证
    train_df = df.iloc[:split]
    val_df = df.iloc[split:]

    # 数值特征标准化（基于训练集统计量）
    for col in num_cols:
        train_mean = train_df[col].mean()
        train_std = train_df[col].std() + 1e-8
        train_df[col] = (train_df[col] - train_mean) / train_std
        val_df[col] = (val_df[col] - train_mean) / train_std

    # 处理类别特征
    cat_vocab_sizes = []
    train_cat_list = []
    val_cat_list = []
    train_num_list = []
    val_num_list = []
    train_x_val = [(train_df[cat_cols] != -1).astype('float32').values]  # 类别特征用one-hot编码后输入（有值就是1，没值就是0）
    val_x_val = [(val_df[cat_cols] != -1).astype('float32').values]
    train_label_list = train_df['label'].values.astype('float32')
    val_label_list = val_df['label'].values.astype('float32')
    for col in cat_cols:
        # 基于训练集构建词表并编码
        train_enc, val_enc, idx = encode(train_df[col], val_df[col])
        train_cat_list.append(train_enc)
        val_cat_list.append(val_enc)
        cat_vocab_sizes.append(idx)   # 词表大小
    for col in num_cols:
        train_num_list.append(train_df[col].values.astype('float32'))
        val_num_list.append(val_df[col].values.astype('float32'))
        train_x_val.append(train_df[col].values.astype('float32'))
        val_x_val.append(val_df[col].values.astype('float32'))
    print("数据读取完成！")
    train_cat_list = np.column_stack(train_cat_list)  # 将类别特征列表转换为二维数组 (n, 26)
    val_cat_list = np.column_stack(val_cat_list)
    train_num_list = np.column_stack(train_num_list)  # 将数值特征列表转换为二维数组 (n, 13)
    val_num_list = np.column_stack(val_num_list)
    train_x_val = np.column_stack(train_x_val)  # 将特征列表转换为二维数组 (n, 26)
    val_x_val = np.column_stack(val_x_val) # 将特征列表转换为二维数组 (n, 26)
    return train_cat_list, train_label_list, cat_vocab_sizes, val_cat_list, val_label_list, train_num_list, val_num_list, train_x_val, val_x_val
    
import torch
from torch.utils.data import Dataset, DataLoader
def collect_fn(batch):
    """
    自定义collate_fn，DataLoader 传入的 batch 是 list of dict
    每个 dict 包含 'cat', 'num', 'label', 'x_val'
    """
    cat_list = [torch.tensor(item['cat'], dtype=torch.long) for item in batch]
    num_list = [torch.tensor(item['num'], dtype=torch.float32) for item in batch]
    label_list = [torch.tensor(item['label'], dtype=torch.float32) for item in batch]
    x_val_list = [torch.tensor(item['x_val'], dtype=torch.float32) for item in batch]

    return {
        'cat': torch.stack(cat_list),       # (batch_size, 26)
        'num': torch.stack(num_list),       # (batch_size, 13)
        'label': torch.stack(label_list),   # (batch_size,)
        'x_val': torch.stack(x_val_list),   # (batch_size, 39)
    }

class deepFMDataset(Dataset):
    def __init__(self, cat_data, num_data, label_data, x_val_data):
        self.cat = cat_data   # 形状 (n_samples, n_cat_features)
        self.num = num_data   # 形状 (n_samples, n_num_features)
        self.label = label_data  # (n_samples,)
        self.x_val = x_val_data  # (n_samples, n_fields)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
            'cat': self.cat[idx],
            'num': self.num[idx],
            'label': self.label[idx],
            'x_val': self.x_val[idx]
        }