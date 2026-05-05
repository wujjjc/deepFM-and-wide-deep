import torch
from torch import nn
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size: list, output_size,
                 activation=nn.ReLU(), dropout=0.1, use_bn=True):
        super(MLP, self).__init__()
        layer = []
        for x in hidden_size:
            layer.append(nn.Linear(input_size, x))
            if use_bn:
                layer.append(nn.BatchNorm1d(x))
            layer.append(activation)
            layer.append(nn.Dropout(dropout))
            input_size = x
        layer.append(nn.Linear(input_size, output_size))
        self.net = nn.Sequential(*layer)

    def forward(self, x):
        return self.net(x)

class FM(nn.Module):
    def __init__(self, kinds=39):
        """_summary_

        Args:
            kinds: 特征总数，包含数值特征和类别特征
        """
        super(FM, self).__init__()
        self.kinds = kinds
        self.linear = nn.Linear(kinds, 1)  # 一阶线性部分
    
    def forward(self, x_val, x_emb):
        """_summary_

        Args:
            x_val (_type_): batch * 39, 数值特征直接输入，类别特征用one-hot编码后输入（有值就是1，没值就是0）
            x_emb (_type_): batch * 39 * embedding_dim
        """
        linear_part = self.linear(x_val)  # batch * 1
        # 二阶交叉特征部分
        sum_emb = torch.sum(x_emb, dim=1) # batch * embedding_dim
        sum_emb = sum_emb * sum_emb  # batch * embedding_dim
        sum_ = torch.sum(x_emb * x_emb, dim=1)  # batch * embedding_dim
        interaction_part = torch.sum(0.5 * (sum_emb - sum_), dim=1, keepdim=True)  # batch * 1
        return linear_part + interaction_part  # batch * 1

class DeepFM(nn.Module):
    def __init__(self, num_kinds, cat_kinds:list, kinds=39, embedding_dim=16, hidden_size=[256, 128, 64, 32], dropout=0.1):
        """_summary_

        Args:
            num_kinds (int): 数值特征的种类数
            cat_kinds (list): 类别特征的种类数列表
            kinds (int, optional): 总种类数，包含数值特征和类别特征. Defaults to 39.
            embedding_dim (int, optional): 隐向量维度. Defaults to 16.
            hidden_size (list, optional): FFN隐藏层维度列表. Defaults to [256, 128, 64, 32].
            dropout (float, optional): dropout率 Defaults to 0.1.
        """
        super(DeepFM, self).__init__()
        self.fm = FM(kinds)
        self.num_embbed = nn.Embedding(num_kinds, embedding_dim)
        self.embbeding_dim = embedding_dim
        layers = []
        for cat_k in cat_kinds:
            layers.append(nn.Embedding(cat_k, embedding_dim, padding_idx=0))
        self.cat_embeddings = nn.ModuleList(layers)
        self.deep = MLP(kinds * embedding_dim, hidden_size, 1, nn.ReLU(), dropout=dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_val, x_num, x_cat, mode='train'):
        """

        Args:
            x_val (_type_): batch * 39, 数值特征直接输入，类别特征用one-hot编码后输入（有值就是1，没值就是0）
            x_num (_type_): batch * 13, 数值特征输入
            x_cat (_type_): batch * 26, 类别特征输入，已经编码成整数索引
        """
        x_emb = []
        for i, net in enumerate(self.cat_embeddings):
            emb = net(x_cat[:, i]).unsqueeze(1)  # (batch, 1, emb_dim)
            x_emb.append(emb)
        num_indices = torch.arange(x_num.shape[1], device=x_num.device)
        num_emb = self.num_embbed(num_indices).unsqueeze(0)  # (1, num_kinds, emb_dim)
        num_emb = num_emb * x_num.unsqueeze(-1)              # (batch, num_kinds, emb_dim)
        x_emb.append(num_emb)
        x_emb = torch.cat(x_emb, dim=1)  # (batch, 39, emb_dim)

        fm_out = self.fm(x_val, x_emb)  # (batch, 1)
        deep_out = self.deep(x_emb.flatten(start_dim=1))  # batch * 1
        out = fm_out + deep_out  # batch * 1
        if mode == 'train':
            return out  # batch * 1, 直接返回原始输出，后续会传入BCEWithLogitsLoss
        else:
            return self.sigmoid(out)  # (batch, 1)


class wide(nn.Module):
    def __init__(self, cross_size= int(26 * 25 / 2), num_size=13, embedding_dim=32):
        super(wide, self).__init__()
        num_cat = 26

        # 325 张 Embedding(4, emb_dim) 合并为 1 张 Embedding(1300, emb_dim)
        self.embbed = nn.Embedding(cross_size * 4, embedding_dim)

        # 预计算所有交叉 pair 的列索引和偏移
        pairs_i, pairs_j = [], []
        for i in range(num_cat):
            for j in range(i + 1, num_cat):
                pairs_i.append(i)
                pairs_j.append(j)
        self.register_buffer('pair_i', torch.tensor(pairs_i, dtype=torch.long))
        self.register_buffer('pair_j', torch.tensor(pairs_j, dtype=torch.long))
        self.register_buffer('offset', torch.arange(cross_size, dtype=torch.long) * 4)

        self.linear = nn.Linear(cross_size * embedding_dim + num_size, 1, bias=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x_cat, x_num):
        # 一次向量化: (batch, 325), 每个位置编码为 0-3 + pair偏移
        cross = (x_cat[:, self.pair_i] * 2 + x_cat[:, self.pair_j]).long() + self.offset
        emb = self.embbed(cross)  # (batch, 325, emb_dim)
        emb = emb.flatten(1)  # (batch, 325 * emb_dim)
        emb = torch.cat([emb, x_num], dim=1)
        emb = self.dropout(emb)
        return self.linear(emb)

class WideDeep(nn.Module):
    def __init__(self, num_kinds, cat_kinds:list, kinds=39, embedding_dim=32, hidden_size=[512, 256, 128, 64, 32], dropout=0.1):
        super(WideDeep, self).__init__()
        num_cat = len(cat_kinds)

        self.wide = wide(
            cross_size=num_cat * (num_cat - 1) // 2,
            num_size=num_kinds,
            embedding_dim=embedding_dim,
        )
        self.deep = MLP(kinds * embedding_dim, hidden_size, 1, nn.ReLU(), dropout=dropout)
        self.sigmoid = nn.Sigmoid()

        # 数值特征 embedding
        self.num_embbed = nn.Embedding(num_kinds, embedding_dim)

        # 类别特征 embedding: 合并为一张大表，一次向量化 lookup
        total_vocab = sum(cat_kinds)
        cumsum = torch.tensor([0] + list(cat_kinds[:-1]), dtype=torch.long).cumsum(0)
        self.register_buffer('cat_offset', cumsum)
        self.cat_emb = nn.Embedding(total_vocab, embedding_dim, padding_idx=0)

    def forward(self, x_val, x_num, x_cat, mode='train'):
        wide_out = self.wide(x_val[:, :x_cat.shape[1]], x_num)

        # Deep: 合并的 cat embedding 一次向量化 lookup
        # 值为 0 → 映射到全局 padding idx 0；>0 → 加上累计 vocab 偏移
        x_cat_offset = torch.where(
            x_cat == 0,
            torch.zeros_like(x_cat),
            x_cat + self.cat_offset,
        )
        emb = self.cat_emb(x_cat_offset)  # (batch, 26, emb_dim)
        num_indices = torch.arange(x_num.shape[1], device=x_num.device)
        num_emb = self.num_embbed(num_indices).unsqueeze(0)  # (1, num_kinds, emb_dim)
        num_emb = num_emb * x_num.unsqueeze(-1)  # (batch, num_kinds, emb_dim)
        x_emb = torch.cat([emb, num_emb], dim=1)  # (batch, 39, emb_dim)
        deep_out = self.deep(x_emb.flatten(start_dim=1, end_dim=-1))

        out = wide_out + deep_out
        if mode == 'train':
            return out
        else:
            return self.sigmoid(out)
        
        