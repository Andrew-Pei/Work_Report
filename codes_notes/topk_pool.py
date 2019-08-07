import torch
from torch.nn import Parameter
from torch_scatter import scatter_add, scatter_max
from torch_geometric.utils import softmax

from ..inits import uniform
from ...utils.num_nodes import maybe_num_nodes


def topk(x, ratio, batch, min_score=None, tol=1e-7):
# batch is a column vector which maps each node to its respective graph in the batch:
# batch=[0 ⋯ 0 1 ⋯ n−2 n−1 ⋯ n−1]⊤
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter_max(x, batch)[0][batch] - tol
		# result: tensor([2, 2, 2, 4, 4, 4, 4], dtype=torch.int32)
        scores_min = scores_max.clamp(max=min_score)
		
		# pytorch.clamp函数：将输入input张量每个元素的夹紧到区间 [min,max]

        perm = torch.nonzero(x > scores_min).view(-1)
		# pytorch.nonzero函数：返回一个包含输入 input 中非零元素索引的张量
		# torch.view函数旨在reshape张量形状。
    else:
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
		
		#返回一个tensor，其中每个数字代表每个图的节点数
        # new_zeros: Returns a Tensor of size size filled with 0. By default, 
		# the returned Tensor has the same torch.dtype and torch.device as this tensor.
		batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()
		# Use torch.Tensor.item() to get a Python number from a tensor containing
		# a single value
        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1),
             num_nodes.cumsum(dim=0)[:-1]], dim=0)
		# cumsum: y_i = x_1 + x_2 + x_3 + \dots + x_i
		# cat: 
		# result: [[0],[3]]
        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

        dense_x = x.new_full((batch_size * max_num_nodes, ), -2)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)

        _, perm = dense_x.sort(dim=-1, descending=True)

        perm = perm + cum_num_nodes.view(-1, 1)
        perm = perm.view(-1)
		# 每个图中节点值按照大小进行排序之后的顺序
        k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
		# k为经过一次池化后剩余的num_nodes张量。
        mask = [
            torch.arange(k[i], dtype=torch.long, device=x.device) +
            i * max_num_nodes for i in range(batch_size)
        ]
        mask = torch.cat(mask, dim=0)

        perm = perm[mask]

    return perm
	# 这个函数的入参中scores已经被计算好了（就在x张量里面），只需要排序即可，

def filter_adj(edge_index, edge_attr, perm, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
	# maybe_num_nodes func: return index.max().item() + 1 if num_nodes is None else num_nodes
    mask = perm.new_full((num_nodes, ), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = edge_index
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return torch.stack([row, col], dim=0), edge_attr


class TopKPooling(torch.nn.Module):
    r""":math:`\mathrm{top}_k` pooling operator from the `"Graph U-Net"
    <https://openreview.net/forum?id=HJePRoAct7>`_, `"Towards Sparse
    Hierarchical Graph Classifiers" <https://arxiv.org/abs/1811.01287>`_
    and `"Understanding Attention and Generalization in Graph Neural
    Networks" <https://arxiv.org/abs/1905.02850>`_ papers

    if min_score :math:`\tilde{\alpha}` is None:

        .. math::
            \mathbf{y} &= \frac{\mathbf{X}\mathbf{p}}{\| \mathbf{p} \|}

            \mathbf{i} &= \mathrm{top}_k(\mathbf{y})

            \mathbf{X}^{\prime} &= (\mathbf{X} \odot
            \mathrm{tanh}(\mathbf{y}))_{\mathbf{i}}

            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}}

    if min_score :math:`\tilde{\alpha}` is a value in [0, 1]:

        .. math::
            \mathbf{y} &= \mathrm{softmax}(\mathbf{X}\mathbf{p})

            \mathbf{i} &= \mathbf{y}_i > \tilde{\alpha}

            \mathbf{X}^{\prime} &= (\mathbf{X} \odot \mathbf{y})_{\mathbf{i}}

            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}},

    where nodes are dropped based on a learnable projection score
    :math:`\mathbf{p}`.

    Args:
        in_channels (int): Size of each input sample.
        ratio (float): Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`.
            This value is ignored if min_score is not None.
            (default: :obj:`0.5`)
        min_score (float, optional): Minimal node score :math:`\tilde{\alpha}`
            which is used to compute indices of pooled nodes
            :math:`\mathbf{i} = \mathbf{y}_i > \tilde{\alpha}`.
            When this value is not :obj:`None`, the :obj:`ratio` argument is
            ignored. (default: :obj:`None`)
        multiplier (float, optional): Coefficient by which features gets
            multiplied after pooling. This can be useful for large graphs and
            when :obj:`min_score` is used. (default: :obj:`1`)
        nonlinearity (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.tanh`)
    """

    def __init__(self, in_channels, ratio=0.5, min_score=None, multiplier=1,
                 nonlinearity=torch.tanh):
        super(TopKPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.weight = Parameter(torch.Tensor(1, in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform(size, self.weight)

    def forward(self, x, edge_index, edge_attr=None, batch=None, attn=None):
        """"""

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
		# 如果没有batch张量，就证明只有一个图，那么就是这个图的节点数个0就是batch张量
        attn = x if attn is None else attn
		# attn一般是attention的缩写
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
		# squeeze函数是把张量中长度为1的维度降维打击。unsqueeze函数是插入一个长度为1的维度
        score = (attn * self.weight).sum(dim=-1)
		# score得出四个节点的分数的计算结果，是个1*4的张量。

        if self.min_score is None:
            score = self.nonlinearity(score / self.weight.norm(p=2, dim=-1))
        else:
            score = softmax(score, batch)

        perm = topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                           num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, score[perm]

    def __repr__(self):
        return '{}({}, {}={}, multiplier={})'.format(
            self.__class__.__name__, self.in_channels,
            'ratio' if self.min_score is None else 'min_score',
            self.ratio if self.min_score is None else self.min_score,
            self.multiplier)
