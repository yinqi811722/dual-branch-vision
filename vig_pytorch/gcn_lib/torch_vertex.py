import numpy as np
import torch
from torch import nn
from .torch_nn import BasicConv, batched_index_select, act_layer
from .torch_edge import DenseDilatedKnnGraph
from .pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F
from timm.models.layers import DropPath


class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.in_channels = in_channels
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None, edge_weight=None):
        # if edge_index is not None:
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        # ===== LRGC modification starts here =====
        # relative message
        diff = x_j - x_i  # (B, C, N, K)

        # ⭐ 核心改动：edge_weight 在 max 之前
        if edge_weight is not None:
            diff = diff * edge_weight.unsqueeze(1)  # (B, C, N, K)

        # max-relative aggregation
        x_j, _ = torch.max(diff, -1, keepdim=True)
        # ===== LRGC modification ends here =====
        # else:
        #     x_j=y
        B, C, N, K = x_j.shape
        x_j = x_j.reshape(B * C // self.in_channels, self.in_channels, N, K)
        x = x.reshape(B * C // self.in_channels, self.in_channels, N, K)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        return self.nn(x)


class VIHGNNConv(nn.Module):

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(VIHGNNConv, self).__init__()
        self.linear = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.act = act_layer(act)
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = nn.GroupNorm(1, out_channels)
        else:
            self.norm = None

    def forward(self, x, edge_index, y=None):
        node2edge_idx, edge2node_idx = edge_index  # (B, E, k1), (B, N, k2)

        # Step 1: Dv^{-1/2}
        deg_v = edge2node_idx.shape[-1]  # k2
        x = x / (deg_v ** 0.5)

        # Step 2: X -> H^T X
        if y is not None:
            x_e = batched_index_select(y, node2edge_idx)
        else:
            x_e = batched_index_select(x, node2edge_idx)  # (B, C, E, k1)
        deg_e = x_e.shape[-1]  # k1
        x_e = x_e.sum(dim=-1, keepdim=True) / deg_e  # H^T X * D_e^{-1} (B, C, E, 1)

        # Step 3: H X_e ->
        x_n = batched_index_select(x_e, edge2node_idx)  # (B, C, N, k2)
        x_n = x_n.sum(dim=-1, keepdim=True)  # (B, C, N, 1)

        # Step 4: Dv^{-1/2}
        x_n = x_n / (deg_v ** 0.5)

        out = self.linear(x_n)
        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out


class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])

        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class GraphSAGE(nn.Module):
    """
    GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216) for dense data type
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        return self.nn2(torch.cat([x, x_j], dim=1))


class GINConv2d(nn.Module):
    """
    GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for dense data type
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GINConv2d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)
        return self.nn((1 + self.eps) * x + x_j)


class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """

    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'mr':
            self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'sage':
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == 'gin':
            self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'vihgnn':
            self.gconv = VIHGNNConv(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index, y=None, edge_weight=None):  # ⭐ 新增参数
        return self.gconv(x, edge_index, y, edge_weight)


class DyGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """

    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1, net_type="knn", num_heads=4):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation,
                                                      stochastic, epsilon, net_type=net_type, dim=in_channels,
                                                      num_heads=num_heads)

    def forward(self, x, relative_pos=None):
        B, C, H, W = x.shape
        y = None

        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()
        x = x.reshape(B, C, -1, 1).contiguous()

        edge_index,edge_weight = self.dilated_knn_graph(x, y, relative_pos)  # [2,b,node_num,k]

        x = super(DyGraphConv2d, self).forward(x, edge_index, y, edge_weight)

        return x.reshape(B, -1, H, W).contiguous(), edge_index


class DyGraphConv2dMultiGroup(GraphConv2d):
    """
    Dynamic graph convolution layer
    """

    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1, net_type="knn",num_groups=2,num_heads=4):
        super(DyGraphConv2dMultiGroup, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.num_groups = num_groups
        self.num_heads = num_heads
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation,
                                                      stochastic, epsilon, net_type=net_type, dim=in_channels//num_groups,
                                                      num_heads=num_heads)
        # self.conv=nn.ModuleList([GraphConv2d(in_channels//num_head, out_channels//num_head, conv, act, norm, bias)
        #                          for _ in range(num_head)])
        # self.nn=BasicConv([out_channels, out_channels], act, norm, bias)

    def forward(self, x, relative_pos=None):
        B, C, H, W = x.shape
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()
        x = x.reshape(B, C, -1, 1).contiguous()

        dim_per_head = C // self.num_groups
        if y is not None:
            y = y.reshape(B * self.num_groups, dim_per_head, -1, 1).contiguous()
        x = x.reshape(B * self.num_groups, dim_per_head, -1, 1).contiguous()
        edge_index = self.dilated_knn_graph(x, y, relative_pos)
        x = super(DyGraphConv2dMultiGroup, self).forward(x, edge_index, y)
        return x.reshape(B, -1, H, W).contiguous(), edge_index


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """

    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True, stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False,
                 use_multi_group=False, num_groups=2,
                 net_type="knn", num_heads=4):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        if use_multi_group:
            self.graph_conv = DyGraphConv2dMultiGroup(in_channels, in_channels * 2, kernel_size, dilation, conv,
                                                      act, norm, bias, stochastic, epsilon, r, net_type=net_type,
                                                      num_groups=num_groups,num_heads=num_heads)
        else:
            self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation, conv,
                                            act, norm, bias, stochastic, epsilon, r, net_type=net_type,num_heads=num_heads)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None
        if relative_pos:
            print('using relative_pos')
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                                                                                        int(n ** 0.5)))).unsqueeze(
                0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                relative_pos_tensor, size=(n, n // (r * r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x):
        _tmp = x

        x = self.fc1(x)

        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)

        x, edge_idx = self.graph_conv(x, relative_pos)

        x = self.fc2(x)

        x = self.drop_path(x) + _tmp

        return x, edge_idx
