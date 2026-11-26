
import math
import torch
from torch import nn
import torch.nn.functional as F


def pairwise_distance(x):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        x_inner = -2*torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)


def part_pairwise_distance(x, start_idx=0, end_idx=1):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        x_part = x[:, start_idx:end_idx]
        x_square_part = torch.sum(torch.mul(x_part, x_part), dim=-1, keepdim=True)
        x_inner = -2*torch.matmul(x_part, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square_part + x_inner + x_square.transpose(2, 1)


def xy_pairwise_distance(x, y):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        xy_inner = -2*torch.matmul(x, y.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        y_square = torch.sum(torch.mul(y, y), dim=-1, keepdim=True)
        return x_square + xy_inner + y_square.transpose(2, 1)


class MultiHeadDSAQKGraph(nn.Module):
    """
    Multi-head learnable QK attention graph builder with DSA style
    (Lightning Indexer implementation for routing)
    """

    def __init__(self, dim, num_heads=4):
        super(MultiHeadDSAQKGraph, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim

        # ✅ 修改1: Learnable Q, K, and W projections (DSA style)
        self.q = nn.Linear(dim, dim, bias=False)  # 保持不变
        self.k = nn.Linear(dim, dim, bias=False)  # 保持不变
        self.w = nn.Linear(dim, num_heads, bias=False)  # ⭐ 新增：权重投影

    def forward(self, x, k_top, relative_pos=None):
        """
        Args:
            x: (B, C, N, 1) input features
            k_top: int, number of neighbors
            relative_pos: (B, N, N) optional relative position bias
        Returns:
            edge_index: (2, B, N, k_top)
        """
        B, C, N, _ = x.shape
        device = x.device

        # Reshape to (B, N, C)
        x = x.transpose(2, 1).squeeze(-1)

        # Project to Q, K, and W
        q = self.q(x)  # (B, N, C)
        k = self.k(x)  # (B, N, C)
        w = self.w(x)  # ⭐ 修改2: 新增权重投影 (B, N, num_heads)

        # Split into multi-heads: (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # ✅ 修改3: Scaled dot-product attention per head
        attn = torch.matmul(q, k.transpose(-2, -1))  # (B, num_heads, N, N)
        attn = attn / (self.head_dim ** 0.5)  # 可选：保持缩放或移除

        # Add relative position bias if provided
        if relative_pos is not None:
            attn = attn + relative_pos.unsqueeze(1)  # (B, num_heads, N, N)

        # ⭐ 修改4: 使用ReLU激活代替softmax (DSA核心)
        attn = torch.relu(attn)  # (B, num_heads, N, N)

        # ⭐ 修改5: 加权求和多头 (DSA style)
        # w: (B, N, num_heads) -> (B, num_heads, N, 1)
        w = w.transpose(1, 2).unsqueeze(-1)  # (B, num_heads, N, 1)

        # 加权: (B, num_heads, N, N) * (B, num_heads, N, 1)
        weighted_attn = attn * w  # 广播: (B, num_heads, N, N)

        # 对头维度求和
        attn = weighted_attn.sum(dim=1)  # (B, N, N)

        # Top-k sparse routing
        _, nn_idx = torch.topk(attn, k=k_top, dim=-1)  # (B, N, k_top)

        # Center index
        center_idx = torch.arange(N, device=device) \
            .view(1, N, 1) \
            .expand(B, N, k_top)

        return torch.stack((nn_idx, center_idx), dim=0)  # (2, B, N, k_top)


class MultiHeadLRGCGraph(nn.Module):
    def __init__(self, dim, num_heads=4, tau_init=-1.0):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.tau = nn.Parameter(torch.tensor(tau_init))

        self.edge_attr = None

    def forward(self, x, k_top, relative_pos=None):
        """
        Returns:
            edge_index: (2, B, N, k_top) - 但现在是全连接的前 k_top 个节点

        Side effect:
            self.edge_attr: (B, N, k_top) - 对应的注意力权重
        """
        B, C, N, _ = x.shape
        device = x.device

        x = x.transpose(2, 1).squeeze(-1)

        q = self.q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if relative_pos is not None:
            attn = attn + relative_pos.unsqueeze(1)

            # LRGC 核心
        gate = torch.sigmoid(attn) - torch.sigmoid(self.tau)
        gate = torch.relu(gate)
        attn = torch.tanh(gate)

        attn = attn.mean(dim=1)  # (B, N, N)

        # ================= 移除 top-k,直接取前 k_top 个节点 =================
        # 邻居索引: 每个节点连接到 [0, 1, 2, ..., k_top-1]
        nn_idx = torch.arange(k_top, device=device) \
            .view(1, 1, k_top) \
            .expand(B, N, k_top)  # (B, N, k_top)

        # 提取对应的注意力权重 (完全可微!)
        edge_weight = torch.gather(attn, -1, nn_idx)  # (B, N, k_top)

        self.edge_attr = edge_weight
        # ====================================================================

        center_idx = torch.arange(N, device=device) \
            .view(1, N, 1) \
            .expand(B, N, k_top)

        return torch.stack((nn_idx, center_idx), dim=0)


def small_world_knn_matrix(x, k=16, num_shortcuts=4, num_iterations=0, relative_pos=None):
    """
    Small World KNN图构建 - 显存优化版

    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int, 总邻居数
        num_shortcuts: int, 长程快捷连接数量
        num_iterations: int, 优化迭代次数 (设为0可大幅节省显存)
        relative_pos: optional (batch_size, num_points, num_points)
    Returns:
        (2, batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)  # (B, N, D)
        batch_size, n_points, n_dims = x.shape
        device = x.device

        k_local = max(1, k - num_shortcuts)
        num_shortcuts = k - k_local

        # ============ Step 1: 局部KNN（分块计算） ============
        n_part = 2048  # 减小分块大小节省显存
        if n_points > n_part:
            nn_idx_list = []
            groups = math.ceil(n_points / n_part)
            for i in range(groups):
                start_idx = n_part * i
                end_idx = min(n_points, n_part * (i + 1))
                dist = part_pairwise_distance(x.detach(), start_idx, end_idx)
                if relative_pos is not None:
                    dist = dist + relative_pos[:, start_idx:end_idx]
                _, nn_idx_part = torch.topk(-dist, k=k_local)
                nn_idx_list.append(nn_idx_part)
                del dist
            nn_idx = torch.cat(nn_idx_list, dim=1)
        else:
            dist = pairwise_distance(x.detach())
            if relative_pos is not None:
                dist = dist + relative_pos
            _, nn_idx = torch.topk(-dist, k=k_local)
            del dist

        # ============ Step 2: 长程快捷连接 ============
        if num_shortcuts > 0:
            base = torch.arange(n_points, device=device)
            shortcut_list = []
            for i in range(num_shortcuts):
                scale = 2 ** (i + 1)
                noise = torch.randint(0, max(1, scale // 2), (batch_size, n_points), device=device)
                jump = (base.unsqueeze(0) + scale + noise) % n_points
                shortcut_list.append(jump)
            shortcuts = torch.stack(shortcut_list, dim=-1)  # (B, N, num_shortcuts)
            nn_idx = torch.cat([nn_idx, shortcuts], dim=-1)

        # ============ Step 3: 迭代优化（分块处理） ============
        for _ in range(num_iterations):
            current_k = nn_idx.shape[-1]
            sample_k = min(current_k, 2)  # 减少采样数

            new_nn_idx_list = []
            n_part_opt = 1024  # 优化时用更小的块
            groups = math.ceil(n_points / n_part_opt)

            for g in range(groups):
                s = n_part_opt * g
                e = min(n_points, n_part_opt * (g + 1))
                chunk_size = e - s

                # 当前块的查询点和邻居
                query = x[:, s:e, :]  # (B, chunk, D)
                nn_chunk = nn_idx[:, s:e, :]  # (B, chunk, current_k)

                # 获取邻居的邻居
                sampled = nn_chunk[:, :, :sample_k]  # (B, chunk, sample_k)
                batch_exp = torch.arange(batch_size, device=device).view(-1, 1, 1)
                batch_exp = batch_exp.expand(-1, chunk_size, sample_k)
                non = nn_idx[batch_exp, sampled, :]  # (B, chunk, sample_k, current_k)
                non_flat = non.reshape(batch_size, chunk_size, -1)

                # 合并候选并限制数量
                all_cand = torch.cat([nn_chunk, non_flat], dim=-1)
                max_cand = min(all_cand.shape[-1], k * 2)
                all_cand = all_cand[:, :, :max_cand]
                n_cand = all_cand.shape[-1]

                # 计算距离（再分小块）
                dists = torch.full((batch_size, chunk_size, n_cand), float('inf'), device=device)
                cand_part = 256

                for cb in range(0, n_cand, cand_part):
                    ce = min(cb + cand_part, n_cand)
                    cand_idx = all_cand[:, :, cb:ce]

                    # 获取候选坐标
                    b_exp = torch.arange(batch_size, device=device).view(-1, 1, 1).expand(-1, chunk_size, ce - cb)
                    cand_coords = x[b_exp.reshape(-1), cand_idx.reshape(-1)]
                    cand_coords = cand_coords.reshape(batch_size, chunk_size, ce - cb, n_dims)

                    # 距离
                    dists[:, :, cb:ce] = ((cand_coords - query.unsqueeze(2)) ** 2).sum(dim=-1)

                    if relative_pos is not None:
                        rel = torch.gather(relative_pos[:, s:e, :], dim=2, index=cand_idx)
                        dists[:, :, cb:ce] = dists[:, :, cb:ce] + rel

                    del cand_coords

                # 排除自身
                self_idx = torch.arange(s, e, device=device).view(1, -1, 1)
                dists.masked_fill_(all_cand == self_idx, float('inf'))

                # 选k个最近
                _, topk_idx = torch.topk(-dists, k=k, dim=-1)
                new_chunk = torch.gather(all_cand, dim=2, index=topk_idx)
                new_nn_idx_list.append(new_chunk)

                del dists, all_cand, non, non_flat

            nn_idx = torch.cat(new_nn_idx_list, dim=1)

        # ============ 返回 ============
        center_idx = torch.arange(n_points, device=device).view(1, -1, 1).expand(batch_size, -1, k)
        return torch.stack((nn_idx, center_idx), dim=0)

def  dense_knn_matrix(x, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        ### memory efficient implementation ###
        n_part = 10000
        if n_points > n_part:
            nn_idx_list = []
            groups = math.ceil(n_points / n_part)
            for i in range(groups):
                start_idx = n_part * i
                end_idx = min(n_points, n_part * (i + 1))
                dist = part_pairwise_distance(x.detach(), start_idx, end_idx)
                if relative_pos is not None:
                    dist += relative_pos[:, start_idx:end_idx]
                _, nn_idx_part = torch.topk(-dist, k=k)
                nn_idx_list += [nn_idx_part]
            nn_idx = torch.cat(nn_idx_list, dim=1)
        else:
            dist = pairwise_distance(x.detach())
            if relative_pos is not None:
                dist += relative_pos
            _, nn_idx = torch.topk(-dist, k=k) # b, n, k
        ######
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


import torch


def dense_knn_lrg_matrix_vectorized(x, k=16, tau=2.5, relative_pos=None):
    """
    向量化版本的 KNN 图 + LRG 超边集合表示

    Args:
        x: (B, C, N, 1) 输入点云特征
        k: int, KNN 邻居数
        tau: float, LRG 扩散尺度
        relative_pos: optional (B, N, N) 位置偏差矩阵

    Returns:
        edge_index: (2, B, N, 1) 张量
          第一维0: 超边集合ID
          第一维1: 顶点索引
    """
    with torch.no_grad():
        # 确保输入有效
        if x is None:
            raise ValueError("输入x不能为None")

        # 数据预处理 (B, N, C)
        x = x.transpose(2, 1).squeeze(-1)
        B, N, C = x.shape
        device = x.device

        # 1. 计算批处理的 KNN 图
        # 计算点云数据的点对欧氏距离
        x_norm = (x ** 2).sum(dim=-1, keepdim=True)
        dist = x_norm + x_norm.transpose(1, 2) - 2 * torch.bmm(x, x.transpose(1, 2))
        dist = torch.sqrt(torch.clamp(dist, min=1e-6))  # 确保距离为正

        # 添加相对位置偏差（如果提供）
        if relative_pos is not None:
            # 确保相对位置矩阵形状正确
            if relative_pos.dim() != 3 or relative_pos.shape != (B, N, N):
                raise ValueError(f"relative_pos 应有形状 (B, N, N) = ({B}, {N}, {N}), 但得到 {relative_pos.shape}")
            dist = dist + relative_pos

        # 获取 k 个最近邻
        _, nn_idx = torch.topk(dist, k=k, dim=-1, largest=False)

        # 构建对称邻接矩阵
        adj = torch.zeros(B, N, N, dtype=torch.bool, device=device)
        adj = adj.scatter(-1, nn_idx, True)
        adj = adj | adj.transpose(1, 2)  # 确保对称性

        # 2. 计算批处理的拉普拉斯矩阵
        degree = adj.sum(dim=-1).float()
        D = torch.diag_embed(degree)
        A = adj.float()
        L = D - A

        # 确保L不为None
        if L is None:
            raise RuntimeError("拉普拉斯矩阵L计算失败")

        # 3. 批处理计算扩散核和 zeta 矩阵
        # 计算矩阵指数 (扩散核)
        # 确保tau和L有效
        if tau is None:
            tau = 1.5  # 设置默认值

        # 添加类型检查
        if not isinstance(tau, (int, float)):
            raise TypeError(f"tau应为数值类型，但得到{type(tau)}")

        # 确保L是张量
        if not torch.is_tensor(L):
            raise TypeError(f"L应为张量，但得到{type(L)}")

        K_tau = torch.matrix_exp(-tau * L)  # (B, N, N)

        # 计算迹并归一化
        trace = torch.einsum('bii->b', K_tau).view(B, 1, 1)
        rho = K_tau / (trace + 1e-8)  # 添加小常数避免除零

        # 计算 rho_prime 和 zeta
        diag = torch.diagonal(rho, dim1=1, dim2=2)  # (B, N)
        min_mat = torch.min(
            diag.unsqueeze(2),  # (B, N, 1)
            diag.unsqueeze(1)  # (B, 1, N)
        ) + 1e-6
        rho_prime = rho / min_mat
        zeta = rho_prime >= 1.0  # (B, N, N)

        # 4. 为每个batch生成超边集合ID
        # 创建结果列表
        edge_list = []

        # 遍历每个batch
        for b in range(B):
            # 初始化并查集
            parent = torch.arange(N, device=device)

            # 定义并查集函数 - 使用nonlocal解决作用域问题
            def find(i, parent=parent):
                if parent[i] != i:
                    parent[i] = find(parent[i], parent)
                return parent[i]

            # 应用连接关系
            for i in range(N):
                for j in range(i + 1, N):
                    if zeta[b, i, j]:
                        root_i = find(i)
                        root_j = find(j)
                        if root_i != root_j:
                            # 按索引合并
                            if root_i < root_j:
                                parent[root_j] = root_i
                            else:
                                parent[root_i] = root_j

            # 压缩路径
            for i in range(N):
                parent[i] = find(i)

            # 创建映射：原始组件ID -> 连续ID
            unique_ids = torch.unique(parent)
            id_map = {orig_id.item(): new_id for new_id, orig_id in enumerate(unique_ids)}

            # 将组件ID映射到连续整数
            component_ids = torch.tensor([id_map[parent[i].item()] for i in range(N)],
                                         dtype=torch.long, device=device)

            # 构造超边列表 (2, N)
            edge_b = torch.stack([
                component_ids,  # 超边集合ID
                torch.arange(N, device=device)  # 顶点索引
            ], dim=0)

            # 添加到列表
            edge_list.append(edge_b)

        # 堆叠所有batch (2, B, N)
        edge_index = torch.stack(edge_list, dim=1)

        # 添加最后一个维度 (2, B, N, 1)
        return edge_index.unsqueeze(-1)

def xy_dense_knn_matrix(x, y, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)

        y = y.transpose(2, 1).squeeze(-1)

        batch_size, n_points, n_dims = x.shape
        dist = xy_pairwise_distance(x.detach(), y.detach())

        if relative_pos is not None:
            dist += relative_pos
        _, nn_idx = torch.topk(-dist, k=k)
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)

    return torch.stack((nn_idx, center_idx), dim=0)


class DenseDilated(nn.Module):
    """
    Find dilated neighbor from neighbor list

    edge_index: (2, batch_size, num_points, k)
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):

        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, :, randnum]
            else:
                edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, :, ::self.dilation]

        return edge_index


class DenseDilatedKnnGraph(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """

    def __init__(self, k=9, dilation=1, stochastic=False,
                 epsilon=0.0, net_type="knn", k_random=2, dim=None, num_heads=4):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self.k_random = k_random
        self.net_type = net_type
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)

        # DSA/LRGC attention graph
        if net_type in ["dsa", "lrgc"]:
            assert dim is not None, "dim must be provided when net_type='dsa' or 'lrgc'"
            self.dsa_qk = MultiHeadDSAQKGraph(dim, num_heads)

    def forward(self, x, y=None, relative_pos=None):
        edge_weight = None  # ⭐ 新增: 存储边权重

        if y is not None:
            x = F.normalize(x, p=2.0, dim=1)
            y = F.normalize(y, p=2.0, dim=1)
            edge_index = xy_dense_knn_matrix(x, y, self.k * self.dilation, relative_pos)
        else:
            x = F.normalize(x, p=2.0, dim=1)

            if self.net_type == "small_world":
                #x, k=16, num_shortcuts=4, num_iterations=2, relative_pos=None
                edge_index = small_world_knn_matrix(x, self.k * self.dilation, num_shortcuts=1,num_iterations=0, relative_pos=relative_pos)
            elif self.net_type in ["dsa", "lrgc"]:  # ⭐ 修改: 支持 LRGC
                edge_index = self.dsa_qk(x, self.k * self.dilation, relative_pos)
                #edge_weight = self.dsa_qk.edge_attr  # ⭐ 获取边权重 (完全可微!)
            elif self.net_type == "lrg":
                edge_index = dense_knn_lrg_matrix_vectorized(x, self.k * self.dilation, relative_pos)
                return edge_index, None  # ⭐ 返回 None 权重
            else:
                edge_index = dense_knn_matrix(x, self.k * self.dilation, relative_pos)

                # ⭐ 修改: 对 edge_index 做 dilation, 同时处理 edge_weight
        edge_index_dilated = self._dilated(edge_index)

        # ⭐ 如果有边权重, 也需要做相应的 dilation
        if edge_weight is not None:
            edge_weight_dilated = self._dilated_weight(edge_weight)
        else:
            edge_weight_dilated = None

        return edge_index_dilated, edge_weight_dilated

    def _dilated_weight(self, edge_weight):
        """
        对边权重做 dilation 采样 (保持可微)
        edge_weight: (B, N, k * dilation)
        返回: (B, N, k)
        """
        if self.dilation > 1:
            B, N, _ = edge_weight.shape
            # 按 dilation 步长采样
            idx = torch.arange(0, self.k * self.dilation, self.dilation, device=edge_weight.device)
            edge_weight = edge_weight[:, :, idx]  # (B, N, k) - 完全可微!
        return edge_weight
