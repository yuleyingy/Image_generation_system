"""预览生成算法：基线方法、论文方法（全局下采样矩阵选择 + 交换子归零引导）与本文提出的逐块选择方法。

记号与论文一致：t 从 0（噪声）到 1（数据），Euler 更新 x_{i+1} = x_i + (t_{i+1}-t_i) v(x_i, t_i)。
s = 2 时共有 4 个二值下采样候选 D_k，k = (dy, dx) 表示在每个 2×2 块中取第 (dy, dx) 个像素。
"""
import jax, jax.numpy as jnp, numpy as np
from model import velocity_jit

S = 2
CANDS = [(dy, dx) for dy in range(S) for dx in range(S)]
LR_COST = 1.0 / S ** 2          # 一次低分辨率前向 ≈ 1/4 次高分辨率前向（卷积计算量与像素数成正比）


def v(params, x, t):
    """在时刻 t 对整批输入 x 计算速度场 v_theta(x, t)。"""
    return velocity_jit(params, x, jnp.full((x.shape[0],), t, jnp.float32))


def sub(x, k):
    """二值下采样 D_k：在每个 s×s 块中取固定位置的元素。"""
    dy, dx = k
    return x[:, :, dy::S, dx::S]


def avgpool(x):
    """s×s 面积平均下采样（非二值下采样矩阵，作为对比方法）。"""
    n, c, h, w = x.shape
    return x.reshape(n, c, h // S, S, w // S, S).mean((3, 5))


def comm_map(target, v_lr):
    """逐像素交换子范数：||D v_HR - v_LR(Dx)||_2（沿通道），形状 (N,h,w)。"""
    return jnp.sqrt(((target - v_lr) ** 2).sum(1))


def euler(params, x, ts, i0, i1):
    """用 Euler 法从第 i0 个时间步积分到第 i1 个时间步。"""
    for i in range(i0, i1):
        x = x + (ts[i + 1] - ts[i]) * v(params, x, ts[i])
    return x


def hr_sample(params, x0, N=30):
    """完整的高分辨率 N 步采样，返回结果与计算量（高分辨率前向次数）。"""
    ts = np.linspace(0, 1, N + 1).astype(np.float32)
    return euler(params, x0, ts, 0, N), float(N)


def direct_lr(params, x0_lr, N=30):
    """直接在低分辨率上从噪声开始 N 步采样（对比方法），返回结果与计算量。"""
    ts = np.linspace(0, 1, N + 1).astype(np.float32)
    return euler(params, x0_lr, ts, 0, N), N * LR_COST


def preview(params, x0, N=30, D=10, select="global", cg=False, m=5, alpha=0.1, rng=None, track=False):
    """预览生成。

    select: 'nearest'（固定取左上角，朴素下采样）| 'avgpool'（面积平均）| 'random'（随机全局候选）
            | 'global'（论文：逐样本选交换子最小的全局 D_k，式(8)）
            | 'block'（本文：逐 2×2 块选交换子最小的位置）
    cg:     是否施加交换子归零引导（论文式(12)），作用于 t_D 及其后 m 个时间步。
    返回 (低分辨率预览, 以高分辨率前向次数计的计算量, 统计信息)。
    """
    ts = np.linspace(0, 1, N + 1).astype(np.float32)
    x = euler(params, x0, ts, 0, D)                      # 1) 高分辨率采样到 t_D
    cost = float(D)
    v_hr = v(params, x, ts[D]); cost += 1.0             # 保存高分辨率速度 v(x_{t_D}, t_D)
    n = x.shape[0]
    stats = {}

    if select == "avgpool":
        x_lr, tgt = avgpool(x), avgpool(v_hr)
        v_cur = None
    else:
        # 2) 计算 4 个候选的低分辨率速度与交换子（一次批量前向，开销 4 × 1/4 = 1 次高分辨率前向）
        Xk = jnp.stack([sub(x, k) for k in CANDS])            # (4,N,3,h,w)
        Tk = jnp.stack([sub(v_hr, k) for k in CANDS])
        if select in ("global", "block", "argmax"):
            Vk = v(params, Xk.reshape(-1, *Xk.shape[2:]), ts[D]).reshape(Xk.shape)
            cost += len(CANDS) * LR_COST
            Ck = jnp.sqrt(((Tk - Vk) ** 2).sum(2))           # (4,N,h,w) 逐像素交换子
            stats["comm_all"] = np.asarray(Ck.mean((2, 3)).T)   # (N,4) 各候选的平均交换子
        if select == "nearest":
            idx = np.zeros(n, int)
        elif select == "random":
            idx = rng.integers(0, len(CANDS), n)
        elif select == "global":
            idx = np.asarray(jnp.argmin(Ck.mean((2, 3)), 0))   # 逐样本选平均交换子最小的 D_k
        elif select == "argmax":
            idx = np.asarray(jnp.argmax(Ck.mean((2, 3)), 0))   # 消融：选交换子最大的 D_k
        if select == "block":
            bidx = jnp.argmin(Ck, 0)                            # (N,h,w) 每个块独立选择
            onehot = jax.nn.one_hot(bidx, len(CANDS), axis=0)[:, :, None]   # (4,N,1,h,w)
            x_lr = (Xk * onehot).sum(0)
            tgt = (Tk * onehot).sum(0)
            v_cur = None                                       # 混合矩阵的 v(Dx) 需重新计算
            stats["block_idx"] = np.asarray(bidx)
        else:
            sel = jnp.asarray(idx)
            x_lr = Xk[sel, jnp.arange(n)]
            tgt = Tk[sel, jnp.arange(n)]
            v_cur = Vk[sel, jnp.arange(n)] if select in ("global", "argmax") else None
            stats["idx"] = idx

    if v_cur is None:
        v_cur = v(params, x_lr, ts[D]); cost += LR_COST
    stats["comm_tD"] = np.asarray(comm_map(tgt, v_cur).mean((1, 2)))   # 选定 D 后 t_D 处的交换子

    # 3) 在低分辨率空间完成剩余采样；t_D ≤ t_i < t_{D+m+1} 时施加交换子归零引导
    for i in range(D, N):
        if i > D:
            v_cur = v(params, x_lr, ts[i]); cost += LR_COST
        if track and i == D + m + 1:
            stats["x_at"] = x_lr                                # 供分析：t_{D+m+1} 处的低分辨率状态
        if cg and i <= D + m:
            x_lr = x_lr + alpha * (tgt - v_cur)                 # 式(12)：复用 t_D 处的高分辨率速度
            v_cur = v(params, x_lr, ts[i]); cost += LR_COST
        x_lr = x_lr + (ts[i + 1] - ts[i]) * v_cur
    # gather：把同一个下采样矩阵作用到任意高分辨率张量上（供实验分析使用）
    if select == "block":
        stats["gather"] = lambda z: (jnp.stack([sub(z, k) for k in CANDS]) * onehot).sum(0)
    elif select == "avgpool":
        stats["gather"] = avgpool
    else:
        stats["gather"] = lambda z: jnp.stack([sub(z, k) for k in CANDS])[jnp.asarray(idx), jnp.arange(n)]
    return x_lr, cost, stats
