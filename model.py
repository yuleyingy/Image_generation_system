"""全卷积 U-Net 速度场模型 v_theta(x, t)，可在任意 4 的倍数分辨率上运行（16×16 与 32×32）。

全卷积结构使同一组参数既能生成高分辨率图像，也能直接在低分辨率上生成，
对应于 FLUX / SD3 等 DiT 模型可处理不同 token 数的性质。
"""
import jax
import jax.numpy as jnp
import numpy as np

CH = (32, 64, 64)    # 三个尺度的通道数
TDIM = 128           # 时间嵌入维度


def _conv_init(key, cin, cout, k=3):
    """初始化一个 k×k 卷积层的权重（He 初始化）和偏置。"""
    w = jax.random.normal(key, (cout, cin, k, k)) * np.sqrt(2.0 / (cin * k * k))
    return {"w": w, "b": jnp.zeros((cout,))}


def _dense_init(key, din, dout):
    """初始化一个全连接层的权重和偏置。"""
    return {"w": jax.random.normal(key, (din, dout)) * np.sqrt(1.0 / din), "b": jnp.zeros((dout,))}


def _res_init(key, cin, cout):
    """初始化一个带时间调制的残差块（两层卷积 + 组归一化 + 时间嵌入投影）。"""
    k = jax.random.split(key, 4)
    p = {"c1": _conv_init(k[0], cin, cout), "c2": _conv_init(k[1], cout, cout),
         "t": _dense_init(k[2], TDIM, 2 * cout),
         "g1": jnp.ones((cin,)), "b1": jnp.zeros((cin,)), "g2": jnp.ones((cout,)), "b2": jnp.zeros((cout,))}
    p["c2"]["w"] = p["c2"]["w"] * 0.1
    if cin != cout:
        p["skip"] = _conv_init(k[3], cin, cout, 1)
    return p


def init_params(key):
    """初始化整个 U-Net 速度场网络的全部参数，返回参数字典。"""
    ks = iter(jax.random.split(key, 32))
    c0, c1, c2 = CH
    return {
        "temb1": _dense_init(next(ks), TDIM, TDIM), "temb2": _dense_init(next(ks), TDIM, TDIM),
        "in": _conv_init(next(ks), 3, c0),
        "d0": _res_init(next(ks), c0, c0), "down0": _conv_init(next(ks), c0, c1),
        "d1": _res_init(next(ks), c1, c1), "down1": _conv_init(next(ks), c1, c2),
        "mid1": _res_init(next(ks), c2, c2), "mid2": _res_init(next(ks), c2, c2),
        "up1": _conv_init(next(ks), c2, c1), "u1": _res_init(next(ks), 2 * c1, c1),
        "up0": _conv_init(next(ks), c1, c0), "u0": _res_init(next(ks), 2 * c0, c0),
        "gout": jnp.ones((c0,)), "bout": jnp.zeros((c0,)),
        "out": {"w": jnp.zeros((3, c0, 3, 3)), "b": jnp.zeros((3,))},
    }


def _conv(p, x, stride=1):
    """执行一次 NCHW 格式的二维卷积并加上偏置，stride=2 时用于下采样。"""
    y = jax.lax.conv_general_dilated(x, p["w"], (stride, stride), "SAME",
                                     dimension_numbers=("NCHW", "OIHW", "NCHW"))
    return y + p["b"][None, :, None, None]


def _gn(x, g, b, groups=8, eps=1e-5):
    """组归一化（Group Normalization），g、b 为缩放和平移参数。"""
    n, c, h, w = x.shape
    xg = x.reshape(n, groups, c // groups, h, w)
    m = xg.mean((2, 3, 4), keepdims=True)
    v = xg.var((2, 3, 4), keepdims=True)
    xg = (xg - m) / jnp.sqrt(v + eps)
    return xg.reshape(n, c, h, w) * g[None, :, None, None] + b[None, :, None, None]


def _res(p, x, temb):
    """残差块前向计算：卷积-归一化-时间调制（scale/shift）-卷积，再与捷径相加。"""
    h = _conv(p["c1"], jax.nn.silu(_gn(x, p["g1"], p["b1"])))
    ss = temb @ p["t"]["w"] + p["t"]["b"]
    scale, shift = jnp.split(ss, 2, axis=-1)
    h = _gn(h, p["g2"], p["b2"]) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
    h = _conv(p["c2"], jax.nn.silu(h))
    skip = _conv(p["skip"], x) if "skip" in p else x
    return skip + h


def _up(x):
    """最近邻 2 倍上采样。"""
    return jnp.repeat(jnp.repeat(x, 2, axis=2), 2, axis=3)


def _time_embed(t):
    """把时间 t 映射为正弦/余弦位置编码向量。"""
    half = TDIM // 2
    freqs = jnp.exp(-np.log(1000.0) * jnp.arange(half) / half)
    a = (t * 1000.0)[:, None] * freqs[None]
    return jnp.concatenate([jnp.sin(a), jnp.cos(a)], -1)


def velocity(params, x, t):
    """x: (N,3,H,W)，t: (N,)，返回预测速度 v_theta(x,t)。"""
    te = _time_embed(t)
    te = jax.nn.silu(te @ params["temb1"]["w"] + params["temb1"]["b"])
    te = te @ params["temb2"]["w"] + params["temb2"]["b"]
    h0 = _res(params["d0"], _conv(params["in"], x), te)
    h1 = _res(params["d1"], _conv(params["down0"], h0, 2), te)
    h2 = _conv(params["down1"], h1, 2)
    h2 = _res(params["mid2"], _res(params["mid1"], h2, te), te)
    u1 = _res(params["u1"], jnp.concatenate([_conv(params["up1"], _up(h2)), h1], 1), te)
    u0 = _res(params["u0"], jnp.concatenate([_conv(params["up0"], _up(u1)), h0], 1), te)
    return _conv(params["out"], jax.nn.silu(_gn(u0, params["gout"], params["bout"])))


velocity_jit = jax.jit(velocity)
