"""用整流流（rectified flow）目标在 16×16 与 32×32 两种分辨率上联合训练速度场模型。

损失：|| v_theta(x_t, t) - (x_1 - x_0) ||^2，其中 x_t = (1-t) x_0 + t x_1。
低分辨率数据为高分辨率图像的 2×2 面积平均。
用法：python train.py --steps 12000
"""
import argparse, pickle, time
import jax, jax.numpy as jnp, numpy as np, optax
from data import sample_batch, avgpool
from model import init_params, velocity


def loss_fn(params, x1, x0, t):
    """整流流训练损失：网络预测速度与真实速度 x1 - x0 的均方误差。"""
    xt = (1 - t)[:, None, None, None] * x0 + t[:, None, None, None] * x1
    v = velocity(params, xt, t)
    return jnp.mean((v - (x1 - x0)) ** 2)


def main():
    """解析参数、在两种分辨率上交替训练模型，并定期保存 EMA 参数检查点。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--out", default="ckpt.pkl")
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    key = jax.random.PRNGKey(0)
    params = init_params(key)
    sched = optax.warmup_cosine_decay_schedule(0, 2e-3, min(300, a.steps // 10), a.steps, 1e-5)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4))
    state = opt.init(params)
    ema = jax.tree_util.tree_map(lambda p: p, params)

    @jax.jit
    def step(params, state, ema, x1, x0, t):
        """单步训练：计算梯度、用 AdamW 更新参数，并更新指数滑动平均（EMA）参数。"""
        l, g = jax.value_and_grad(loss_fn)(params, x1, x0, t)
        upd, state = opt.update(g, state, params)
        params = optax.apply_updates(params, upd)
        ema = jax.tree_util.tree_map(lambda e, p: 0.999 * e + 0.001 * p, ema, params)
        return params, state, ema, l

    t0, hist = time.time(), []
    for it in range(a.steps):
        x1 = sample_batch(rng, a.bs)
        if it % 2 == 1:                       # 交替训练两种分辨率
            x1 = avgpool(x1)
        x0 = rng.standard_normal(x1.shape).astype(np.float32)
        t = rng.uniform(0, 1, a.bs).astype(np.float32)
        params, state, ema, l = step(params, state, ema, x1, x0, t)
        hist.append(float(l))
        if (it + 1) % 200 == 0:
            print(f"step {it+1}  loss {np.mean(hist[-200:]):.4f}  {time.time()-t0:.0f}s", flush=True)
        if (it + 1) % 2000 == 0 or it + 1 == a.steps:
            pickle.dump({"params": jax.device_get(ema), "loss": hist}, open(a.out, "wb"))


if __name__ == "__main__":
    main()
