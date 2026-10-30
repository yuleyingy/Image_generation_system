"""全部实验：交换子分析、主对比、消融、超参数、速度场一致性与逐块选择方法。

用法：python experiments.py --ckpt ckpt.pkl --out results   （--quick 用少量样本快速检查）
所有数值写入 results/results.json，图写入 results/*.png。
"""
import argparse, json, os, pickle, time
import jax, jax.numpy as jnp, numpy as np
from scipy.stats import wilcoxon
from skimage.metrics import structural_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from preview import v, hr_sample, direct_lr, preview, avgpool, sub, CANDS, euler

N, D = 30, 10


def noise(seeds, hr=32):
    """同一种子：高分辨率噪声；直接低分辨率生成时用同一种子生成另一形状的噪声（模拟实际软件中的行为）。"""
    x0 = np.stack([np.random.default_rng(s).standard_normal((3, hr, hr)) for s in seeds]).astype(np.float32)
    x0l = np.stack([np.random.default_rng(s).standard_normal((3, hr // 2, hr // 2)) for s in seeds]).astype(np.float32)
    return jnp.asarray(x0), jnp.asarray(x0l)


def to01(x):
    """把 [-1,1] 范围的图像张量转换为 [0,1] 的 numpy 数组，便于计算指标。"""
    return np.clip((np.asarray(x) + 1) / 2, 0, 1)


def metrics(pred, ref):
    """计算预览 pred 与参考 ref 的逐样本 PSNR、SSIM 和布局色彩误差 LCE。"""
    p, r = to01(pred), to01(ref)
    mse = ((p - r) ** 2).mean((1, 2, 3))
    psnr = 10 * np.log10(1 / np.maximum(mse, 1e-10))
    ssim = np.array([structural_similarity(a.transpose(1, 2, 0), b.transpose(1, 2, 0), channel_axis=2,
                                           data_range=1.0, win_size=5) for a, b in zip(p, r)])
    # 布局色彩误差：4×4 粗网格平均颜色的平均绝对误差（×255），反映构图与配色是否一致
    pc = p.reshape(len(p), 3, 4, 4, 4, 4).mean((3, 5)); rc = r.reshape(len(r), 3, 4, 4, 4, 4).mean((3, 5))
    lce = np.abs(pc - rc).mean((1, 2, 3)) * 255
    return psnr, ssim, lce


def run(fn, seeds, bs):
    """按批运行某个生成方法 fn，返回全部输出、单张计算量、附加统计和总耗时（秒）。"""
    outs, cost, extra, t0 = [], None, [], time.time()
    for i in range(0, len(seeds), bs):
        x0, x0l = noise(seeds[i:i + bs])
        o = fn(x0, x0l)
        outs.append(np.asarray(o[0])); cost = o[1]
        extra.append(o[2] if len(o) > 2 else {})
        jax.block_until_ready(o[0])
    return np.concatenate(outs), cost, extra, time.time() - t0


def summarize(name, out, ref, cost, sec, n):
    """汇总一个方法的平均指标、计算量、理论加速比和每张图耗时。"""
    ps, ss, lc = metrics(out, ref)
    return {"name": name, "psnr": float(ps.mean()), "ssim": float(ss.mean()), "lce": float(lc.mean()),
            "psnr_all": ps.tolist(), "cost": cost, "flop_speedup": 30.0 / cost, "sec_per_img": sec / n}


def main():
    """依次执行全部实验，并把数值结果与样例图像保存到输出目录。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt.pkl"); ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    P = pickle.load(open(a.ckpt, "rb"))["params"]
    P = jax.tree_util.tree_map(jnp.asarray, P)
    ntest, nval, bs = (60, 40, 30) if a.quick else (300, 100, 100)
    test, val = list(range(ntest)), list(range(10000, 10000 + nval))
    R = {}
    rng = np.random.default_rng(123)

    # warm-up jit for all shapes
    x0, x0l = noise(test[:bs]); preview(P, x0, select="global", cg=True); direct_lr(P, x0l, 2)

    # ---------- 参考：30 步高分辨率 ----------
    hr, _, _, sec_hr = run(lambda x0, x0l: hr_sample(P, x0, N), test, bs)
    ref = np.asarray(avgpool(jnp.asarray(hr)))
    R["ref"] = {"cost": 30.0, "sec_per_img": sec_hr / ntest}

    # ---------- 1. 交换子分析（对应论文表1 / 图7） ----------
    x0, _ = noise(test[:bs])
    ts = np.linspace(0, 1, N + 1).astype(np.float32)
    x, comm_t, cos_k = x0, {}, []
    vs = []
    for i in range(N):
        vi = v(P, x, ts[i]); vs.append(vi)
        if i in (3, 6, 10, 15, 20, 25):
            c = jnp.sqrt(((sub(vi, (0, 0)) - v(P, sub(x, (0, 0)), ts[i])) ** 2).sum(1)).mean((1, 2))
            vnorm = jnp.sqrt((sub(vi, (0, 0)) ** 2).sum(1)).mean((1, 2))
            comm_t[f"{ts[i]:.2f}"] = {"comm": float(c.mean()), "comm_std": float(c.std()), "vnorm": float(vnorm.mean())}
        x = x + (ts[i + 1] - ts[i]) * vi
    for k in range(1, 11):
        a_, b_ = vs[D].reshape(bs, -1), vs[D + k].reshape(bs, -1)
        cs = (a_ * b_).sum(1) / (jnp.linalg.norm(a_, axis=1) * jnp.linalg.norm(b_, axis=1))
        cos_k.append([float(cs.mean()), float(cs.std())])
    R["comm_over_t"], R["vel_cos"] = comm_t, cos_k

    # ---------- 2. 超参数（验证集上选 m 与 α） ----------
    vhr, _, _, _ = run(lambda x0, x0l: hr_sample(P, x0, N), val, bs)
    vref = np.asarray(avgpool(jnp.asarray(vhr)))
    grid = {}
    ms, alphas = ([3, 5], [0.1, 0.3]) if a.quick else ([3, 5], [0.05, 0.1, 0.2, 0.3])
    for sel in ("global", "block"):
        for m in ms:
            for al in alphas:
                o, c, _, _ = run(lambda x0, x0l: preview(P, x0, N, D, sel, True, m, al), val, bs)
                grid[f"{sel}|{m}|{al}"] = float(metrics(o, vref)[0].mean())
                print("grid", sel, m, al, grid[f"{sel}|{m}|{al}"], flush=True)
    R["grid"] = grid
    best = {sel: max(((k, s) for k, s in grid.items() if k.startswith(sel)), key=lambda z: z[1])[0].split("|")
            for sel in ("global", "block")}
    R["best"] = {k: [int(v_[1]), float(v_[2])] for k, v_ in best.items()}

    # ---------- 3. 主对比与消融（测试集） ----------
    mg, ag = R["best"]["global"]; mb, ab = R["best"]["block"]
    # 每个方法写成 (名称, 生成函数)；生成函数输入同一种子的高/低分辨率噪声，返回 (输出, 计算量[, 统计])
    methods = [
        ("NFE=20（高分辨率）", lambda x0, x0l: (avgpool(hr_sample(P, x0, 20)[0]), 20.0)),
        ("NFE=10（高分辨率）", lambda x0, x0l: (avgpool(hr_sample(P, x0, 10)[0]), 10.0)),
        ("NFE=6（高分辨率）", lambda x0, x0l: (avgpool(hr_sample(P, x0, 6)[0]), 6.0)),
        ("直接低分辨率生成", lambda x0, x0l: direct_lr(P, x0l, N)),
        ("朴素下采样（最近邻）", lambda x0, x0l: preview(P, x0, N, D, "nearest")),
        ("面积平均下采样", lambda x0, x0l: preview(P, x0, N, D, "avgpool")),
        ("随机 D", lambda x0, x0l: preview(P, x0, N, D, "random", rng=rng)),
        ("argmax D", lambda x0, x0l: preview(P, x0, N, D, "argmax")),
        ("全局 D*（无 CG）", lambda x0, x0l: preview(P, x0, N, D, "global")),
        ("最近邻 D + CG", lambda x0, x0l: preview(P, x0, N, D, "nearest", True, mg, ag)),
        ("论文方法：全局 D* + CG", lambda x0, x0l: preview(P, x0, N, D, "global", True, mg, ag, track=True)),
        ("逐块 D（无 CG）", lambda x0, x0l: preview(P, x0, N, D, "block")),
        ("本文方法：逐块 D + CG", lambda x0, x0l: preview(P, x0, N, D, "block", True, mb, ab, track=True)),
        ("论文方法（N=15）", lambda x0, x0l: preview(P, x0, 15, 5, "global", True, 3, 2 * ag)),
        ("本文方法（N=15）", lambda x0, x0l: preview(P, x0, 15, 5, "block", True, 3, 2 * ab)),
    ]
    R["methods"], outs, extras = [], {}, {}
    for name, fn in methods:
        o, c, ex, sec = run(fn, test, bs)
        R["methods"].append(summarize(name, o, ref, c, sec, ntest))
        outs[name], extras[name] = o, ex
        print(name, {k: R["methods"][-1][k] for k in ("psnr", "ssim", "lce", "flop_speedup")}, flush=True)

    # 交换子：t_D 处各方法选定 D 后的平均交换子
    R["comm_tD"] = {name: float(np.concatenate([e["comm_tD"] for e in extras[name]]).mean())
                    for name in extras if extras[name] and "comm_tD" in extras[name][0]}

    # ---------- 4. 引导前后交换子变化（对应论文图6）：沿真实高分辨率轨迹在 t_{D+m+1} 处测量 ----------
    def comm_change(sel, cg, m, al):
        """测量 t_D 处与 t_{D+m+1} 处的交换子范数及轨迹偏差，用于分析交换子归零引导的作用。"""
        before, after, dev = [], [], []
        for i in range(0, ntest, bs):
            x0, _ = noise(test[i:i + bs])
            out, _, st = preview(P, x0, N, D, sel, cg, m, al, track=True)
            xh = euler(P, x0, ts, 0, D + m + 1)
            tgt = st["gather"](v(P, xh, ts[D + m + 1]))
            c = jnp.sqrt(((tgt - v(P, st["x_at"], ts[D + m + 1])) ** 2).sum(1)).mean((1, 2))
            before.append(st["comm_tD"]); after.append(np.asarray(c))
            # 轨迹偏差：低分辨率状态与高分辨率轨迹下采样之间的距离（逐像素 L2 的空间平均）
            dev.append(np.asarray(jnp.sqrt(((st["x_at"] - st["gather"](xh)) ** 2).sum(1)).mean((1, 2))))
        b, aft = np.concatenate(before), np.concatenate(after)
        return b, aft, np.concatenate(dev)
    cc = {}
    for lab, sel, cg, m, al in [("global_noCG", "global", False, mg, ag), ("global_CG", "global", True, mg, ag),
                                 ("block_noCG", "block", False, mb, ab), ("block_CG", "block", True, mb, ab)]:
        b, aft, dv = comm_change(sel, cg, m, al)
        d = aft - b
        p = float(wilcoxon(d, alternative="less" if d.mean() < 0 else "greater").pvalue)
        cc[lab] = {"tD": float(b.mean()), "after": float(aft.mean()), "delta": float(d.mean()), "p": p,
                   "dev": float(dv.mean()), "d_all": d.tolist(), "dev_all": dv.tolist()}
    R["comm_change"] = cc

    # 噪声统计：面积平均会使噪声分量的标准差缩小为 1/s（理论分析佐证）
    z = np.random.default_rng(0).standard_normal((200, 3, 32, 32)).astype(np.float32)
    R["noise_std"] = {"subsample": float(np.asarray(sub(jnp.asarray(z), (0, 0))).std()),
                      "avgpool": float(np.asarray(avgpool(jnp.asarray(z))).std())}
    # 逐块方法中各候选位置被选中的比例
    bi = np.concatenate([e["block_idx"] for e in extras["本文方法：逐块 D + CG"]])
    R["block_pick_frac"] = [float((bi == k).mean()) for k in range(4)]
    gi = np.concatenate([e["idx"] for e in extras["论文方法：全局 D* + CG"]])
    R["global_pick_frac"] = [float((gi == k).mean()) for k in range(4)]

    # 逐样本配对检验：本文方法 vs 论文方法（PSNR）
    pa = np.array(R["methods"][[m_["name"] for m_ in R["methods"]].index("论文方法：全局 D* + CG")]["psnr_all"])
    pb = np.array(R["methods"][[m_["name"] for m_ in R["methods"]].index("本文方法：逐块 D + CG")]["psnr_all"])
    R["paired"] = {"mean_diff": float((pb - pa).mean()), "win_rate": float((pb > pa).mean()),
                   "p": float(wilcoxon(pb - pa).pvalue)}

    json.dump(R, open(f"{a.out}/results.json", "w"), ensure_ascii=False, indent=1)
    np.savez_compressed(f"{a.out}/samples.npz", hr=hr[:16], ref=ref[:16],
                        **{f"m{i}": outs[n][:16] for i, (n, _) in enumerate(methods)})
    print("done")


if __name__ == "__main__":
    main()
