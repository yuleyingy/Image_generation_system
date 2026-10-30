"""扩散模型低分辨率预览生成系统（Gradio 网页界面）。

功能：
  1. 预览工作流：一次生成多张低分辨率预览 → 点选满意的一张 → 只对该种子生成高分辨率图像，
     并显示预览与高清结果的一致性指标、交换子热力图和节省的计算量；
  2. 方法对比：对同一种子并排比较直接低分辨率、朴素下采样、论文方法与本文方法，
     并可视化 4 个候选下采样矩阵的交换子图和本文逐块选择的结果；
  3. 使用说明。
启动：python app.py   然后在浏览器打开 http://127.0.0.1:7860
"""
import pickle, time
import numpy as np
import jax, jax.numpy as jnp
import gradio as gr
from PIL import Image
import matplotlib
matplotlib.use("Agg")
from matplotlib import cm

from preview import v, sub, CANDS, euler, hr_sample, direct_lr, preview, avgpool, LR_COST
from experiments import metrics

N, D = 30, 10
BATCH = 16                      # 固定批大小，避免不同预览数量触发重新编译
DISP = 224                      # 界面中图像的显示尺寸（像素）
P = jax.tree_util.tree_map(jnp.asarray, pickle.load(open("ckpt.pkl", "rb"))["params"])

SHORT = {"本文方法：逐块 D + 交换子归零引导": "本文：逐块D+CG", "论文方法：全局 D* + 交换子归零引导": "论文：全局D*+CG",
         "朴素下采样（最近邻）": "朴素下采样", "直接低分辨率生成": "直接低分辨率"}   # 画廊中的简短标签

METHODS = {
    "本文方法：逐块 D + 交换子归零引导": dict(select="block", cg=True),
    "论文方法：全局 D* + 交换子归零引导": dict(select="global", cg=True),
    "朴素下采样（最近邻）": dict(select="nearest", cg=False),
    "直接低分辨率生成": None,
}


# ------------------------------------------------------------------ 工具函数
def noise(seeds):
    """为一组种子生成高分辨率噪声 (n,3,32,32) 和同种子的低分辨率噪声 (n,3,16,16)。"""
    x0 = np.stack([np.random.default_rng(s).standard_normal((3, 32, 32)) for s in seeds]).astype(np.float32)
    x0l = np.stack([np.random.default_rng(s).standard_normal((3, 16, 16)) for s in seeds]).astype(np.float32)
    return jnp.asarray(x0), jnp.asarray(x0l)


def to_pil(x, size=DISP):
    """把 (3,H,W)、范围 [-1,1] 的图像转换为放大显示的 PIL 图像（最近邻放大，保留像素感）。"""
    a = (np.clip((np.asarray(x).transpose(1, 2, 0) + 1) / 2, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(a).resize((size, size), Image.NEAREST)


def heat_pil(m, vmax, size=DISP):
    """把二维数组 m 渲染为 inferno 配色的热力图 PIL 图像，vmax 为色标上限。"""
    rgb = (cm.inferno(np.clip(np.asarray(m) / max(vmax, 1e-6), 0, 1))[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb).resize((size, size), Image.NEAREST)


def run_method(name, x0, x0l, m, alpha):
    """按界面上选择的方法生成低分辨率预览，返回 (预览张量, 每张图的计算量)。"""
    cfg = METHODS[name]
    if cfg is None:
        return direct_lr(P, x0l, N)[:2]
    out, cost, _ = preview(P, x0, N, D, cfg["select"], cfg["cg"], int(m), float(alpha))
    return out, cost


def commutator_maps(x0):
    """计算 t_D 时刻 4 个候选下采样矩阵的逐像素交换子图 (4,n,16,16) 及逐块选择的索引图 (n,16,16)。"""
    ts = np.linspace(0, 1, N + 1).astype(np.float32)
    x = euler(P, x0, ts, 0, D)
    v_hr = v(P, x, ts[D])
    Xk = jnp.stack([sub(x, k) for k in CANDS])
    Tk = jnp.stack([sub(v_hr, k) for k in CANDS])
    Vk = v(P, Xk.reshape(-1, *Xk.shape[2:]), ts[D]).reshape(Xk.shape)
    Ck = jnp.sqrt(((Tk - Vk) ** 2).sum(2))
    return np.asarray(Ck), np.asarray(jnp.argmin(Ck, 0))


# ------------------------------------------------------------------ 页面 1：预览工作流
def make_previews(seed0, n, method, m, alpha):
    """生成从 seed0 开始的 n 张预览，返回画廊内容、状态信息和保存的种子列表。"""
    n = int(n)
    seeds = list(range(int(seed0), int(seed0) + n))
    padded = seeds + [seeds[-1]] * (BATCH - n)
    x0, x0l = noise(padded)
    t = time.time()
    out, cost = run_method(method, x0, x0l, m, alpha)
    jax.block_until_ready(out)
    sec = time.time() - t
    gallery = [(to_pil(out[i]), f"种子 {s}") for i, s in enumerate(seeds)]
    info = (f"已用 **{method}** 生成 {n} 张 16×16 预览，耗时 {sec:.2f} s。\n\n"
            f"每张预览的理论计算量为 {cost:.2f} 次高分辨率前向（完整高清生成为 {N} 次）。"
            f"请在左侧点选一张满意的预览，再点击“② 生成高清图”。")
    return gallery, info, {"seeds": seeds, "method": method, "m": m, "alpha": alpha, "cost": cost, "sec": sec}


def on_select(state, evt: gr.SelectData):
    """记录用户在画廊中点选的预览序号，返回更新后的状态和提示文字。"""
    state = dict(state or {})
    state["pick"] = evt.index
    return state, f"已选择：种子 {state['seeds'][evt.index]}"


def make_final(state):
    """对选中的种子生成高分辨率图像，计算预览与高清结果的一致性，并给出交换子热力图与计算量对比。"""
    if not state or "pick" not in state:
        raise gr.Error("请先生成预览并点选其中一张。")
    seeds, i = state["seeds"], state["pick"]
    seed = seeds[i]
    padded = [seed] * BATCH
    x0, x0l = noise(padded)
    t = time.time()
    hr, _ = hr_sample(P, x0, N)
    jax.block_until_ready(hr)
    sec_hr = time.time() - t
    prev, _ = run_method(state["method"], x0, x0l, state["m"], state["alpha"])
    ref = avgpool(hr)
    ps, ss, lc = metrics(np.asarray(prev[:1]), np.asarray(ref[:1]))
    Ck, _ = commutator_maps(x0)
    cmin = Ck[:, 0].min(0)
    n = len(seeds)
    cost_all = n * N
    cost_flow = n * state["cost"] + N
    report = (f"### 种子 {seed} 的结果\n"
              f"| 指标 | 数值 |\n|---|---|\n"
              f"| 预览与高清结果的 PSNR | {ps[0]:.2f} dB |\n"
              f"| SSIM | {ss[0]:.3f} |\n"
              f"| 布局色彩误差 LCE（越小越好） | {lc[0]:.2f} |\n"
              f"| 高清生成耗时 | {sec_hr:.2f} s |\n\n"
              f"### 工作流计算量对比（{n} 个候选）\n"
              f"- 传统方式：{n} 张都生成高清图，共 **{cost_all:.0f}** 次高分辨率前向；\n"
              f"- 预览工作流：{n} 张预览 + 1 张高清，共 **{cost_flow:.1f}** 次，节省 **{(1 - cost_flow / cost_all) * 100:.1f}%**。")
    return (to_pil(hr[0]), to_pil(prev[0]), to_pil(ref[0]),
            heat_pil(cmin, Ck.max()), report)


# ------------------------------------------------------------------ 页面 2：方法对比
def compare(seed, m, alpha):
    """对同一种子运行全部方法，返回各方法的图像、指标表和交换子可视化。"""
    x0, x0l = noise([int(seed)] * BATCH)
    hr, _ = hr_sample(P, x0, N)
    ref = avgpool(hr)
    imgs, rows = [(to_pil(hr[0]), "高清结果 32×32"), (to_pil(ref[0]), "参考 16×16")], []
    for name in reversed(list(METHODS)):
        out, cost = run_method(name, x0, x0l, m, alpha)
        ps, ss, lc = metrics(np.asarray(out[:1]), np.asarray(ref[:1]))
        imgs.append((to_pil(out[0]), SHORT[name]))
        rows.append([name, f"{N / cost:.2f}×", f"{ps[0]:.2f}", f"{ss[0]:.3f}", f"{lc[0]:.2f}"])
    Ck, idx = commutator_maps(x0)
    vmax = Ck[:, 0].max()
    heat = [(heat_pil(Ck[k, 0], vmax, 160), f"候选D{k + 1}：{Ck[k, 0].mean():.3f}") for k in range(4)]
    pal = np.array([[230, 75, 53], [77, 187, 213], [0, 160, 135], [243, 155, 127]], np.uint8)
    sel = Image.fromarray(pal[idx[0]]).resize((160, 160), Image.NEAREST)
    heat.append((sel, "逐块选择图"))
    heat.append((heat_pil(Ck[:, 0].min(0), vmax, 160), f"逐块最小：{Ck[:, 0].min(0).mean():.3f}"))
    return imgs, rows, heat


# ------------------------------------------------------------------ 界面
HELP = """
## 系统说明
本系统演示 CVPR 2026 论文提出的**低分辨率预览生成**工作流，以及本文提出的**逐块交换子最小化下采样（BCD）**改进。

- **为什么需要预览**：文生图模型换一个随机种子结果就完全不同，用户往往要试很多个种子。
  预览工作流先快速生成一批小图，选中满意的种子后，只对它生成一次高清图。
- **关键要求**：预览必须和最终的高清图“长得一样”。“方法对比”页可以看到，直接生成小图完全做不到这一点。
- **论文方法**：在高分辨率下采样到第 D 步后下采样，从 4 个候选下采样矩阵中选交换子最小的一个，
  再用“交换子归零引导”修正若干步。
- **本文方法**：在每个 2×2 块内独立选择交换子最小的位置（“方法对比”页的彩色选择图），
  噪声分布保持不变，额外开销只有一次低分辨率前向。

**参数**：引导步数 m 与步长 α 的默认值（5，0.1）是在验证集上选出的最优值。
**说明**：模型是在合成数据（渐变背景 + 几何体）上从零训练的小型流匹配模型，可在 CPU 上实时运行。
"""


def build_ui():
    """搭建 Gradio 界面并绑定各按钮的回调函数，返回 Blocks 对象。"""
    with gr.Blocks(title="扩散模型低分辨率预览生成系统") as demo:
        gr.Markdown("# 扩散模型低分辨率预览生成系统\n先看小图、再出大图：免训练、感知一致的预览生成与逐块下采样改进")
        with gr.Tab("① 预览工作流"):
            state = gr.State({})
            with gr.Row():
                with gr.Column(scale=1):
                    seed0 = gr.Number(value=0, precision=0, label="起始随机种子")
                    n = gr.Slider(4, BATCH, value=12, step=1, label="预览数量")
                    method = gr.Radio(list(METHODS), value=list(METHODS)[0], label="预览生成方法")
                    with gr.Accordion("引导参数", open=False):
                        m = gr.Slider(1, 7, value=5, step=1, label="引导步数 m")
                        alpha = gr.Slider(0.0, 0.4, value=0.1, step=0.01, label="引导步长 α")
                    btn1 = gr.Button("① 生成预览", variant="primary")
                    info = gr.Markdown()
                    pick = gr.Markdown()
                    btn2 = gr.Button("② 生成高清图", variant="secondary")
                with gr.Column(scale=2):
                    gallery = gr.Gallery(label="低分辨率预览（点选一张）", columns=4, height=520, allow_preview=False)
            with gr.Row():
                o_hr = gr.Image(label="高清结果 32×32", type="pil")
                o_pv = gr.Image(label="所选预览 16×16", type="pil")
                o_ref = gr.Image(label="高清结果下采样（参考）", type="pil")
                o_heat = gr.Image(label="t_D 时刻交换子热力图（逐块最小值）", type="pil")
            o_rep = gr.Markdown()
            btn1.click(make_previews, [seed0, n, method, m, alpha], [gallery, info, state])
            gallery.select(on_select, [state], [state, pick])
            btn2.click(make_final, [state], [o_hr, o_pv, o_ref, o_heat, o_rep])
        with gr.Tab("② 方法对比"):
            with gr.Row():
                cseed = gr.Number(value=0, precision=0, label="随机种子")
                cm_ = gr.Slider(1, 7, value=5, step=1, label="引导步数 m")
                ca = gr.Slider(0.0, 0.4, value=0.1, step=0.01, label="引导步长 α")
                cbtn = gr.Button("运行对比", variant="primary")
            cgal = gr.Gallery(label="各方法生成结果", columns=6, height=300, allow_preview=False)
            ctab = gr.Dataframe(headers=["方法", "理论加速比", "PSNR/dB", "SSIM", "LCE"], label="与参考图的一致性")
            hgal = gr.Gallery(label="t_D 时刻交换子分析（越亮表示交换子越大；数值为空间平均；逐块选择图中颜色表示所选候选）", columns=6, height=240, allow_preview=False)
            cbtn.click(compare, [cseed, cm_, ca], [cgal, ctab, hgal])
        with gr.Tab("③ 使用说明"):
            gr.Markdown(HELP)
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860)
