# 梯度域局部融接工作台

离散梯度域融合工具：把来源图的选区以梯度引导的方式融进目标图，并给出可核验的
数值诊断（方程残差、边界跳变、亮度漂移、超色域比例等），而不是按键值观感判断。

本仓库在原有功能之上加入了**稀疏求解优化**与**拼缝分区求解**，并把对照基准与
可执行证据一并入库。

---

## 1. 运行方式

### 1.1 界面（推荐）

```bash
python3 -m pip install numpy scipy pillow flask    # 依赖
python3 -m webapp.server                            # 启动工作台
# 浏览器打开 http://127.0.0.1:5000/
```

界面是梯度融接内核驱动的控件：选区绘制、放置点、锚点、羽化、引导策略、优先级、
拼缝分区都在同一个页面里编辑，点击「建立并求解离散系统」直接调用 `blendtool`
内核。数值诊断、求解器路径、缓存命中、拼缝分区状态都回显在右侧面板；求解失败
以「求解失败：…」红字反馈，不会静默吞掉。

典型操作（已在 jsdom 中逐按钮验证）：载入合成示例 → 在目标图点击加锚点 →
拖动放置点 / 勾选拼缝分区并填写 x → 求解 → 用锚点列表里的 ✕ 移除右侧最后一个
锚 → 再求解，面板会显示 `pending_right` 与「右侧绝对亮度已确定：否（待定）」，
并提示补锚重解。

### 1.2 命令行

```bash
# 合成对照（已知解析梯度）
python3 -m blendtool.cli demo --size 64 --outdir demo_out

# 单次矩形融接；--seam 启用拼缝分区
python3 -m blendtool.cli blend src.png tgt.png \
    --rect 12,12,52,52 --target 32,32 --strategy source --feather 4 \
    --seam 32 --out result.png --report report.json

# 按保存的参数链重跑
python3 -m blendtool.cli run project.json --outdir out/
```

### 1.3 测试与基准

```bash
python3 -m pytest -q
python3 -m blendtool.benchmark --sizes 64,96,128,192 --feathers 4,6 \
    --out bench_out          # 写入 bench_out/benchmark.{json,csv}
```

---

## 2. 稀疏求解优化

### 2.1 原来的瓶颈

软选区 WLS 路径（`_solve_wls`）每次求解都做三件事：

1. 逐边 Python 循环组装最小二乘矩阵 `A`（每条边一次解释器步骤）；
2. 对每个颜色通道跑一遍**无预条件 CG**（`rtol=1e-12`），即 3 次迭代求解；
3. 从不复用，每次编辑都从头再来。

实测 H=192 时，3 次 CG 约占去 0.56 s / 0.74 s；且无预条件 CG 的迭代次数随区域
增大上升（见基准表 `ref_cg_iters` 702→1752），单次耗时在相同输入上也会数倍抖动
（实测 398–1804 ms）。这不是"参数没调好"，而是迭代法与条件数耦合的结构性成本。

### 2.2 改动

- **向量化组装**（`assemble_wls`）：用 NumPy 索引一次性构造 COO 三元组，O(E)
  的向量运算取代 O(E) 的 Python 步骤，数学完全不变。
- **缓存分解**（`blendtool/system_cache.py` + `_solve_wls`）：对
  `AᵀA` 做稀疏 LU（`splu`），以**结构指纹**为键存入有界 LRU。
  指纹覆盖一切会改变 `A` 的输入：未知像素集合、边拓扑、每边权重、锚点位置与
  权重、优先级、gauge 选择；**不含**锚点颜色、目标/来源像素值这些只改右端项的
  东西。因此"只改颜色/拖动放置"这类编辑命中缓存，只付右端组装 + 三次三角回代。
- **硬选区路径同样缓存**：硬约束 Laplacian 只依赖区域几何，按同一机制复用分解。
- **基准与参照**：保留优化前的 `solve_wls_reference` 原样可跑，用于前后对照。

### 2.3 计算顺序（为何先组装、再指纹、后分解）

```
blend → 嵌入源图/掩膜 → build_edges → guidance_field
      → (_solve_hard | assemble_wls → fingerprint → cache.get ?
            命中: 复用 LU            : 计算 AᵀA → splu → cache.put)
      → Aᵀb → 3 次 lu.solve → 写回像素 → _finalize_metrics
```

指纹必须在 `assemble_wls` 之后取，因为组装阶段的 `singular`（是否加 gauge 行）
本身会改变 `A` 的形状；把它纳入指纹才能保证 gauge 与非 gauge 的分解不会互相串用。

### 2.4 实测（`bench_out/benchmark.csv`）

`ref` = 原 CG 路径；`cold` = 向量化+分解首次；`warm` = 同结构改右端（换锚色）后的
缓存命中；`full` = 同结构的无缓存直解。残差为正规方程残差 `‖Aᵀ(Ax-b)‖_rms`。

| size | unknowns | ref ms | cold ms | warm ms | ref CG iters | 残差(ref/cold/warm/full) |
|------|---------:|-------:|--------:|--------:|-------------:|---------------------------|
| 64   | 2397     | 27.3   | 5.0     | 0.67    | 702          | 4e-14 / 2e-16 / 2e-16 / 2e-16 |
| 128  | 8277     | 87.5   | 16.3    | 2.30    | 1164         | 4e-14 / 2e-16 / 2e-16 / 2e-16 |
| 192  | 19032    | 682–3147 | 53–56 | 6.2–7.4 | 1611–1752    | 5e-14 / 1e-16 / 1e-16 / 1e-16 |

要点：
- 冷启动即 5–10×；只改右端的编辑 20–400×（且随规模拉大）。
- 三条路径的正规方程残差同量级（~1e-16），说明加速没有以精度为代价。
- 参照列给出迭代次数，是比墙钟更稳定的成本证据；墙钟抖动源于无预条件 CG 的
  收敛波动，表格按多次运行取中位数并保留 `time_ms_all` 原始样本，**不以默认值
  或单次幸运值掩盖真实成本**。

### 2.5 撤回准则

若在**方程残差与求解成本之外**观察到行为退化（例如警告文本、连通分量信息、
漂移/边界指标变化，或输出与参照路径超出求解容差），应撤回对应优化：

- 只把 `_solve_wls` 的 `use_cache` 默认改为 `False` 可退回"每次重解"；
- 若向量化组装有问题，可让 `_solve_wls` 调用 `solve_wls_reference`；
- 回归门槛见 `tests/test_optimization.py::TestOptimizedMatchesReference` 与
  `TestFactorizationCache`（形状改变必须失效、缓存与直解须在 1e-10 内一致）。

---

## 3. 拼缝分区求解（`blendtool/seam.py`）

在选区里沿目标画布某列 `seam_column` 剪一条竖直细缝，把区域一分为二：

- **左侧 L**：常规行——内部梯度行、朝向目标的外边界 Dirichlet 行、自身锚点。
- **右侧 R**：只有内部梯度行；外边界**不**钉到目标。每条跨缝边贡献一条相对行
  `x_R - x_L - c = g_seam`，`c` 是一个共享的自由标量，也就是右侧唯一的绝对
  亮度基准。
- 右侧内有一个亮度锚 → 通过跨缝行把 `c` 钉住 → 右侧绝对亮度确定
  （`status="determined"`）。此时左右各有锚，**改动任一侧锚色只移动该侧均值亮度，
  另一侧基本不动**（测试验证：改左锚，右均值变化 < 0.01；改右锚，左均值不变）。
- 移除右侧最后一个锚 → `c` 自由，梯度与拼缝差只决定右侧**相对**亮度 →
  `status="pending_right"`。此时**不发布右侧绝对色值**：右侧像素保持为目标图，
  `unknown_mask` 只覆盖左侧，并给出提示，让编辑者补锚重解或保留待定。
- 指纹包含锚点位置与权重，所以"有锚时的分解"不会在"去锚后"被复用而发布旧的
  绝对分解；区域形状改变会改变指纹并触发重解，新像素不会漏掉。

导出后用两项验收（均写入 `metrics["seam"]`）：

- `left_mean_lum` / `right_mean_lum`：两侧平均亮度；
- `seam_gradient_rms`、`seam_gradient_error_rms`、`seam_visible_jump_rms_lum`：
  拼缝处实现梯度、对请求梯度的误差、以及可见亮度跳变。

> 说明：`seam_gradient_error_rms` 是拼缝行残差 `x_R - x_L - c - g`；`c` 是解的一部分，
> 因此另列 `seam_source_gradient_error_rms`（对源梯度的偏差）与可见跳变，避免把
> "有意的绝对偏移 c"误读成求解误差。

---

## 4. 相对完整求解的对照与缓存污染排查

- `tests/test_optimization.py::test_seam_fusion_comparison_against_direct_full_solve`
  同时跑拼缝解与直接全解，确认两者都是有限结果、拼缝行残差小、锚点被满足。
- `test_full_solve_distinguishes_pollution_from_tolerance`：热缓存解与无缓存直解
  必须在 1e-10 内一致——若缓存被污染，这里会出现大偏差而不是容差级差异。
- `test_shape_change_invalidates_cache`：扩大选区后 `cache_hit=False`，且新像素
  确实被求解（不是沿用旧矩阵漏掉的新像素）。

---

## 5. 模块地图

| 文件 | 作用 |
|------|------|
| `blendtool/solver.py` | 系统组装、求解、指标（含参照 CG 路径） |
| `blendtool/seam.py` | 拼缝分区求解与分区验收指标 |
| `blendtool/system_cache.py` | 结构指纹 + 有界 LRU 分解缓存 |
| `blendtool/gradient.py` | 引导梯度场与邻边构造 |
| `blendtool/masks.py` | 选区栅格化、羽化、连通分量、放置 |
| `blendtool/pipeline.py` | 可保存参数链（支持 `seam_column`） |
| `blendtool/benchmark.py` | 参照/优化/缓存/直解对照与成本分解 |
| `blendtool/cli.py` | `run` / `blend --seam` / `demo` |
| `webapp/server.py` + `templates/index.html` | Flask 工作台 |
| `bench_out/` | 入库的基准证据（json + csv） |

## 6. 已知限制

- 拼缝目前是**竖直单缝**（x 列切分）；斜缝/多条缝不在本实现内。
- 右侧在 `pending_right` 时不求解，保持目标色；这是刻意的"宁可不发布"策略。
- 无预条件 CG 参照仍保留抖动特性，这是被记录的现象而非被抹平的缺陷。
