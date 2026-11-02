# OBPDS — Open Band Parameters Device Simulator

一维 III-V 化合物半导体异质结能带与输运仿真器，附带 Web 可视化工作台。

本项目是对 [scott-maddox/obpds](https://github.com/scott-maddox/obpds)（2015 年
停更，Python 2.7）的现代重写：保留其学术内核（Kane 非抛物统计、多能谷、阻尼牛顿
求解），修正其中的物理与数值缺陷，并补上原项目完全没有的漂移-扩散输运，从而第一次
给出 I–V 特性。

## 快速开始

```bash
pip install -e ".[web]"
obpds-web                      # 打开 http://127.0.0.1:8000
# 或
uvicorn web.app:app --reload
```

命令行求解：

```python
from obpds import Layer, OhmicContact, TwoTerminalDevice

dev = TwoTerminalDevice(
    [Layer(1000, "GaAs", Na=1e17),      # 厚度 nm；Na 受主，Nd 施主
     Layer(1000, "GaAs", Nd=1e17)],
    [OhmicContact(), OhmicContact()],
    T=300, N=800, approx="kane",
)
sol = dev.get_equilibrium()             # 平衡能带
dd  = dev.get_drift_diffusion(0.8)      # 漂移-扩散，dd.J 为 A/cm^2
V, J = dev.get_iv(-0.5, 1.0, 31)        # I–V 扫描
V, C = dev.get_cv(-1.0, 0.3, 41)        # C–V 扫描
```

### 参数标定

表值对不上具体样品时，`Layer` 可以按层覆盖价带偏移与形变势：

```python
Layer(8, "GaInAs", composition=0.8, overrides={"VBO": -0.42, "b_shear": -1.9})
```

覆盖的是**未应变**的表值，衬底应变在其后照常作用。只开放这两类参数——其余参数会牵动
`ni`、`Nc`、`Nv`、非抛物系数等派生量，半套更新比不改更糟，那些请直接改
`src/obpds/materials/data.py`。键名写错会直接抛 `ValueError`，不会静默忽略。

## 物理模型

| 模块 | 内容 |
| --- | --- |
| `fermi.py` | 费米-狄拉克积分 `F_{1/2}` 与 Kane 广义积分 `H_{1/2} = G_{1/2} + 2αG_1`，Gauss-Legendre 求积配 `t = u²` 代换消除端点奇性 |
| `materials/` | Vurgaftman et al. (2001) 的 12 种二元 + 18 种三元 III-V 参数，Adachi 介电常数，Varshni `Eg(T)`，Γ/L/X 多能谷 |
| `solver.py` | 有限体积泊松求解，界面介电常数取**调和平均** |
| `transport.py` | Scharfetter-Gummel 离散的漂移-扩散，Gummel 迭代，SRH + 辐射复合 |
| `newton.py` | Bank & Rose 阻尼牛顿 + 行均衡预条件 |
| `quantum.py` | 自洽 Schrödinger–Poisson：BenDaniel-Duke 有效质量方程、2D 子带占据、连续谱权重 |
| `strain.py` | 赝晶应变：双轴应变张量、Bir-Pikus 价带分裂、静水压位移、Matthews-Blakeslee 临界厚度 |

### 与原始实现的差异

1. **异质结处的 ε 梯度项。** 原实现把 `div(ε grad ψ)` 离散成 `ε d²ψ/dx²`，丢掉了
   `dε/dx · dψ/dx`。有限体积格式下界面介电常数取调和平均
   `ε_{i+1/2} = 2ε_i ε_{i+1}/(ε_i + ε_{i+1})` 后该项自动保留。
2. **广义 SG 势。** 不假设玻尔兹曼统计，而以
   `Psi_n = F_n − kT ln(n/Nc_ref)`、`Psi_p = F_p + kT ln(p/Nv_ref)` 定义广义势。
   这对任意统计都是恒等式，一次性携带 ψ、能带失调、DOS 梯度与简并修正，且使平衡态
   电流精确为零。
3. **SRH 陷阱密度。** 分母必须用陷阱能级对应的 `n1`、`p1`（中点陷阱即
   `n_eff = √(n0 p0)`），不是局域平衡密度 `n0`、`p0`。后者等价于让陷阱能级随费米能级
   滑动，在耗尽区会把分母抬高五个数量级，反向产生电流严重偏小且几乎不随偏压变化。
4. **复合的参考态。** 驱动力取 `np − n0·p0` 而非 `np − ni²`：Kane / 费米-狄拉克统计下
   平衡时 `np ≠ ni²`，残余会伪造饱和电流。
5. **准费米能级初值。** 冶金结按净掺杂**变号**定位（不是"首次变为 n 型"），少子剖面取
   `c(s) = c0(e^{V/kT}(1−s) + s)` 的对数形式，对正反偏都成立。
6. **不做偏置延续。** 欧姆接触钉扎的是载流子**密度**而非准费米能级；复用上一偏压点的
   电平会让接触密度差 `e^{ΔV/kT}`，首次非线性泊松求解即发散。冷启动 2–3 次 Gummel
   迭代收敛。
7. **终端电流取自耗尽区。** 准中性区以 `q μ c dF/dx` 承载电流，`c ~ 1e17` 时
   1e-15 A/cm² 对应的梯度比存储电平的舍入误差低五个数量级，那些网格是纯噪声。

## 量子修正（自洽 Schrödinger–Poisson）

经典泊松求解把电子气堆在导带边的局部极小点上，半导体物理上这是错的：沟道里的电子
被限制在几个 nm 宽的势阱中，其最低能级高于带边，且电荷质心位于界面**之后**若干 nm。
Web 界面勾选「量子修正」（仅平衡能带模式）或调用

```python
sol = dev.get_quantum_equilibrium(n_states=8)
sol.subbands["levels_eV"]        # 本征能级
sol.subbands["sheet_density_cm2"]  # 2DEG 面密度
sol.subbands["psi"]              # 各束缚态波函数
```

实现要点：

- **BenDaniel-Duke 形式** `−(ħ²/2) d/dx(1/m* dφ/dx) + Ec φ = Eφ`，界面 `1/m*` 取调和
  平均，与介电常数同一套串联论证。离散后是对称三对角，用 `eigh_tridiagonal` 取最低
  若干个本征态。
- **约束态判据**：`E < E_cut`，`E_cut` 取两端接触带边较低者——能量高于它的电子可以
  逃向其中一个接触，不是束缚态。
- **2D 占据**：二维费米-狄拉克积分就是 `ln(1 + e^η)`，所以子带面密度
  `N_i = (m* kT/πħ²) ln(1 + e^{(F−E_i)/kT})` 对简并统计也精确，无需求积。
- **连续谱**：`E_cut` 之上的态仍按三维处理，权重为 `Q(3/2, (E_cut−Ec)/kT)`（正则化
  不完全 gamma），这是玻尔兹曼极限的闭式表达，保证阱外平滑过渡回经典。
- **Gummel 迭代**：泊松内层的牛顿保持量子密度固定，外层刷新子带。不加阻尼时每步约
  ×0.78，本身就是压缩映射，仅在出现反弹时才回退步长。

已知边界：`n_states` 限制了参与求和的子带数，宽阱（例如表面有重掺杂帽层时形成的
近表面势阱）需要更大的 `n_states` 才能让子带求和收敛到三维结果；沟道子带只处理 Γ
谷电子，空穴未做量子化；阱内未做交换-关联修正。

## 赝晶应变

薄于临界厚度的外延层无法弛豫，只能被衬底锁定面内晶格常数，再以面外应变补偿。
指定衬底即可打开（`substrate="GaAs"`，或给一个晶格常数；生长方向取 [001]）：

```python
dev = TwoTerminalDevice([Layer(20, "GaAs", Nd=1e17),
                         Layer(8, "GaInAs", composition=0.8),   # In0.2Ga0.8As
                         Layer(20, "GaAs", Nd=1e17)],
                        substrate="GaAs")
sol = dev.get_quantum_equilibrium()
# 带隙张开 ~70 meV，重空穴升到轻空穴之上 ~80 meV，阱中 2DEG ~6e11 cm^-2
```

- **应变状态**：`eps_in = (a_sub − a_layer)/a_layer`，
  `eps_out = −2(C12/C11)·eps_in`（自由表面 `sigma_zz = 0`）。
- **导带**：只有静水压项，`δEc = a_c·Tr(ε)`。Γ/L/X 用同一个 `a_c`。
- **价带**：Bir-Pikus 应变哈密顿量加自旋-轨道耦合。|3/2,±3/2> 块本身对角，
  |3/2,±1/2> 与 |1/2,±1/2> 构成 2×2，令

      h = a_v·Tr(ε),  d = −2b(ε⊥ − ε∥),  s = sqrt(Δ² − Δ·d + 9d²/4)

  则 `E_hh = Ev + h + d/2`，`E_lh = Ev + h − Δ/2 − d/4 + s/2`，
  `E_so = Ev + h − Δ/2 − d/4 − s/2`。无应变时严格退化为
  `E_hh = E_lh = Ev`、`E_so = Ev − Δ`；剪切部分无迹，故三者和与 `d` 无关。
- **态密度**：分裂后轻空穴按 `exp(−(E_top−E_lh)/kT)` 逐步退出价带 DOS，
  无分裂时精确回到 `(m_hh^{3/2} + m_lh^{3/2})`，不会出现跳变。
- **临界厚度**：Matthews-Blakeslee（60° 位错，`b = a/√2`，
  `ν = C12/(C11+C12)`）不动点求解。只报告，不自动弛豫——超过 h_c 的层会被
  标记 ⚠，而不是悄悄当成部分弛豫。

形变势取 Van de Walle / Chuang 的约定（`a_v > 0`，`δEg = (a_c − a_v)·Tr(ε)`），
它给出 GaAs 的 `dEg/dP = 11.0 meV/kbar`，实测 10.8。Vurgaftman 表把 `a_v` 印成
相反符号、并把能隙位移写成 `(a_c + a_v)`，两者数值一致。

已知边界：L/X 谷用同一个 `a_c`，且未做剪切分裂（需要 `Ξ_u`）；有效质量未随应变
改变（[001] 应变下重空穴质量其实是各向异性的）；未建模弛豫；[111] 生长才有压电
电荷，未建模；氮化物只有弹性常数、没有形变势，应变会计算但能带不移动，并在层表
里标记出来。

## 验证

```bash
python tests/validate_physics.py     # 解析解对照（经典）
python tests/validate_quantum.py     # 解析解对照（量子）
python tests/validate_strain.py      # 解析解对照（应变）
python tests/web_smoke.py            # Web API 端到端（需先启动服务）
```

`validate_physics.py` 对照的解析结果：内建电势、耗尽宽度与峰值电场（耗尽近似）、
AlGaAs/GaAs 的 ΔEc/ΔEg = 0.62、短基区二极管 `J = q ni²(Dn/WpNa + Dp/WnNd) e^{V/kT}`、
理想因子双区（低偏压 SRH n≈2 → 高偏压扩散 n≈1）、反向电流按 1/τ 标度、平衡态零电流。

`validate_quantum.py` 对照的解析结果：方井与三角阱的能级（三角阱用 Airy 函数零点，
并计入 Dirichlet 墙位于网格外一步带来的 `F·h` 偏移）、BenDaniel-Duke 质量阶跃井的
解析谱（超越方程数值求根）、二维占据的玻尔兹曼与简并极限、以及自洽循环的物理论证
——pn 结上量子结果必须**逐点等同**经典结果且无伪束缚态；HEMT 上面密度落在 1e12
量级、沟道质心在界面之后约 5 nm、量子沟道电荷不多于经典、基态能量低于裸三角阱的
Airy 估计。

`validate_strain.py` 对照的解析结果：无应变极限下 HH/LH 严格简并、SO 恰在
`Δ_so` 之下；剪切部分无迹（三边之和与分裂量 `d` 无关）；`Δ_so → ∞` 时 HH–LH 分裂
趋于 `d = −2b(ε⊥−ε∥)` 且误差按 `1/Δ_so` 下降（这一收敛阶也被单独断言）；静水压
系数 `dEg/dP = −(a_c−a_v)/B` 落在 5–20 meV/kbar 且 GaAs 对上实测 10.8；
In₀.₅₃Ga₀.₄₇As 与 InP 晶格匹配；临界厚度随失配单调下降。

## 一次完整试跑

```bash
python examples/comprehensive_run.py        # 约一分钟，输出与 .out 文件对照
```

依次跑平衡能带、漂移-扩散、I–V、C–V、量子修正和应变六部分，打印固定的文本块，
可与 `examples/comprehensive_run.out` 直接 diff。这是投稿包里给审稿人的那份示例。

## 许可

AGPL-3.0-or-later，与原项目一致，全文见 `LICENSE`。材料参数表版权归各自作者所有，
详见 `src/obpds/materials/data.py`。
