# Chapter 6《STOCHASTIC PROBLEM EXPLORATION》模型与实验设计详解

> 用途：这份文件不是论文正文，而是给作者/答辩/审稿准备用的"设计说明书"。它把 Chapter 6 的
> 每一个论文化陈述、每一张表/图、每一段结果解释，对应回 `og/` 里的代码与 `results/` 里的
> 已提交产物。**论文里出现的每一个数字都来自这些产物**（`results/*/summary.csv`、
> `rows.csv`、`results/tables/prose_values.txt`），没有手填值。
>
> 复现命令与产物清单见 §9；仓库根的 `README_ch6.md` 是代码侧的英文说明，本文是设计侧的中文说明。
>
> 论文侧文件：`../thesis-overleaf/Thesis/chapters/STOCHASTIC PROBLEM EXPLORATION.tex`；
> 论文产物：`../thesis-overleaf/Thesis/tables/tab_ch6_*.tex`、`Thesis/figures/fig_ch6_*.pdf`。

---

## 0. 章节 / 代码 / 产物对照

| 论文位置 | 代码 | 产物 |
|---|---|---|
| §6.3 预测模型（`sec:unc-estimation`） | `filter.py`（局部线性趋势 Kalman + 事件三通道）、`process.py`（`risk_deadline_lead` 等） | — |
| §6.4 风险可容日、触发、稳定性（`sec:unc-window`, `sec:unc-replanning`） | `controller.py`（`OnlineRunner`, `Strategy`, `PendingTask`） | — |
| §6.5 场景与信息通道（`subsec:unc-instances`） | `config.py`（`ScenarioConfig`/`PolicyConfig`）、`scenarios.py`（`World`，真值过程 + 观测信道） | `results/main/config.json` |
| §6.5 策略（`subsec:unc-policies`） | `policies.py`（9 个 `Strategy`） | — |
| §6.5 指标（`subsec:unc-measures`） | `metrics.py` | `rows.csv` 列 |
| §6.6 结果表（`tab:ch6-results`, `tab:ch6-paired`, `tab:ch6-calibration*`, `tab:ch6-sensitivity`） | `experiment.py`（跑矩阵 + 配对 Wilcoxon）、`tex_table.py`（由 CSV 生成 LaTeX） | `results/main/`, `results/sweep_*/`, `results/tables/tab_ch6_*.tex` |
| 图 `fig:unc-policies/calibration/sensitivity/timeline` | `figures.py` | `results/figures/fig_ch6_*.pdf` |
| Chapter 5 示例算例（`fig:grouping-result`） | `ch5.py` | `results/ch5/ch5_illustrative.json` |

---

## 1. 这一章要回答什么（四个问题 → 哪个实验回答）

Chapter 5 的分组 MILP 假设"已知每个 PMEC 的退化速率 $\delta_i$ 与当前压力"，于是可容日集合
$\mathcal{F}_n^{\mathcal{W}}$ 是确定的。Chapter 6 把这条假设撤掉：速率只能从带噪、缺失、
偶发粗差的逐日压力记录里**估计**，因此所有日期都是**预测**的，并且新记录到达时必须**改计划**。

| 论文里的问题 | 由哪个对比回答 | 由哪个指标度量 |
|---|---|---|
| Q1 预测窗口是否"校准"（风险容差是否名副其实） | S5 自身的预测 vs 仿真真值 | PICP（按 lead time 分层）、平均预测标准差（sharpness）、参考日误差 MAE |
| Q2 数据不确定下分组价值还剩多少 | S1 vs S3 vs O1（与 S2） | 总成本 $C^{\mathrm{tot}}$、出勤次数 $n_b$、$n/n_b$（consolidation ratio）、regret vs O1 |
| Q3 哪条重规划规则买到安全、代价是多少（重算 + 计划抖动） | S3 → S4 → S5 → S6 的消融阶梯 + $\varepsilon$/承诺消融扫描 | 成本、$A^{\mathrm{alarm}}/A^{\mathrm{floor}}$、$N^{\mathrm{slv}}$、$N^{\mathrm{chg}}$ |
| Q4 一次性算完从不修改会怎样（= Chapter 5 的开环使用） | S7 vs 在线策略；O2 vs O1 | 成本构成（几乎全是 shift penalty）、$A^{\mathrm{alarm}}$、$n$（任务数本身就错） |

**关键点**：四个问题分别对应"估计层质量 / 优化的信息价值 / 机制设计的价值 / 滚动重规划的必要性"。
它们不是同一件事，所以用了不同的对照，而不是只报一个"我们的方法更省"。

---

## 2. 模型设计：四层，每层只改一件事

设计上刻意做到：**物理层与优化层完全沿用 Chapter 4/5**，不确定性只从"信息层"进入。
这样 Chapter 6 的结果才能被解释为"把同一个模型搬到不完备信息下会发生什么"，而不是"换了一个模型"。

### 2.1 物理与决策层（继承，不改）
`ScenarioConfig` 与 Chapter 4/5 完全一致：
$T=180$ d、$|\mathcal{I}|=30$ 个 PMEC、$\bar p=3.5$、$p^{\mathrm{ref}}=3.2$、$\underline p=3.0$ bar、
$L^{\mathrm{win}}=10$、$\Delta^{\mathrm{adv}}=3$、$\Delta^{\mathrm{safe}}=2$、$K^{\mathrm{clus}}=10$、
$c^{\mathrm{set}}=100$、$\eta^-=5$、$\eta^+=10$、$\ell=1$ d。
真值过程：周期内恒定漏率、GR 把压力恢复到 $\bar p$（`scenarios.py` 的 `World.alpha`）。

### 2.2 信息层（Chapter 6 新增的唯一"随机性来源"）
`PolicyConfig`，参数取自真实监测数据的标定值（Chapter 3 §3.5 的 `tab:parameter-practice`）：

| 机制 | 参数（默认） | 来源/理由 |
|---|---|---|
| 测量噪声 | $\sigma_y=0.008$ bar | 真实逐日拟合残差 0.007–0.011 bar |
| 上报分辨率 | 量化 0.01 bar | 传感器密度分辨率 |
| 缺测 | 逐日缺失率 5%，连续中断平均 3 d | 真实记录有孤立缺失和成段中断 |
| 突变漏况 | `shock_rate`（默认 0，扫描时 0.005/PMEC·d，倍率 1–4×） | 真实"漏况改变"事件 |
| 粗差 | `outlier_rate`（默认 0） | 真实数据里存在 >0.2 bar 的阶跃 |

**没有把速率本身做成随机游走作为真值**：真值在每个周期内恒定（`q_alpha` 只是滤波器的过程噪声），
所以滤波器的保守性才能被 Q1 的 PICP 检验暴露出来（见 §6.5 结果）。

### 2.3 估计层：局部线性趋势 Kalman（§6.3，本轮已补全推导）
状态 $\bm x_k=(p_k,\delta_k)^\top$，$\bm F=\begin{pmatrix}1&-1\\0&1\end{pmatrix}$，
$\bm Q=\mathrm{diag}(q_p,q_\delta)$（Assumption 6.3.1：零均值、时间白、同刻两分量不相关），
$\bm h=(1,0)$ 只观测压力。

- **递推**：预测（含分量式协方差剪切 `eq:unc-kf-predict-cov`）→ innovation
  $\nu_k,S_k,z_k$（`eq:unc-innovation`）→ 增益显式 $\bm K_k=(P^{pp},P^{p\delta})^\top/S_k$
  （`eq:unc-kf-gain`）→ 状态更新 → **协方差更新** `eq:unc-kf-cov` 及其 rank-one 形式
  `eq:unc-kf-cov-comp`（说明为什么一条记录定不了速率）。
- **事件三通道**（代码 `filter.py`，与 `gate_z=3`、`outlier_streak=2`、`jump_z=6`、`jump_bar=0.10`
  一一对应）：level jump → 周期重置（速率回到先验、水位不确定度 `reset_sd=0.01`）；
  同号连续 out-of-band → **协方差膨胀** `eq:unc-inflation`（模型适应，而不是丢弃数据）；
  孤立 out-of-band → **Huber 型测量方差膨胀** `eq:unc-huber`。
  自己的 GR 是**已知过程输入**，直接注入滤波器，且 `cooldown_days=3` 内不重复开任务。
- **多步预测**（`prop:unc-pred`）：
  $\mu_i(h)=\widehat p_{i|d}-h\widehat\delta_{i|d}$，
  $\sigma_i^2(h)=P^{pp}_d-2hP^{p\delta}_d+h^2P^{\delta\delta}_d+hq_p+\frac{h(h-1)(2h-1)}6q_\delta$。
  阶数结论：水平噪声 $O(h)$、残余速率误差 $O(h^2)$、速率扩散 $O(h^3)$ —— 这是后面所有
  "长视界不可信 → 窗口晚进入、事件触发重规划"的设计依据。
- **缺测**：用 $\bm F^g$ 与 $\sum_{r<g}\bm F^r\bm Q\bm F^{r\top}$ 跨空洞推进，方差按缺测长度增长。

### 2.4 决策层：把预测分布变成 MILP 能用的日期集合
1. 参考日改为预测交叉：`eq:unc-ref-estimate`。
2. 压力地板的 deadline 改为机会约束：`eq:unc-chance` → `eq:unc-risk-deadline`；
   再与操作上限 $\Delta^{\mathrm{delay}}$ 取紧：`eq:unc-deadline`；集合 `eq:unc-feasible`。
   由于正态，机会约束等价于把均值下移 $z_{1-\varepsilon}\sigma_i(h)$，**不需要轨迹模拟**。
3. **窗口进入测试** `eq:unc-entry`：不看"平均交叉日是否在窗口内"，而看"窗口末端的
   风险调整压力是否已低于 $p^{\mathrm{ref}}$"。原因正是 $h^2P^{\delta\delta}$ 项：刚 refill 的
    compart ment 速率没定下来，平均交叉日期误差可达数周，若按均值进入会把任务提前几个月塞进窗口。
4. **事件触发重规划** `eq:unc-trigger-open`…`eq:unc-trigger-periodic`：新任务开 / deadline 已过 /
   剩余余量 ≤ $m^{\mathrm{margin}}$ / 参考日漂移 ≥ $m^{\mathrm{drift}}$（相对上次求解值，度量"新信息"
   而非估计抖动）/ 周期兜底 7 d。到期未指派走**紧急通道**并记 $C^{\mathrm{emg}}=$`emergency_extra=200`，
   把不可行性显式记进结果而不是丢任务。
5. **承诺机制**：软锚定 `eq:unc-stability`（$\eta^{\mathrm{stab}}=2$/天）+ 硬锁 `eq:unc-lock`
   （$L^{\mathrm{lock}}=2$ d）。若新信息使已宣布日不可容，**释放承诺并计一次 plan change**，
   不让模型不可行。
6. **退化性质**（`prop:unc-limit`）：$\bm P_d=0$、$q_p=q_\delta=0$、$\varepsilon=1/2$ 时
   $\mathcal F_n(\varepsilon)$ 退回 Chapter 5 的 $\mathcal F_n^{\mathcal W}$。这句话是整章的
   "贡献定位"：Chapter 5 是本模型的零不确定极限，不是另一个模型。

---

## 3. 实验设计（重点）

### 3.1 为什么用合成实例，而数据只用来定范围
Chapter 3 §3.4–3.6 已论证：真实记录是稀疏、带噪、偶发自相矛盾的，且没有任何"被测得的漏率"；
而真实舰队平均漏率比可决策的速率慢 1–2 个数量级（7 年一次补气的平均下，15 天窗口里没有任何
值得优化的决策）。所以：**实例是合成的，参数阶次来自真实数据**；`config.json` 把每个数值写死，
可完全复现。这一点在论文里由 Chapter 3 `sec:industrial_parameters` + Chapter 6 的
`subsec:unc-instances` 两处一致声明。

### 3.2 三个实验轴

**轴一：场景变体（沿用 Chapter 5 的两条轴，交叉）**

| 变体 | 速率范围 (bar/d) | 时间形态 | 每 PMEC 周期数 | 设计意图 |
|---|---|---|---|---|
| A | 0.001–0.003 | 均匀 | ≈1 | 与真实监测数据标定一致（信息轴的主战场） |
| B | 0.001–0.003 | 3 个日期簇（jitter 3 d） | ≈1 | 低密度 + 可分组 |
| C | 0.004–0.008 | 均匀 | 2–4 | 密集 + 不可分组（分组上限的对照） |
| D | 0.004–0.008 | 3 个日期簇 | 2–4 | 密集 + 可分组（分组价值最大处，操作轴主战场） |

选 A/B/C/D 而不是更多：A/C 是"均匀"对照、B/D 是"集中"对照；低/高速率分别对应"真实观测阶次"
与"Chapter 5 敏感性设定"。论文里 C/D 明确写了**不是**用实测斜率标定，避免过度声称。

**轴二：信息质量**（只在扫描里改，见 §6）—— $\sigma_y$、缺测率、粗差/冲击、clean（完全干净）。

**轴三：策略**（见 §3.4）。

矩阵规模：4 变体 × 30 seeds × 9 策略 = **1080 次运行**（主比较）；扫描 20 组 × 20 seeds；
每次运行 = 一整条 180 天的在线仿真。

### 3.3 配对（paired）设计：这是统计上最重要的一条
`scenarios.py` 不用"顺序 RNG"，而是用 **坐标键控**的确定性哈希：
`draw(seed, 'shock', day, pmec)`、`draw(seed, pmec, cycle)` …

含义：**同一个 seed 下的世界（真实漏率、初始压力、簇中心、冲击时刻与倍率、噪声实现、
缺测模式、粗差位置）与"被测策略"完全无关**。因此：

- 同一 seed 上，两个策略面对的是**同一条物理轨迹与同一份观测序列**；
- 每 seed 的成本差是**精确配对**的差（差为 0 表示两策略在该 seed 上产生**同一个计划**，
  而不是"数值碰巧接近"）；
- 于是可用 **Wilcoxon signed-rank**（`metrics.wilcoxon_paired`）对 per-seed 差做检验，
  并同时报告 **median/mean difference** 与 **share_worse / share_better**（多少比例的 seed 变差/变好）。

报告口径因此统一为"**per-seed 中位数差 + Wilcoxon p**"（`tab:ch6-paired`），而不是简单的
样本均值比较。这样做的原因：单 seed 内的分布很偏（S7 的 shift penalty 有长尾），
均值差容易被少数极端 seed 主导；配对 + 中位数 + 非参数检验能抵抗这一点。

### 3.4 策略阶梯 = 消融设计（每个策略只改一件事）

| 策略 | 分组 | 重规划 | 窗口 | 承诺 | 信息 |
|---|---|---|---|---|---|
| S1 | 无（一任务一出勤） | 每日 | 风险 | — | 滤波 |
| S2 | Chapter 5 M2 | 周期 7 d | 标称(Ch.5) | 无 | 滤波 |
| **S3** | Chapter 5 M2 | **事件触发** | **标称(Ch.5)** | 无 | 滤波 |
| S4 | Chapter 5 M2 | 事件触发 | 风险 $\mathcal F_n(\varepsilon)$ | 无 | 滤波 |
| **S5** | Chapter 5 M2 | 事件触发 | 风险 | **锚定+锁** | 滤波 |
| S6 | Chapter 5 M2 | 每日 | 风险 | 锚定+锁 | 滤波 |
| S7 | Chapter 5 M2 | **从不重规划**（day-0 一次性开环） | 标称 | 无 | day-0 估计 |
| O2 | Chapter 5 M2 | 从不重规划 | 风险 | 无 | **全知（含冲击表）** |
| O1 | Chapter 5 M2 | 事件触发 | 风险 | 锚定+锁 | **全知** |

**差 = 某个设计要素的价值**（这就是"为什么要这样比"的答案）：

| 两个策略之差 | 隔离出的东西 | 论文中的实测 |
|---|---|---|
| S3 − S1 | **分组本身**的价值（相对厂里"报警就单独跑一趟"） | A 省 46%、D 省 72%；consolidation 2.22 / 4.61 vs 1.00 |
| O1 − S3 | **数据不完美的价格**（information value / regret） | 326（A）、314（D）≈11–18% |
| S4 − S3 | **机会约束窗口**这一项的价值 | 中位差 0（p 0.33–0.94），暴露 0.0 天 —— 与"预测偏保守 + $\Delta^{\mathrm{delay}}$ 才是紧约束"一致 |
| S5 − S4 | **承诺机制**的价值 | 改期次数 −32/−35/−82/−88（p<0.001，降 29–31%）；D 上约省 90 次改期、成本影响 +1.5% |
| S6 − S5 | **重规划频率**（有信息 vs 无信息也重算） | A 上更贵 +142、求解次数 145.8 vs 69.4；D 上 −18、157.2 vs 101.8 → 触发逻辑用约 1/3 的求解次数达到同等 churn |
| S3 − S7 | **必须在线重规划**（Chapter 5 的开环使用） | S7 consolidation 反而更高（3.18），但 $A^{\mathrm{alarm}}$ 77.5（A）/2353.2（D）、最低压力 2.532 < 地板，成本 13695–19966（5–7 倍），且任务数本身就错（A 做 52.7 次其中只需 36.1；D 只做 37.8 而需 96.8） |
| O2 − O1 | **滚动视野 vs 一次性完美计划**（把"信息"和"滚动"两个代价分开） | 稀疏变体 ≈0；密集变体 +138（C）、+407（D），暴露也更高 → 密集场景下滚动重优化本身更优 |

这个阶梯是**为什么这样比较**的核心答案：
1. 如果只比 S5 vs S1，会把"分组的功劳"错记到"预测/随机建模的功劳"上；
2. 如果只比 S5 vs O1，会把"数据质量的代价"错记到"算法好坏"上；
3. 只有 S3→S4→S5 这种"单要素差分"，才能说清 Chapter 6 新增的每个机制各自值多少、
   以及在什么条件下不值（S4 的 0 差就是一个真实且有用的负面结果）。

### 3.5 指标体系（为什么是这些）

| 类别 | 指标 | 为什么 |
|---|---|---|
| 成本 | $C^{\mathrm{set}}$、$\Phi^{\mathrm{pen}}$、$C^{\mathrm{emg}}$、$C^{\mathrm{tot}}$ | 与 Chapter 5 完全同定义，才能跨章比较 |
| 合并度 | $n$、$n_b$、$n/n_b$ | 分组研究的本质指标；成本会受 shift penalty 干扰 |
| 安全 | $A^{\mathrm{alarm}}$、$A^{\mathrm{floor}}$、$\min p$ | **成本模型不给压力违规定价**（它给时间偏移定价），所以安全必须单列，否则"更省"可能意味着"更危险" |
| 适应性 | $N^{\mathrm{slv}}$、$N^{\mathrm{chg}}$、未响应的报警数、求解耗时 | 在线方法有额外运行成本与计划抖动，这两项是我们的方法的"代价"，必须自报 |
| 预测质量 | PICP($\varepsilon$) 按 lead 分层、mean sd（sharpness）、$|\widehat\tau_n^0-\tau_n^{0,\mathrm{real}}|$ | Q1 的直接度量；只有分布校准，机会约束才有意义 |

三条**防自欺的口径约定**（都是有意设计的，答辩时值得主动讲）：

1. **shift penalty 在结果里按"真值报警日"评估**（`reference_real`），而不是按产生该计划的
   预测报警日。否则预测误差会被自己的估计抵消，任何策略都"零偏移"。优化目标内部仍用预测值。
2. **PICP 只统计不含计划 GR 的预测窗口**（`metrics.calibration_metrics` 里 `clean` 过滤）：
   跨补气点的压力阶跃是结构性断点，不是预测失败，混进去会让覆盖率数字失去意义。
3. **紧急通道显式记账**（$C^{\mathrm{emg}}$、`fallback_log` 按 reason 计数），
   所以"策略靠漏掉任务变便宜"不可能发生。

### 3.6 样本量、求解与可复现
- 主比较 30 seeds；扫描 20 seeds；每 seed 环境相同 → 配对有效。
- 求解器：窗口规模在许可证范围内用 Gurobi，其余用 `scipy.optimize.milp` 分支定界；
  同一相对间隙 $10^{-4}$、同一每窗口时限；两者只影响求解时间不影响报告成本
  （可复现性重跑中，69 个指标列里只有求解耗时两列不同）。
- 表/图全部由 CSV 生成（`tex_table.py`、`figures.py`），正文数字取自 `prose_values.txt`。

---

## 4. "与确定性模型对比"到底对比的是什么

这一点容易被误读，明确三条：

1. **确定性方法 = S3**，即 Chapter 5 的 M2 分组模型、同样的 $L^{\mathrm{win}}$、同样的成本系数、
   同样事件触发，唯一区别是它的可容日用**标称（均值）窗口**而不是机会约束窗口。
   所以 S5 − S3 的差**不是**"优化器更强"，也**不是**"重规划更多"，
   而是"风险调整的可容日集合 + 承诺机制"这两项的净效果。
2. **确定性方法的另一种使用方式 = S7**：Chapter 5 的模型在 day-0 用一份估计算一次完整计划、
   之后不改。它用来反驳"反正我们有 MILP，算一次就好"的读法：合并度看起来更好，
   但安全与总成本崩坏（5–7 倍成本、最低压力低于地板、任务集合本身错）。
3. **对比的口径是同一环境、同一成本函数、同一配对 seed**，因此论文里敢于报告负面结果：
   S5 相对 S3 在 A/B **更贵** 92/110（p<0.001）、C 上 72（p=0.058）、D 上 −15（p=0.805）；
   A 上暴露改善 5.5 天（p<0.001）但 C 上恶化 5.0 天（p=0.026）——机制是密集均匀场景下
   硬锁会挡住紧急任务前移。论文据此把结论写成"**把几个百分点的成本（个别场景下几天暴露）
   换成显著更稳的计划**"，而不是"全面更优"。

这正是要突出的研究重要性：**分组的价值在预测输入下大部分仍在**（S3 已达 S1 的 46–72% 成本降幅、
距全知参考只差 11–18%），但它**不会自动保住**（S7），而且**哪些机制值得加**必须逐项量化
（S4 ≈ 0；S5 换稳定；S6 多花求解次数换不到一致性）。

---

## 5. 敏感性分析是怎么做的

`run_experiments.sh` 里 **20 个 sweep，每个只改一个 knob**，其余与主表完全相同：

| 类别 | sweep | 变体 | 改动（默认 → 值） |
|---|---|---|---|
| 信息 | `noise_A` / `noise_C` | A / C | $\sigma_y$ 0.008 → 0.020（3×） |
| 信息 | `clean_A` | A | 无噪声 + 无缺测 + 无量化（"完美记录"） |
| 信息 | `missing_A` | A | 缺测率 0.05 → 0.20 |
| 信息 | `shock_C` / `shock_D` | C / D | `shock_rate` 0 → 0.005 |
| 风险容差 | `eps01_A` / `eps10_A` / `eps25_A` | A | $\varepsilon$ 0.05 → 0.01 / 0.10 / 0.25（0.05 即主表） |
| 操作 | `delay3_D` / `delay10_D` / `delay20_D` | D | $\Delta^{\mathrm{delay}}$ 5 → 3 / 10 / 20 |
| 操作 | `adv6_D` | D | $\Delta^{\mathrm{adv}}$ 3 → 6 |
| 操作 | `win20_D` | D | $L^{\mathrm{win}}$ 10 → 20 |
| 操作 | `cap3_D` / `cap5_D` | D | $K^{\mathrm{clus}}$ 10 → 3 / 5 |
| 机制消融 | `nolock_S5` | D | $L^{\mathrm{lock}}$ 2→0 且 $\eta^{\mathrm{stab}}$ 2→0（拆掉承诺） |
| 实例形态 | `jitter8_D` / `jitter15_D` | D | 簇内散布 3 → 8 / 15 d |
| 代价偏好 | `eta20_D` | D | $\eta^+$ 10 → 20 |

**报告方式**（`tex_table.paired_delta`）：对每个 sweep 的每一格，取该 sweep 与主表
**同一变体、同一策略**的 per-seed 差，报告 **20 seeds 的中位差**（`tab:ch6-sensitivity`），
另配 `fig:unc-sensitivity` 对 S3/S4/S5 画成本随 knob 的曲线。

**两个方法论细节**：
1. **变体选择有原则**：涉及"分组/容量/操作约束"的轴放在 **D**（密集+集中，分组价值最大处），
   纯"信息质量"的轴放在 **A/C**（避免把实例形态的效应混进信息效应）。
2. **"未被某机制触及的格子应恰好为 0"** 被当作比较噪声的标定：例如 $\varepsilon$ 轴对不用风险窗口的
   S3/S2 应给 0，`nolock_S5` 只对带承诺的策略非零。这些 0 出现在表里不是冗余，
   而是证明"差异只来自被改的那一项"。

**敏感性给出的核心结论（也是研究定位）**：
- 信息轴是**真实但有界**的：$\sigma_y\times3$ 代价 228–340、缺测 20% 代价 262–352、
  突发漏 415–585 并增加最多 30 天暴露；完全干净记录省下 255。
  这些量级与 O1−S3 的 regret（282–388）**同阶** → 说明 regret 确实来自记录质量而非优化质量。
- 操作轴更大：$K^{\mathrm{clus}}$ 10→3 代价 1340–1350（>最差信息轴的 4 倍），却**减少** 16–23 天暴露；
  $\Delta^{\mathrm{adv}}$ 3→6 省 200–230 并降暴露；$\Delta^{\mathrm{delay}}$ 5→3 花 65–180 换 12 天暴露；
  $\eta^+=20$ 花 418–562 换 26 天暴露；簇散布增大侵蚀分组收益 38–125。
- $\varepsilon$ 从 0.01 到 0.25 对 S3/S5 成本 **每 seed 恰好 0**：因为预测保守（§6.6.5）+
  操作上限 $\tau_n^0+\Delta^{\mathrm{delay}}$ 通常比 $\overline t_n(\varepsilon)$ 更紧。
  → **机会约束在当前标定下是"安全网"而非"成本杠杆"**；这也解释了 S4−S3 为何为 0。

结论被写成："在本模型的当前标定下，随机/预测机制不是主导杠杆；主导杠杆是现场容量、
允许的提前期，以及延迟的定价与许可。预测层在'边界宽松 + 记录差'的区域起作用"——
这比"我们的方法最省"更站得住，也直接指出下一步（把 $\Delta^{\mathrm{delay}}$、容量、
consignation 代价纳入模型）。

---

## 6. 结果如何进入论文（无手填值链条）

```
run_experiments.sh            → results/main/, results/sweep_*/（rows.csv, summary.csv, config.json）
python -m online_grouping.figures   --results results --out results/figures
python -m online_grouping.tex_table --results results/main --out results/tables
./sync_thesis_assets.sh       → Thesis/figures/fig_ch6_*.pdf, Thesis/tables/tab_ch6_*.tex, ch6_summary_main.csv
```
- 正文里被引用的数字由 `tex_table.py` 打印到 `results/tables/prose_values.txt`，
  写作时从那里取；`tab:ch6-*` 全部 `\input` 生成文件。
- 图：`fig_ch6_policies`（per-seed 分布：box/IQR/1.5×IQR whisker）、
  `fig_ch6_calibration`（PICP vs 名义容差，按 lead 分层）、`fig_ch6_sensitivity`、
  `fig_ch6_timeline`（D 变体 seed 3、S5 的 4 个 PMEC 逐日 trace，`results/trace.csv`）。
- 可复现性已验证：整套重跑到 `results_repro/` 后与已提交产物比较，
  主表 69 列中只有求解耗时两列不同；20 个 sweep 非计时列零差异；38 张生成表逐字节一致；
  4 张图与 trace 一致。详见 `og/README_ch6.md` 的 "Reproducibility check"。

---

## 7. 这套实验设计对"研究重要性"的作用（可直接用于答辩表述）

1. **把"信息不完备"变成一个可定价的对象**：O1−S3 给出记录质量的货币价值（11–18%），
   O2−O1 给出"滚动视野"的价值，二者不混。这是纯确定性 Chapter 5 无法给出的陈述。
2. **证明 Chapter 5 的结论在不确定下大部分存活**：consolidation 2.22–4.61，
   成本比反应式低 46–72% → 确定性建模的工作不是白做。
3. **证明 Chapter 5 的使用方式必须改**：S7（一次性开环）在密集变体里 43% 的资产日低于报警、
   最低压力跌破地板 → "算一次就好"是错的用法。
4. **逐项给机制定价，包括不值的那些**：风险窗口 0（并解释为什么 0）、承诺 −30% 改期换 1.5% 成本、
   每日重算换不到一致性 → 展示机制设计是**有取舍的工程设计**，不是"越复杂越好"。
5. **诚实边界**：S5 不全面优于 S3；$\varepsilon$ 当前不敏感；C/D 的速率不是实测标定；
   consignation、升级类工作、维修有效性不在模型内（`paragraph{Scope and limits}`）。

---

## 8. 不能从这套实验得出的结论（防止过度声称）

- 不能声称"随机模型比确定性分组更省"：主比较里 S5 与 S3 差异在 ±5% 内，方向随变体变化。
- 不能声称"机会约束改善了安全"：$A^{\mathrm{floor}}$ 所有在线策略都是 0，改善由
  $\Delta^{\mathrm{delay}}$ 驱动，不是由 $\varepsilon$ 驱动。
- 不能把 20 seeds 扫描里的中位差当作点估计的精确值：扫描只有 20 seeds，且不同 sweep 换用了变体；
  它们用于**排序各轴的量级**，不是用于给出弹性系数。
- 不能把 PICP 的保守性当作"滤波器不必要"：它说明的是 $q_\delta$ 与 $P^{\delta\delta}$ 的标定
  相对真值（周期内恒定速率）偏保守，因此 $\varepsilon$ 的实际风险低于名义值；
  若真值本身有漏况漂移（`shock_rate`>0 的扫描），保守性正是安全余量的来源。

---

## 9. 一键复现

```bash
cd og
./run_experiments.sh > results/suite.log 2>&1          # 1080 runs + 20 sweeps + trace，约 40–60 min
python3 -m online_grouping.figures   --results results --out results/figures
python3 -m online_grouping.tex_table --results results/main --out results/tables
./sync_thesis_assets.sh                                # 同步到 thesis-overleaf
# 可选：整套重跑到 results_repro/ 做逐位一致性检查
bash repro_check.sh > results_repro_run.log 2>&1
# Chapter 5 示例算例与图
python3 -m online_grouping.ch5 --figure
```
