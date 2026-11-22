# 无 commitment 的事件触发维修主程序

## 文件
- maintenance_system_no_commitment.py：完整独立主程序，不依赖旧主程序。
- example_config_no_commitment.json：默认配置。
- test_no_commitment.py：回归测试。
- validation_no_commitment.json：本次实际测试报告。

## 运行
将 Python 文件和 JSON 配置放在同一项目目录，在该目录运行：

```bash
python -m pip install numpy pandas scipy
python maintenance_system_no_commitment.py --config example_config_no_commitment.json --out results_no_commitment
python test_no_commitment.py
```

对照模式：
```bash
python maintenance_system_no_commitment.py --config example_config_no_commitment.json --policy daily --out results_no_commitment_daily
python maintenance_system_no_commitment.py --config example_config_no_commitment.json --policy periodic --out results_no_commitment_periodic
```


```bash
主程序运行：
python python maintenance_system_no_commitment.py --config example_config_no_commitment.json --out results_no_commitment
绘制实际执行分组：
python python plot_grouping.py --results results_no_commitment --mode executed
绘制第 20 天的计划快照：
python python plot_grouping.py --results results_no_commitment --mode planned --decision-day 20
绘制单台设备压力及安全余量：
python python plot_trajectories.py --results results_no_commitment --assets PMEC_001 --show-replanning --diagnostics
```

Python >= 3.10，SciPy 需要提供 scipy.optimize.milp。
命令行读取旧 JSON 时会提示并忽略 commitment_days。不要把 commitment_days 直接传给新版 Config 构造函数。
输出目录若包含旧 commitment_log.csv 会拒绝运行，请使用新目录。其他同名新版输出会被覆盖。

## 调度逻辑
每日滤波、预测、安全检查；有任务且满足触发条件才求解。未执行任务均可调整，没有固定日期约束。保留 PROPOSED、DEFERRED、EXECUTED 三种状态。
任务根据 proposed_day 执行，不再要求先转为 COMMITTED。收到实际确认后才重置滤波周期、标记执行。缺失执行确认会停止自动运行。
无求解日仍会执行到期计划。
事件触发包含首次计划、未分配任务、风险余量、保留安排失效、容量冲突、最大间隔，以及可选参考日期变化。periodic 模式同样在新任务或保留安排失效时提前求解，不是完全固定周期。
风险判据为 M=预测下界-p_plan <= risk_trigger_margin。默认 p_safe=3.0、p_plan=3.05、margin=0、最大间隔7天，这些是实验设置，不是校准后的安全推荐。
准备时间仍从任务创建日开始计算，不是从最后一次改期开始计算。
未执行任务若退出有限窗口，撤回临时安排并转为 DEFERRED；保留任务ID及准备日期，之后可重新激活。
无可行日期、当前物理下限概率检查失败、无有效求解结果等仍会停止，不会自动放松B0。
没有实现实际资源预订、人员通知、紧急维修或现场停机程序。停止仿真不等于现场安全处置。

## 模型与参考日期
保留 q_alpha；其预测方差贡献为 h(h-1)(2h-1)/6*q_alpha。q_alpha>0 表示估计器允许周期内斜率随机游走，仿真真实斜率仍为每周期固定。设置 q_alpha=0 可恢复周期内固定未知斜率模型。
首次初始化使用首个观测一次；GR后按 reset_pressure 及 reset_pressure_var 重置，忽略GR日观测。模拟器重置压力为精确 reset_pressure，即使滤波器 reset_pressure_var 非零也不改变模拟器重置分布。
概率筛选针对当前周期、候选GR时刻的真实压力，不是未来传感器读数，也不是全路径联合安全保证。
nominal_at_execution 保存实际执行前最后一次每日任务更新时的参考日期（通常为前一天更新），不是承诺时参考日期，也不是执行后重新计算的参考日期。
执行成本字段为 executed_shift_penalty_at_execution_reference。比较旧版冻结参考成本时须注意口径不同。

## 末日处理
末日 ingest 后检查当前物理安全并记录已有未来安排的余量；不创建任务、不撤回计划、不重求解、不发出未来动作。completed 表示给定仿真区间执行完毕，不表示期末保留计划均已执行或保证在区间外继续可行。终端未来余量仅记录，不自动求解处理。

## 输出与绘图兼容
空CSV也有列名。
保留 observations、simulation_truth、executions、state_estimates、forecasts、plans、tasks、solver_log、schedule_changes、exceptions、plan_monitoring、replanning_log。
删除 commitment_log 及所有承诺字段。plan_monitoring 新增 asset 列。
上一轮 plot_trajectories.py 可继续读取这些输出；诊断图只有 PROPOSED，没有已承诺点。
上一轮 plot_grouping.py 的 planned 模式可用，executed 模式需要用 nominal_at_execution 替换 nominal_at_commitment，并删除对 committed_day 的必需列检查。committed 模式不再适用。本包没有包含重新生成的绘图脚本。
不能仅用某次有限区间 recorded cost 比较策略优劣：已执行数量及期末剩余任务可能不同。

## 测试
测试含协方差闭式公式与30步递推比较、初始化和GR重置、临近日期可移动、旧日期风险失效后重规划、无求解日执行、缺失确认停止、无可行日期停止及完整策略运行。
默认种子42的event/daily/periodic均完成；额外种子7、21、100的event运行完成。详见测试报告。未验证真实业务数据或概率校准。
