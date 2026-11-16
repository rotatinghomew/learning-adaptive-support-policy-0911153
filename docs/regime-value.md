# regime_value：随状态调整的支持规则的策略价值评估

`regime_value` 与现有 `path_inventory` 并存。它在**同一初始学员群体**（roster 全体）
上评估一套会随学习状态调整的支持规则，并与现行流程对照。估计对象为全部初始学员在
固定第八周的 `score`；退出支持不等于退出目标人群。

## 方法要点

- **估计量**：序贯逆概率加权（IPW-Horvitz-Thompson 与 Hajek）与向后递归 Q 回归的
  增广 IPW（AIPW，结果中的主估计）。行为倾向直接采用事件记录中的
  `action_probabilities`，不做倾向模型拟合；Q 回归只用决策时可见特征（线性最小二乘）。
- **无未来泄漏**：规则条件只能引用决策时可见字段（见下），引用其他字段直接被拒绝；
  估计可以使用结局，决定不能。
- **终态分类**：`completed`/`withdrawn`/`lost` 与“未到下一阶段”（`ongoing_incomplete`）、
  从未决策（`never_decided`）分别计数；失访学员的固定期结局按缺测处理，不当作完成，
  也不以零或过程测量（rewards）顶替。
- **缺测结局**：报告可识别分量 `E[Y·1(可观测)]`、可观测比例、可观测者条件均值，
  以及用分数范围构成的全人群均值**边界**；不给缺测者编造点值。
- **支持不足**：逐阶段逐规则核对所开动作的历史倾向。支持为零或低于阈值的路径写入
  `identification.applicability_boundary`（含示例状态）；全部路径无支持时
  `regime_estimate.status = "not_identified"`，只给原因与适用边界，不硬算收益。
- **不确定度**：对学员配对的百分位自助法，种子与次数可复查（默认 reps=500、
  seed=20261116）。

## options

| 字段 | 默认 | 含义 |
| --- | --- | --- |
| `policy` | 各阶段均 `phone` 的基线情景 | 拟议规则，见下 |
| `support_threshold` | 0.05 | 历史倾向低于此值计入适用边界 |
| `score_range` | 观测分数范围 | 缺测结局边界所用分数范围 `[low, high]` |
| `bootstrap_reps` | 500 | 自助重抽样次数，0 表示跳过 |
| `bootstrap_seed` | 20261116 | 自助法随机种子 |

未列出的键会被拒绝（422），避免未声明的假设藏进默认值。

`policy` 形如：

```json
{"name": "phone-then-peer",
 "rules": [
   {"stage": 0, "action": "phone", "note": "首次一律电话提醒"},
   {"stage": 1, "when": {"need": {"gte": 2}}, "action": "peer"},
   {"stage": 1, "action": "phone"},
   {"stage": 2, "action": "none"}
 ]}
```

- 每个阶段按给定顺序取第一条命中的规则；每个阶段必须恰好有一条无 `when`
  的兜底规则，否则 422。
- `when` 的字段只能是 `need`、`prior_attendance`、`initial_need`、`site`、
  `previous_action`；算子为 `eq`/`ne`/`gte`/`lte`/`in`。`previous_action`
  只支持 `eq`/`ne`/`in`，取值为已知动作名。
- `note` 为业务备注，随结果保存，供复核为何建议继续或更换支持。

## 结果结构（result）

- `estimand`：目标人群、估计对象与排除说明。
- `policy`：实际评估的规则（含来源 default/caller），原样保存。
- `data_used`：初始人数、进入/排除人数及原因、终态分类、结局可观测计数。
- `identification`：识别状态（`identified` / `partially_identified` /
  `not_identified`）、逐阶段逐规则的支持度表与适用边界。
- `regime_estimate`：分量、可观测比例、条件均值、边界、各估计量对照、
  权重诊断（最大权重、有效样本量）；不可识别/不可估计时给原因不给数值。
- `current_process`：现行流程的描述性对照（同一初始群体的观测汇总，非反事实）。
- `comparison`：规则与现行流程的差值。
- `uncertainty`：自助区间、重抽样次数与种子。
- `process_measures`：过程测量的描述性汇总（不参与估值，缺失不以零代替）。
- `assumptions`、`diagnostics`：声明的假设与诊断标记。
- `rule_audit`：逐学员的处方、实际动作、一致性、权重与结局可观测性，
  供业务复核每一次继续或更换支持的依据。
- `interpretation`：面向业务的结论摘要。

## 运行

```sh
python -m app.demo --analysis regime_value
python -m unittest tests.feature_acceptance -v
```

正式展示从空临时结果目录启动两个真实服务、创建分析并回读保存结果，通常在数秒内
结束，不下载资料、模型或依赖。
