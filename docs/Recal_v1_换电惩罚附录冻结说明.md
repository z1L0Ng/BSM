# Recal_v1：换电惩罚附录冻结说明

更新时间：2026-03-19

## 1. 主文目标口径（启用）

Recal_v1 主结果采用论文 Eq.(10) 口径：

\[
\max J = J_{service} + \beta \cdot J_{idle}
\]

其中：
- 不使用低电量 bonus（`reposition_low_energy_swap_bonus=0.0`）
- 不使用时间折扣（`reposition_service_discount_gamma=1.0`）

## 2. 附录扩展设计（冻结，不在本轮主跑启用）

定义换电积压惩罚项：

\[
J_{swap\_miss} = -\lambda_{swap}\sum_{t,i,l<L}(B_{i,l}^{t}-\mu_{i,l}^{t})
\]

说明：
- \(B_{i,l}^{t}\) 与 \(\mu_{i,l}^{t}\) 来自 Eq.(1)/(4) 体系；
- 该项用于附录鲁棒性实验，不进入主文主结论；
- 本轮 Recal_v1 不开启该项，仅冻结定义。

