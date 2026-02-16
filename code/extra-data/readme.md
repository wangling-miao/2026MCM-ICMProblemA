# 智能手机电池建模与数据资源综合说明文档

## 1. 理论与建模文档 (PDF)

本部分包含两个核心文档，提供了电池建模的物理基础、数学推导及算法框架。

### 1.1 智能手机功耗建模分析报告
*   **文件名**: `Zhi-Neng-Shou-Ji-Gong-Hao-Jian-Mo-Fen-Xi.pdf` [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/58364172/6826c0fe-6a10-4667-bde6-b618cc9cfb98/Zhi-Neng-Shou-Ji-Gong-Hao-Jian-Mo-Fen-Xi.pdf)
*   **核心内容**:
    *   **KiBaM模型**: 引入动力学电池模型（Kinetic Battery Model），将电池电荷分为“可用井”和“约束井”，用于解释电池的恢复效应和倍率容量效应 。 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/58364172/6826c0fe-6a10-4667-bde6-b618cc9cfb98/Zhi-Neng-Shou-Ji-Gong-Hao-Jian-Mo-Fen-Xi.pdf)
    *   **多物理场耦合**: 建立了包含屏幕（OLED像素权重）、处理器（DVFS动态电压频率调整）、网络（RRC状态机）的多维度功耗方程 。 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/58364172/6826c0fe-6a10-4667-bde6-b618cc9cfb98/Zhi-Neng-Shou-Ji-Gong-Hao-Jian-Mo-Fen-Xi.pdf)
    *   **SOC状态方程**: 定义了连续时间下的 SOC 微分方程，考虑了库仑效率和老化因子 。 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/58364172/6826c0fe-6a10-4667-bde6-b618cc9cfb98/Zhi-Neng-Shou-Ji-Gong-Hao-Jian-Mo-Fen-Xi.pdf)
    *   **优化建议**: 提出了“电池感知调度”和“后台活动抑制”等系统级优化策略 。 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/58364172/6826c0fe-6a10-4667-bde6-b618cc9cfb98/Zhi-Neng-Shou-Ji-Gong-Hao-Jian-Mo-Fen-Xi.pdf)

### 1.2 2026年MCM A题建模指南
*   **文件名**: `2026A-2.pdf` [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/58364172/3d8e0c12-8482-46ba-a51c-6b8ad0651cd2/2026A-2.pdf)
*   **核心内容**:
    *   **二阶RC等效电路**: 详细推导了包含欧姆内阻 \( R_0 \) 和两个极化环节（\( R_1C_1, R_2C_2 \)）的电池状态空间方程 。 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/58364172/3d8e0c12-8482-46ba-a51c-6b8ad0651cd2/2026A-2.pdf)
    *   **温度补偿机制**: 使用 Arrhenius 方程修正内阻，用二次多项式修正有效容量，以模拟低温下的性能衰减 。 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/58364172/3d8e0c12-8482-46ba-a51c-6b8ad0651cd2/2026A-2.pdf)
    *   **TTE预测算法**: 提供了基于 Runge-Kutta 法求解剩余续航时间（Time-to-Empty）的完整算法流程 。 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/58364172/3d8e0c12-8482-46ba-a51c-6b8ad0651cd2/2026A-2.pdf)
    *   **灵敏度分析**: 介绍了使用 Sobol 指数进行全局灵敏度分析的方法 。 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/58364172/3d8e0c12-8482-46ba-a51c-6b8ad0651cd2/2026A-2.pdf)

## 2. 数据集详解 (Excel)

本部分包含四个数据文件，涵盖了从实验室老化测试到真实用户行为的完整数据链。

### 2.1 NASA 锂离子电池老化数据集
*   **文件名**: `nasa_data1.xlsx` [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/58364172/ccf542ca-fb5e-4433-98ed-dfcddbda5782/nasa_data1.xlsx)
*   **用途**: 用于训练电池SOH（健康状态）估算模型及验证电压跌落特性。

#### Sheet: 容量衰减数据 (共 2794 条)
该表记录了电池在不同老化周期下的容量变化。

| 字段名 | 数据类型 | 样例数据 | 说明 |
| :--- | :--- | :--- | :--- |
| **电池ID** | String | B0047 | 实验电池的唯一编号 |
| **测试序号** | Integer | 0 | 实验进行的顺序索引 |
| **唯一ID** | Integer | 1 | 全局唯一记录标识 |
| **文件名** | String | 00001.csv | 对应原始波形文件的名称 |
| **容量(Ah)** | Float | 1.6743 | 当前老化周期下的实测容量 |
| **环境温度(°C)** | Integer | 4 | 实验环境温度（如4°C低温）|

#### Sheet: 放电曲线样本 (共 2000 条)
该表记录了单次放电过程中的高频采样数据。

| 字段名 | 数据类型 | 样例数据 | 说明 |
| :--- | :--- | :--- | :--- |
| **电压(V)** | Float | 4.2467 | 电池端电压瞬时值 |
| **电流(A)** | Float | 0.00025 | 电池电流（负值通常代表放电）|
| **温度(°C)** | Float | 6.21 | 电池表面实时温度 |
| **负载电压(V)** | Float | 0.0 | 负载端的测量电压 |
| **时间(s)** | Float | 9.36 | 相对放电开始的时间戳 |

***

### 2.2 电池剩余寿命 (RUL) 数据集
*   **文件名**: `battery_data-2.xlsx` [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/58364172/704fa41b-bc8e-4886-84ff-25d65763b373/battery_data-2.xlsx)
*   **用途**: 用于构建基于特征工程的电池寿命预测模型（RUL Prediction）。

#### Sheet: RUL数据 (共 15064 条)
包含丰富的充放电循环特征。

| 字段名 | 数据类型 | 样例数据 | 说明 |
| :--- | :--- | :--- | :--- |
| **循环次数** | Integer | 1 | 当前电池经历的充放电圈数 |
| **放电时间(s)** | Float | 2595.3 | 完整放电过程持续时长 |
| **3.6-3.4V衰减时间(s)**| Float | 1151.49 | 电压在特定区间下降所耗时间（关键特征）|
| **最大放电电压(V)** | Float | 3.67 | 放电阶段记录到的最高电压 |
| **4.15V时间(s)** | Float | 5460.0 | 恒压充电阶段保持在4.15V的时长 |
| **剩余寿命(RUL)** | Integer | 1112 | **标签列**：距离电池失效还剩多少次循环 |

***

### 2.3 移动设备用户行为数据集
*   **文件名**: `users_data.xlsx` [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/58364172/88b8561a-fe51-4a4f-9726-306ab619d484/users_data.xlsx)
*   **用途**: 用于生成用户画像（User Persona），模拟不同使用强度下的负载模型。

#### Sheet: 用户行为数据 (共 700 条)
多维度用户使用习惯统计。

| 字段名 | 数据类型 | 样例数据 | 说明 |
| :--- | :--- | :--- | :--- |
| **用户ID** | Integer | 1 | 用户唯一标识 |
| **设备型号** | String | Google Pixel 5 | 手机硬件型号，决定电池基准容量 |
| **日均应用使用(分)** | Integer | 393 | App活跃总时长 |
| **日均屏幕时间(时)** | Float | 6.4 | 屏幕开启总时长（SOT） |
| **日均电池消耗(mAh)**| Integer | 1872 | 每日消耗的电量总和 |
| **安装应用数** | Integer | 67 | 影响后台唤醒频率的潜在因子 |
| **用户行为类别** | Integer | 4 | 聚类标签（如：1=轻度, 5=重度） |

***

### 2.4 智能手机电池实时使用数据集
*   **文件名**: `battery_data.xlsx` [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/58364172/3a9b854d-afcd-4982-be31-b47a9368e7da/battery_data.xlsx)
*   **用途**: 验证微观功耗模型，分析具体App对电流和电压的瞬时影响。

#### Sheet: 电池使用数据 (共 10000 条)
高精度的真实手机运行日志。

| 字段名 | 数据类型 | 样例数据 | 说明 |
| :--- | :--- | :--- | :--- |
| **时间** | DateTime | 2019-10-09 05:45:38 | 采样时间戳 |
| **电量百分比** | Integer | 37 | 系统显示的SOC值 |
| **温度(°C)** | Float | 24.7 | 电池传感器温度 |
| **电压(mV)** | Integer | 3741 | 瞬时电压（毫伏） |
| **电流(mA)** | Integer | 130 | 瞬时电流（正值为放电或充电需结合状态）|
| **屏幕状态** | String | 关 (Off) | 屏幕开启/关闭标记 |
| **当前应用** | String | com.zopper.batteryage | 前台运行的进程包名 |

