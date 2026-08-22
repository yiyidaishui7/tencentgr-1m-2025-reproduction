# 可复制复现手册

## 1. 环境

- Python 3.10+
- PyTorch 2.x
- 训练建议使用单张至少 24 GB 显存的 CUDA GPU；也支持 CPU 和 Ascend 路由。
- 数据和模型需要约数十 GB 工作空间；原始数据不进入 Git 仓库。

```bash
git clone https://github.com/yiyidaishui7/tencentgr-1m-2025-reproduction.git
cd tencentgr-1m-2025-reproduction
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 获取和审计数据

```bash
python scripts/download_tencentgr_1m.py /data/TencentGR-1M
python scripts/validate_tencentgr_1m.py /data/TencentGR-1M
python scripts/audit_id_alignment.py /data/TencentGR-1M
```

数据来源为 `TAAC2025/TencentGR-1M`。审计应在训练前完成；任何缺失文件、Schema 不兼容或 ID 映射错误都应视为硬失败。

## 3. 训练 MM101

```bash
python main.py \
  --data_path /data/TencentGR-1M \
  --output_dir ./runs/mm101 \
  --device cuda:0 \
  --seed 2025 \
  --maxlen 101 \
  --batch_size 2048 \
  --num_epochs 3
```

## 4. 训练 no-MM101

```bash
python main.py \
  --data_path /data/TencentGR-1M \
  --output_dir ./runs/nomm101 \
  --device cuda:0 \
  --seed 2025 \
  --maxlen 101 \
  --batch_size 2048 \
  --num_epochs 3 \
  --disable_mm_emb
```

近期窗口实验只把 `--maxlen` 改为 `50`；不得同时修改优化器、Epoch 或隐藏维度。

## 5. 正式离线评测

```bash
python offline_eval.py \
  --data_path /data/TencentGR-1M \
  --checkpoint ./runs/mm101/checkpoints/<run>/<checkpoint>/model.pt \
  --output_dir ./eval/mm101 \
  --scratch_dir ./scratch/mm101 \
  --device cuda:0 \
  --seed 2025 \
  --valid_ratio 0.1 \
  --maxlen 101
```

评测 no-MM 权重时必须同时加入 `--disable_mm_emb`。`maxlen` 必须与训练一致。

预期输出：

- `offline_metrics.json`：协议、人口审计、整体和历史长度分桶指标。
- `offline_predictions.npz`：逐用户 Top-10、Target、用户 ID 和历史长度。
- 模型权重和执行日志。
- SHA-256 清单。

## 6. 正确性检查

```bash
python -m unittest discover -s tests -v
python -m compileall -q .
```

对比实验还必须验证：

1. `user_ids`、`target_ids` 和 `prefix_lengths` 数组长度及顺序一致。
2. `evaluated_users` 均为 78,921。
3. 每个本地文件与远端清单 SHA-256 一致。
4. 临时检索文件只在评测期间存在。
5. 任何发布结论都注明固定 seed 和非官方榜单边界。

四组预测完成后生成统一的机器可读比较报告：

```bash
python scripts/compare_four_variants.py \
  --mm101 ./eval/mm101/offline_predictions.npz \
  --nomm101 ./eval/nomm101/offline_predictions.npz \
  --mm50 ./eval/mm50/offline_predictions.npz \
  --nomm50 ./eval/nomm50/offline_predictions.npz \
  --output ./eval/four_way_comparison.json
```

该命令会硬性检查用户、Target 和原始历史长度逐行一致，并输出总体指标、四条成对效应、差分中的差分交互项，以及 0–20、21–50、51–80、81+ 四个历史桶的同口径统计。

## 7. 发布

- GitHub：只发布代码、小型指标、说明和测试。
- 公开 Hugging Face：SafeTensors 权重和可公开小产物。
- 私有归档：原始/受限数据、PyTorch 权重、预测和完整日志。
- 上传后重新列举远端文件；LFS/Xet 文件核对对象 SHA，普通文件下载后再计算 SHA-256。
