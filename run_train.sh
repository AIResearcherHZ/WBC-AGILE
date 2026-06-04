#!/usr/bin/env bash
# 在独立的 tmux 会话里启动训练 —— 关闭终端 / SSH 断线都不会再把训练挂掉(SIGHUP)。
#
# 用法:
#   ./run_train.sh <会话名> <完整训练命令...>
#
# 例(单节点多卡 DDP):
#   ./run_train.sh standup_t1 \
#     python -m torch.distributed.run --standalone --nproc-per-node=4 \
#     scripts/train.py --task StandUp-Taks-T1-v0 --num_envs 4096 --headless --distributed
#
# 启动后:
#   tmux attach -t flat     # 进去看实时输出(脱离: 先按 Ctrl-b 再按 d,训练继续跑)
#   tail -f <脚本打印的日志路径>
#   tmux ls                 # 查看所有正在跑的训练会话
set -euo pipefail

SESSION="${1:?用法: ./run_train.sh <会话名> <训练命令...>}"
shift
[ "$#" -gt 0 ] || { echo "❌ 缺少训练命令。" >&2; exit 1; }

cd "$(dirname "$0")"
mkdir -p logs/launch
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/launch/${SESSION}_${TS}.log"

# ── rsl_rl 隔离(不卸载、不影响 atom01_train)────────────────────────────────────
# 本仓库自带定制版 rsl_rl(v2.3.3:L2C2 / 对称性 / 熵退火 / 师生蒸馏),与共享的
# isaaclab conda 环境里 atom01_train 全局安装的 rsl_rl(v3.3.0)不兼容。这里把自带版
# 前置到 PYTHONPATH:import 时 PathFinder 先扫 sys.path,早于全局 editable 安装注册在
# sys.meta_path 末尾的 finder,因此经本脚本启动的训练优先用自带版;atom01_train 的
# 全局安装完全不受影响。注意:务必经本脚本启动训练(裸跑 python scripts/train.py 不带
# PYTHONPATH 会用到环境里那份 v3.3.0,与本仓库配置不兼容)。
RSL_RL_DIR="$(pwd)/agile/algorithms/rsl_rl"
[ -d "$RSL_RL_DIR/rsl_rl" ] || echo "⚠️  未找到自带 rsl_rl 源码: $RSL_RL_DIR/rsl_rl" >&2

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "❌ tmux 会话 '$SESSION' 已存在。先 'tmux attach -t $SESSION' 看看,或换个会话名。" >&2
  exit 1
fi

# 把训练命令安全拼成一段 shell(printf %q 保留引号),在新 tmux 会话里执行并同时 tee 到日志。
# PYTHONUNBUFFERED=1 让日志实时刷新;结尾 read 让窗口在训练结束后保留,方便看尾部输出。
INNER="export PYTHONPATH=$(printf '%q' "$RSL_RL_DIR")\${PYTHONPATH:+:\$PYTHONPATH}; PYTHONUNBUFFERED=1 $(printf '%q ' "$@") 2>&1 | tee $(printf '%q' "$LOG"); echo; echo '==== 训练进程已退出,按回车关闭本窗口 ===='; read"
tmux new-session -d -s "$SESSION" "bash -lc $(printf '%q' "$INNER")"

echo "✅ 已在 tmux 会话 '$SESSION' 启动训练 —— 现在断开 SSH 也不会中断。"
echo "   实时查看:   tmux attach -t $SESSION      (脱离: Ctrl-b 然后 d)"
echo "   跟踪日志:   tail -f $(pwd)/$LOG"
echo "   全部会话:   tmux ls"
echo "   停止训练:   tmux kill-session -t $SESSION"
