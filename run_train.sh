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
# PYTHONPATH 会用到环境里那份 v3.3.0,与本仓库配置不兼容)。
RSL_RL_DIR="$(pwd)/agile/algorithms/rsl_rl"
[ -d "$RSL_RL_DIR/rsl_rl" ] || echo "⚠️  未找到自带 rsl_rl 源码: $RSL_RL_DIR/rsl_rl" >&2

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "❌ tmux 会话 '$SESSION' 已存在。先 'tmux attach -t $SESSION' 看看,或换个会话名。" >&2
  exit 1
fi

# ── GPU 可见性以"调用本脚本的 shell"为准 ──────────────────────────────────────
# tmux 新会话继承的是 tmux 服务器的全局环境,不是当前 shell:若服务器曾被某次
# 单卡训练(export CUDA_VISIBLE_DEVICES=0)带脏,多卡 DDP 的每个 rank 都只看得到
# 1 块卡,直接 invalid device ordinal。这里显式设置/清除,杜绝隐式继承。
if [ -n "${CUDA_VISIBLE_DEVICES+x}" ]; then
  GPU_ENV="export CUDA_VISIBLE_DEVICES=$(printf '%q' "$CUDA_VISIBLE_DEVICES")"
else
  GPU_ENV="unset CUDA_VISIBLE_DEVICES"
fi

# 把训练命令安全拼成一段 shell(printf %q 保留引号),在新 tmux 会话里执行并同时 tee 到日志。
# PYTHONUNBUFFERED=1 让日志实时刷新;结尾 read 让窗口在训练结束后保留,方便看尾部输出。
NCCL_ENV="export NCCL_P2P_DISABLE=\${NCCL_P2P_DISABLE:-1} NCCL_IB_DISABLE=\${NCCL_IB_DISABLE:-1} TORCH_NCCL_ASYNC_ERROR_HANDLING=\${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1} TORCH_NCCL_DESYNC_DEBUG=\${TORCH_NCCL_DESYNC_DEBUG:-1} TORCH_NCCL_ENABLE_MONITORING=\${TORCH_NCCL_ENABLE_MONITORING:-1} TORCH_NCCL_TRACE_BUFFER_SIZE=\${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000} TORCH_NCCL_DUMP_ON_TIMEOUT=\${TORCH_NCCL_DUMP_ON_TIMEOUT:-1} TORCH_NCCL_DEBUG_INFO_TEMP_FILE=\${TORCH_NCCL_DEBUG_INFO_TEMP_FILE:-/tmp/nccl_trace_} NCCL_DEBUG=\${NCCL_DEBUG:-WARN}"
INNER="$GPU_ENV; export PYTHONPATH=$(printf '%q' "$RSL_RL_DIR")\${PYTHONPATH:+:\$PYTHONPATH}; $NCCL_ENV; PYTHONUNBUFFERED=1 $(printf '%q ' "$@") 2>&1 | tee $(printf '%q' "$LOG"); echo; echo '==== 训练进程已退出,按回车关闭本窗口 ===='; read"
tmux new-session -d -s "$SESSION" "bash -lc $(printf '%q' "$INNER")"

echo "✅ 已在 tmux 会话 '$SESSION' 启动训练 —— 现在断开 SSH 也不会中断。"
echo "   实时查看:   tmux attach -t $SESSION      (脱离: Ctrl-b 然后 d)"
echo "   跟踪日志:   tail -f $(pwd)/$LOG"
echo "   全部会话:   tmux ls"
echo "   停止训练:   tmux kill-session -t $SESSION"
