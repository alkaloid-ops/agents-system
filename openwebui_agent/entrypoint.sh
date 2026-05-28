#!/bin/bash
# entrypoint.sh - 下载BGE系列模型并启动服务

set -e  # 出错即退出

# 设置 Hugging Face 镜像源（可通过环境变量覆盖）
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

# 模型存储路径（可通过环境变量覆盖）
MODEL_BASE=${MODEL_BASE:-/app/models}
EMBEDDING_MODEL_PATH="${MODEL_BASE}/bge-m3"
RERANKER_MODEL_PATH="${MODEL_BASE}/bge-reranker-v2-m3"

# 模型ID
EMBEDDING_MODEL_ID="BAAI/bge-m3"
RERANKER_MODEL_ID="BAAI/bge-reranker-v2-m3"

# 颜色输出函数（可选，便于观察日志）
info() { echo -e "\033[32m[INFO]\033[0m $1"; }
warn() { echo -e "\033[33m[WARN]\033[0m $1"; }
error() { echo -e "\033[31m[ERROR]\033[0m $1"; }

# 检查目录是否为空或不存在
is_empty_dir() {
    [ ! -d "$1" ] || [ -z "$(ls -A "$1")" ]
}

# 下载模型函数
download_model() {
    local model_id=$1
    local target_dir=$2
    local model_name=$(basename "$model_id")

    if is_empty_dir "$target_dir"; then
        info "模型 $model_name 不存在或为空，开始从 $HF_ENDPOINT 下载..."
        # 使用 huggingface-cli 下载，支持断点续传
        huggingface-cli download "$model_id" \
            --local-dir "$target_dir" \
            --local-dir-use-symlinks False \
            --resume-download \
            --quiet
        info "模型 $model_name 下载完成。"
    else
        info "模型 $model_name 已存在于 $target_dir，跳过下载。"
    fi
}

if [ -f "${EMBEDDING_MODEL_PATH}/config.json" ] && [ -f "${RERANKER_MODEL_PATH}/config.json" ]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    info "检测到模型已存在，启用离线模式"
fi

# 创建模型目录（如果不存在）
mkdir -p "$EMBEDDING_MODEL_PATH" "$RERANKER_MODEL_PATH"

# 下载 embedding 模型
download_model "$EMBEDDING_MODEL_ID" "$EMBEDDING_MODEL_PATH"

# 下载 reranker 模型
download_model "$RERANKER_MODEL_ID" "$RERANKER_MODEL_PATH"

info "所有模型准备就绪，启动应用服务..."

exec "$@"
