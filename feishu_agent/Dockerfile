FROM python:3.13-slim

WORKDIR /app

# 1. 安装系统编译工具（很少变，放最前面）
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# 2. 先只复制依赖文件（这是关键）
COPY requirements.txt .

# 3. 安装Python依赖（只要 requirements.txt 不变，这层永远缓存）
RUN pip install --no-cache-dir -r requirements.txt

# 4. 修改 llama-index 源码（pip 安装后文件固定，也享受缓存）
RUN sed -i 's|"BAAI/bge-m3"|"/app/models/bge-m3"|g' \
    /usr/local/lib/python3.13/site-packages/llama_index/vector_stores/milvus/utils.py

# 5. 最后复制项目代码（只有这层会因代码变动而重建）
COPY . .

# 6. 脚本权限
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

# 启动应用
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python3", "src/openai_agents_system_integrate_feishu.py"]