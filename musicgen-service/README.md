# MusicGen API Server

基于 Facebook Research AudioCraft 的 MusicGen 模型构建的音乐生成 API 服务器。

## 功能特性

- 🎵 基于文本生成高质量音乐
- 🔑 API Key 认证和使用计数
- 📊 支持多种模型尺寸 (small/medium/large)
- 🎛️ 可配置的生成参数
- 💾 SQLite 数据库存储用户数据
- 🚀 FastAPI + uvicorn 高性能服务
- 📱 完整的 API 文档 (Swagger UI)
- 🌏 支持中国大陆镜像源

## 系统要求

- Python 3.9+
- NVIDIA GPU (推荐，支持 CUDA)
- 至少 16GB 显存 (用于 large 模型)
- 至少 8GB 显存 (用于 medium 模型)
- 至少 4GB 显存 (用于 small 模型)

## 快速开始

### 1. 克隆项目

```bash
git clone https://ghfast.top/https://github.com/facebookresearch/audiocraft.git
```

### 2. 创建虚拟环境

```bash
conda create -n musicgen-api python=3.9 -y
conda activate musicgen-api
```

### 3. 安装依赖

```bash
# 安装项目依赖
pip install -r requirements.txt

# 安装 AudioCraft
cd audiocraft
pip install -e .
cd ..
```

### 4. 配置服务器

```bash
# 创建默认配置文件
python start_server.py --create-config

# 编辑配置文件以适应您的GPU设置
vim config.yaml
```

### 5. 启动服务器

```bash
# 检查依赖
python start_server.py --check-deps

# 启动服务器
python start_server.py
```

服务器将在 http://localhost:8000 启动。

## 配置说明

### GPU 配置

在 `config.yaml` 中配置不同模型使用的 GPU：

```yaml
models:
  small:
    gpu_id: 0      # 使用第一张GPU
    max_duration: 30
  medium:
    gpu_id: 0      # 使用第一张GPU  
    max_duration: 30
  large:
    gpu_id: 1      # 使用第二张GPU（如果有的话）
    max_duration: 60
```

### 镜像源配置

为中国大陆用户提供了镜像源支持：

```yaml
hf_endpoint: "https://hf-mirror.com"  # HuggingFace 镜像
```

### 管理员认证配置

为了保护管理端点的安全，系统使用固定的管理员认证token：

```yaml
admin_token: "admin_token_change_me"  # 管理员认证token，请修改为安全的值
```

⚠️ **重要**: 在生产环境中请务必修改默认的 `admin_token` 为安全的值！

## API 使用

### 获取 API Key

服务器启动后会自动创建一个演示 API Key，可在日志中查看。

或者通过管理接口创建（需要管理员认证）：

```bash
curl -X POST "http://localhost:8000/admin/keys" \
     -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"name": "我的密钥", "daily_limit": 100}'
```

### 管理员功能

管理员功能需要使用 `Authorization: Bearer YOUR_ADMIN_TOKEN` 头部认证：

```bash
# 创建API密钥
curl -X POST "http://localhost:8000/admin/keys" \
     -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"name": "新用户", "usage_limit": 1000, "daily_limit": 50}'

# 列出所有API密钥
curl -X GET "http://localhost:8000/admin/keys" \
     -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

### 生成音乐

```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "upbeat jazz melody with saxophone",
       "model": "medium",
       "duration": 10.0,
       "temperature": 1.0
     }'
```

### 查看 API 文档

访问 http://localhost:8000/docs 查看完整的 API 文档。

## API 端点

| 端点 | 方法 | 描述 | 认证要求 |
|------|------|------|----------|
| `/` | GET | 服务器状态 | 无 |
| `/health` | GET | 健康检查 | 无 |
| `/generate` | POST | 生成音乐 | API Key |
| `/models` | GET | 获取模型信息 | API Key |
| `/usage` | GET | 查看使用统计 | API Key |
| `/admin/keys` | POST | 创建 API Key | 管理员 Token |
| `/admin/keys` | GET | 列出所有 API Key | 管理员 Token |
| `/admin/preload` | POST | 手动预加载模型 | 管理员 Token |

## 生成参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `prompt` | string | - | 音乐生成提示词 |
| `model` | string | "medium" | 模型类型 (small/medium/large) |
| `duration` | float | 8.0 | 生成时长（秒） |
| `temperature` | float | 1.0 | 生成随机性 (0.1-2.0) |
| `top_k` | int | 250 | Top-k 采样 |
| `top_p` | float | 0.0 | Top-p 采样 |
| `cfg_coef` | float | 3.0 | 分类器引导系数 |

## 部署说明

### Docker 部署（可选）

```bash
# 构建镜像
docker build -t musicgen-api .

# 运行容器
docker run -d -p 8000:8000 --gpus all musicgen-api
```

### 生产环境

对于生产环境，建议：

1. 使用反向代理（nginx）
2. 配置 HTTPS
3. 设置适当的日志级别
4. 配置数据库备份
5. 监控 GPU 内存使用

## 故障排除

### 常见问题

1. **CUDA 内存不足**
   - 使用更小的模型
   - 减少并发请求
   - 调整 batch size

2. **模型加载失败**
   - 检查网络连接
   - 确认镜像源配置
   - 检查磁盘空间

3. **音频质量问题**
   - 调整 temperature 参数
   - 使用更大的模型
   - 优化提示词

4. **AudioCraft 循环导入问题**
   - **症状**: 启动时预加载失败，错误信息包含 "circular import" 或 "SegmentWithAttributes"
   - **原因**: AudioCraft 本身存在循环导入问题
   - **解决方案**:
     - 禁用自动预加载：在 `config.yaml` 中设置 `preload_on_startup: false`
     - 使用手动预加载API：启动后调用 `/admin/preload` 端点
     - 模型会在首次使用时自动加载（会有一些延迟）

5. **预加载功能**
   ```bash
   # 手动预加载模型（管理员功能）
   curl -X POST "http://localhost:8000/admin/preload" \
        -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
        -H "Content-Type: application/json" \
        -d '{"models": ["medium"]}'
   ```

### 日志查看

```bash
# 启动时查看详细日志
python start_server.py --log-level DEBUG
```

## 许可证

本项目基于 MIT 许可证。AudioCraft 模型权重遵循 CC-BY-NC 4.0 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 支持

如有问题，请查看：
- [AudioCraft 官方文档](https://github.com/facebookresearch/audiocraft)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
- 项目 Issues 页面 