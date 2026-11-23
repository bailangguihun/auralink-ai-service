# ArtLIVE 画智体 · AI Services

ArtLIVE 的内部 AI 服务集合。当前 R25 由两个轻量 Python 服务提供中国画导赏与评价能力；用户、任务、媒体和业务 API 统一由 [auralink-core-api](https://github.com/bailangguihun/auralink-core-api) 管理。

> AI 服务只监听回环地址，通过内部 token 与 Core API 通信。真实密钥、模型、图片、数据库及运行环境文件不进入仓库。

## 服务概览

| 目录 | 能力 | 技术与 R25 端口 | 当前状态 |
| --- | --- | --- | --- |
| `guide_service/` | 根据画作资料生成结构化中文导赏 | Flask · `127.0.0.1:19084` | R25 使用 |
| `critic-service/` | 四维专家两轮复议与综合评价 | FastAPI · `127.0.0.1:19085` | R25 使用 |
| `musicgen-service/` | 音乐生成 | Python | 保留，R25 不启动 |
| `vision-service/` | 视觉理解 | Python | 保留，R25 不启动 |
| `vmm-service/` | 本地多模态音乐生成 | Python / GPU | 保留，R25 不启动 |

生产进程和发布由 [auralink-deploy](https://github.com/bailangguihun/auralink-deploy) 的 R25 控制器管理；本仓库 CI 不操作生产环境。

## Guide Service

Guide 接收 Core API 提供的结构化画作资料和知识条目，返回摘要、八个章节、亮点与知识引用。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 返回服务状态及 provider/model 配置状态 |
| `POST` | `/v1/guide/generate` | 生成导赏，要求 `X-Auralink-Internal-Token` |

注意：

- 健康检查 HTTP 200 只表示端点可响应，还需检查 `providerConfigured` 与 `modelConfigured`。
- 生成接口先鉴权，再校验输入或调用 provider。
- 返回内容必须通过结构校验，不接受任意模型文本冒充成功结果。
- R25 运行端口为 19084；源码开发默认端口为 5003。
- 绑定地址必须是 `127.0.0.1` 或 `::1`。

详见 [Guide Service 说明](guide_service/README.md)。

## Critic Service

Critic 使用四位专家独立观察、共享意见复议，再由主持人汇总。完整流程包含九个模型步骤，输出结构化意见而非分数。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 返回配置、视觉能力与降级状态 |
| `GET` | `/api/v1/readiness` | 检查评价可用性，要求内部 token |
| `GET` | `/api/v1/evaluation-profiles` | 获取评价方案 |
| `POST` | `/api/v1/evaluations` | 提交图片评价，要求 `X-Auralink-Critic-Token` |

支持 `authoritative_four` 与 `requested_four` 两种评价方案。默认执行策略为 `isolated-stub-only`；生产必须显式选择 `configured-provider`，并同时具备内部令牌、模型、密钥及视觉能力。

健康检查返回 200 不等于可执行评价。应检查 `status`，并使用受保护的 readiness 端点确认 `evaluation_available`。

详见 [Critic Service 说明](critic-service/README.md)。

## 本地开发

### 环境要求

- Python 3.10
- pip
- 独立虚拟环境
- 运行真实 provider 时所需的服务端凭据

### 安装与测试

在仓库根目录执行：

```bash
git clone https://github.com/bailangguihun/auralink-ai-service.git
cd auralink-ai-service

python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate

python -m pip install -r guide_service/requirements.txt \
  -r critic-service/requirements.txt

python -m pytest -q tests
```

测试使用进程内 HTTP 客户端和确定性本地 provider：

- 不使用真实密钥
- 不访问模型 provider
- 不需要 GPU
- 隔离私有环境文件
- 阻止 Guide、Requests 与 HTTPX 的网络传输路径

也可分别运行契约测试：

```bash
python -m pytest -q tests/test_guide_contract.py
python -m pytest -q tests/test_critic_contract.py
```

离线测试通过只证明接口、鉴权和数据契约符合预期，不代表真实模型连通性、生成质量或 GPU 推理已经验证。

## 本地启动

Guide：

```bash
python -B -m guide_service
```

Critic（从 `critic-service` 目录或配置相应的 Python 导入路径）：

```bash
python -B -m uvicorn app.main:app --host 127.0.0.1 --port 19085
```

R25 生产环境由托管包装器注入配置并管理进程，不要用本地开发命令替换生产启动器。

## 配置与安全

可提交的 [`.env.example`](.env.example) 只记录无密钥示例。Guide 与 Critic 的主要配置包括：

### Guide

- `QWEN_API_KEY`
- `QWEN_BASE_URL`
- `QWEN_MODEL`
- `AURALINK_GUIDE_INTERNAL_TOKEN`
- `AURALINK_ENV_FILE`

### Critic

- `LLM_PROVIDER`
- `LLM_API_KEY` 或 provider 对应密钥
- `LLM_BASE_URL`
- `LLM_MODEL`
- `AURALINK_CRITIC_INTERNAL_TOKEN`
- `AURALINK_CRITIC_EXECUTION_POLICY`

安全规则：

- 真实 `.env`、内部令牌与 provider 密钥不得提交到 Git。
- 服务必须仅监听回环地址，不直接暴露到公网。
- 内部鉴权必须先于图片处理、输入解析和 provider 客户端创建。
- 未知执行策略、配置缺失或能力不匹配时必须保持不可用。
- 不要在发布过程中原地重装生产虚拟环境依赖。

## 发布流程

- 日常开发：功能分支 → `main` Pull Request
- 阶段发布：本仓库 `main` → `release` Pull Request
- CI 使用 Python 3.10，执行第一方源码编译和 Guide/Critic 离线契约测试
- `release` 还需验证发布来源
- CI 不发布生产版本，也不证明旧 GPU 服务能够加载或推理

生产发布必须经过版本绑定、健康与就绪检查及可恢复回滚流程。详见 [R25 AI 服务边界](docs/r25-ai-services.md)、[R25 发布说明](docs/r25-release.md) 和 [贡献说明](CONTRIBUTING.md)。
