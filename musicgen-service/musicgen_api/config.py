"""
配置管理模块
"""
import os
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    size: str
    gpu_id: int
    memory_requirement: str
    max_duration: int


@dataclass
class APIConfig:
    """API配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    database_url: str = "sqlite:///./musicgen_api.db"
    
    # Hugging Face镜像配置
    hf_endpoint: str = "https://hf-mirror.com"
    
    # 管理端口认证配置
    admin_token: str = os.getenv("MUSICGEN_ADMIN_TOKEN", "change-me-before-production")
    
    # 模型预加载配置
    preload_models: List[str] = None  # 启动时预加载的模型列表，如 ["medium", "small"]
    preload_on_startup: bool = True  # 是否在启动时预加载模型
    
    # 模型配置
    models: Dict[str, ModelConfig] = None
    
    def __post_init__(self):
        if self.models is None:
            self.models = {
                "small": ModelConfig(
                    name="facebook/musicgen-small", 
                    size="300M", 
                    gpu_id=0, 
                    memory_requirement="4GB",
                    max_duration=30
                ),
                "medium": ModelConfig(
                    name="facebook/musicgen-medium", 
                    size="1.5B", 
                    gpu_id=0, 
                    memory_requirement="8GB",
                    max_duration=30
                ),
                "large": ModelConfig(
                    name="facebook/musicgen-large", 
                    size="3.3B", 
                    gpu_id=1, 
                    memory_requirement="16GB",
                    max_duration=60
                )
            }
        
        # 只有当 preload_models 真的是 None 时才设置默认值
        # 如果从配置文件加载了 preload_models，则保持原值
        if self.preload_models is None:
            # 默认预加载 medium 模型，平衡性能和资源占用
            self.preload_models = ["medium"]


def load_config(config_path: Optional[str] = None) -> APIConfig:
    """加载配置文件"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f) or {}

        env_admin_token = os.getenv("MUSICGEN_ADMIN_TOKEN")
        if env_admin_token:
            config_data["admin_token"] = env_admin_token
        # 转换模型配置
        if 'models' in config_data:
            models = {}
            for model_name, model_data in config_data['models'].items():
                models[model_name] = ModelConfig(**model_data)
            config_data['models'] = models
        
        return APIConfig(**config_data)
    
    return APIConfig()


def save_default_config(config_path: str = "config.yaml"):
    """保存默认配置文件"""
    config = APIConfig()
    
    config_data = {
        "host": config.host,
        "port": config.port,
        "workers": config.workers,
        "reload": config.reload,
        "database_url": config.database_url,
        "hf_endpoint": config.hf_endpoint,
        "preload_models": config.preload_models,
        "preload_on_startup": config.preload_on_startup,
        "models": {
            name: {
                "name": model.name,
                "size": model.size,
                "gpu_id": model.gpu_id,
                "memory_requirement": model.memory_requirement,
                "max_duration": model.max_duration
            }
            for name, model in config.models.items()
        }
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True) 