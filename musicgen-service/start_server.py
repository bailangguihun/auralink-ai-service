#!/usr/bin/env python3
"""
MusicGen API Server 启动脚本
"""
import os
import sys
import argparse
import logging
import uvicorn
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from musicgen_api.config import load_config, save_default_config


def setup_logging(level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def check_dependencies():
    """检查必要的依赖"""
    try:
        import torch
        import torchaudio
        import fastapi
        import uvicorn
        
        # 检查CUDA
        if torch.cuda.is_available():
            print(f"CUDA可用，检测到 {torch.cuda.device_count()} 张GPU")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("未检测到CUDA，将使用CPU运行（性能较低）")
            
        # 检查AudioCraft
        try:
            import audiocraft
            # 避免循环导入，只检查audiocraft包是否可用
            audiocraft_path = audiocraft.__file__
            print("AudioCraft导入成功")
        except ImportError:
            print("AudioCraft导入失败，请安装: pip install -e ./audiocraft")
            return False
            
        print("所有依赖检查完成")
        return True
        
    except ImportError as e:
        print(f"依赖检查失败: {e}")
        print("请运行: pip install -r requirements.txt")
        return False


def create_default_config():
    """创建默认配置文件"""
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"创建默认配置文件: {config_path}")
        save_default_config(config_path)
    else:
        print(f"配置文件已存在: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="MusicGen API Server")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--host", default=None, help="服务器主机地址")
    parser.add_argument("--port", type=int, default=None, help="服务器端口")
    parser.add_argument("--workers", type=int, default=None, help="工作进程数")
    parser.add_argument("--reload", action="store_true", help="开发模式，自动重载")
    parser.add_argument("--check-deps", action="store_true", help="检查依赖并退出")
    parser.add_argument("--create-config", action="store_true", help="创建默认配置文件并退出")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    # 检查依赖
    if args.check_deps:
        success = check_dependencies()
        sys.exit(0 if success else 1)
    
    # 创建配置文件
    if args.create_config:
        create_default_config()
        sys.exit(0)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 创建默认配置（如果不存在）
    create_default_config()
    
    # 加载配置
    try:
        config = load_config(args.config)
        print(f"配置文件加载成功: {args.config}")
    except Exception as e:
        print(f"配置文件加载失败: {e}")
        sys.exit(1)
    
    # 命令行参数覆盖配置文件
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.workers:
        config.workers = args.workers
    if args.reload:
        config.reload = True
    
    print(f"启动MusicGen API Server")
    print(f"   地址: http://{config.host}:{config.port}")
    print(f"   文档: http://{config.host}:{config.port}/docs")
    print(f"   健康检查: http://{config.host}:{config.port}/health")
    print(f"   工作进程: {config.workers}")
    print(f"   重载模式: {config.reload}")
    
    # 启动服务器
    try:
        uvicorn.run(
            "musicgen_api.api:app",
            host=config.host,
            port=config.port,
            workers=config.workers if not config.reload else 1,
            reload=config.reload,
            log_level=args.log_level.lower()
        )
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"服务器启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 