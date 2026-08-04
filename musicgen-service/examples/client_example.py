#!/usr/bin/env python3
"""
MusicGen API 客户端示例

使用方法:

1. 健康检查 (无需认证):
   python client_example.py --check-health

2. 管理员功能 (需要管理员token):
   python client_example.py --admin-token YOUR_ADMIN_TOKEN --admin-create-key "用户1"
   python client_example.py --admin-token YOUR_ADMIN_TOKEN --admin-list-keys
   python client_example.py --admin-token YOUR_ADMIN_TOKEN --admin-preload medium small

3. 用户功能 (需要API key):
   python client_example.py --api-key YOUR_API_KEY --show-models
   python client_example.py --api-key YOUR_API_KEY --show-usage
   python client_example.py --api-key YOUR_API_KEY --prompt "jazz piano melody"

示例:
   # 创建API密钥
   python client_example.py --admin-token admin_token_change_me --admin-create-key "测试用户"
   
   # 预加载模型（提高响应速度）
   python client_example.py --admin-token admin_token_change_me --admin-preload medium
   
   # 生成音乐
   python client_example.py --api-key mgapi_xxxxx --prompt "upbeat jazz with saxophone" --model medium
"""
import requests
import json
import base64
import argparse
from pathlib import Path


class MusicGenClient:
    """MusicGen API 客户端"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_music(self, prompt: str, model: str = "medium", 
                      duration: float = 8.0, **kwargs) -> dict:
        """生成音乐"""
        data = {
            "prompt": prompt,
            "model": model,
            "duration": duration,
            **kwargs
        }
        
        response = requests.post(
            f"{self.base_url}/generate",
            headers=self.headers,
            json=data
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API请求失败: {response.status_code} - {response.text}")
    
    def get_models(self) -> dict:
        """获取可用模型"""
        response = requests.get(
            f"{self.base_url}/models",
            headers=self.headers
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"获取模型信息失败: {response.status_code} - {response.text}")
    
    def get_usage(self) -> dict:
        """获取使用统计"""
        response = requests.get(
            f"{self.base_url}/usage",
            headers=self.headers
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"获取使用统计失败: {response.status_code} - {response.text}")
    
    def health_check(self) -> dict:
        """健康检查"""
        response = requests.get(f"{self.base_url}/health")
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"健康检查失败: {response.status_code} - {response.text}")
    
    def save_audio(self, audio_data: str, output_path: str):
        """保存base64编码的音频到文件"""
        audio_bytes = base64.b64decode(audio_data)
        with open(output_path, 'wb') as f:
            f.write(audio_bytes)
        print(f"音频已保存到: {output_path}")


class MusicGenAdminClient:
    """MusicGen API 管理员客户端"""
    
    def __init__(self, base_url: str, admin_token: str):
        self.base_url = base_url.rstrip('/')
        self.admin_token = admin_token
        self.headers = {
            "Authorization": f"Bearer {admin_token}",
            "Content-Type": "application/json"
        }
    
    def create_api_key(self, name: str, usage_limit: int = -1, daily_limit: int = 100) -> dict:
        """创建新的API密钥"""
        data = {
            "name": name,
            "usage_limit": usage_limit,
            "daily_limit": daily_limit
        }
        
        response = requests.post(
            f"{self.base_url}/admin/keys",
            headers=self.headers,
            json=data
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"创建API密钥失败: {response.status_code} - {response.text}")
    
    def list_api_keys(self) -> dict:
        """列出所有API密钥"""
        response = requests.get(
            f"{self.base_url}/admin/keys",
            headers=self.headers
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"获取API密钥列表失败: {response.status_code} - {response.text}")
    
    def preload_models(self, models: list = None) -> dict:
        """手动预加载模型"""
        data = {}
        if models:
            data["models"] = models
        
        response = requests.post(
            f"{self.base_url}/admin/preload",
            headers=self.headers,
            json=data
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"预加载模型失败: {response.status_code} - {response.text}")


def main():
    parser = argparse.ArgumentParser(description="MusicGen API 客户端")
    parser.add_argument("--url", default="http://localhost:8000", help="API服务器地址")
    parser.add_argument("--api-key", help="API密钥（用于音乐生成）")
    parser.add_argument("--admin-token", help="管理员认证token（用于管理功能）")
    parser.add_argument("--prompt", help="音乐生成提示词")
    parser.add_argument("--model", default="medium", choices=["small", "medium", "large"], help="模型类型")
    parser.add_argument("--duration", type=float, default=8.0, help="生成时长（秒）")
    parser.add_argument("--output", default="generated_music.wav", help="输出文件路径")
    parser.add_argument("--temperature", type=float, default=1.0, help="生成随机性")
    parser.add_argument("--top-k", type=int, default=250, help="Top-k采样")
    
    # 基础功能
    parser.add_argument("--check-health", action="store_true", help="只进行健康检查")
    parser.add_argument("--show-models", action="store_true", help="显示可用模型")
    parser.add_argument("--show-usage", action="store_true", help="显示使用统计")
    
    # 管理员功能
    parser.add_argument("--admin-create-key", help="创建API密钥（需要管理员token）")
    parser.add_argument("--usage-limit", type=int, default=-1, help="API密钥使用次数限制")
    parser.add_argument("--daily-limit", type=int, default=100, help="API密钥每日使用限制")
    parser.add_argument("--admin-list-keys", action="store_true", help="列出所有API密钥（需要管理员token）")
    parser.add_argument("--admin-preload", nargs="*", help="手动预加载模型（需要管理员token），如 --admin-preload medium small")
    
    args = parser.parse_args()
    
    try:
        # 健康检查（无需认证）
        if args.check_health:
            client = MusicGenClient(args.url, "dummy")  # 健康检查不需要真实密钥
            health = client.health_check()
            print("🏥 服务器健康状态:")
            print(json.dumps(health, indent=2, ensure_ascii=False))
            return
        
        # 管理员功能
        if args.admin_create_key or args.admin_list_keys or args.admin_preload is not None:
            if not args.admin_token:
                print("❌ 管理员功能需要提供 --admin-token")
                return
            
            admin_client = MusicGenAdminClient(args.url, args.admin_token)
            
            if args.admin_create_key:
                print(f"🔑 正在创建API密钥: {args.admin_create_key}")
                result = admin_client.create_api_key(
                    name=args.admin_create_key,
                    usage_limit=args.usage_limit,
                    daily_limit=args.daily_limit
                )
                print("✅ API密钥创建成功!")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                return
            
            if args.admin_list_keys:
                print("📋 正在获取API密钥列表...")
                result = admin_client.list_api_keys()
                print("📋 API密钥列表:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                return
            
            if args.admin_preload is not None:
                models_to_preload = args.admin_preload if args.admin_preload else None
                print(f"⚡ 正在预加载模型: {models_to_preload or '使用默认配置'}")
                result = admin_client.preload_models(models_to_preload)
                print("⚡ 预加载结果:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                return
        
        # 用户功能（需要API密钥）
        if not args.api_key:
            print("❌ 用户功能需要提供 --api-key")
            return
        
        client = MusicGenClient(args.url, args.api_key)
        
        # 显示模型信息
        if args.show_models:
            models = client.get_models()
            print("🎵 可用模型:")
            print(json.dumps(models, indent=2, ensure_ascii=False))
            return
        
        # 显示使用统计
        if args.show_usage:
            usage = client.get_usage()
            print("📊 使用统计:")
            print(json.dumps(usage, indent=2, ensure_ascii=False))
            return
        
        # 生成音乐
        if not args.prompt:
            print("❌ 音乐生成需要提供 --prompt")
            return
        
        print(f"🎵 正在生成音乐...")
        print(f"   提示词: {args.prompt}")
        print(f"   模型: {args.model}")
        print(f"   时长: {args.duration}秒")
        
        result = client.generate_music(
            prompt=args.prompt,
            model=args.model,
            duration=args.duration,
            temperature=args.temperature,
            top_k=args.top_k
        )
        
        if result["success"]:
            print("✅ 音乐生成成功!")
            
            # 保存音频文件
            audio_data = result["data"]["audio_data"]
            client.save_audio(audio_data, args.output)
            
            # 显示生成信息
            data = result["data"]
            print(f"   采样率: {data['sample_rate']} Hz")
            print(f"   实际时长: {data['duration']} 秒")
            print(f"   使用模型: {data['model_used']}")
            
        else:
            print(f"❌ 生成失败: {result['message']}")
            
    except Exception as e:
        print(f"❌ 错误: {e}")


if __name__ == "__main__":
    main() 