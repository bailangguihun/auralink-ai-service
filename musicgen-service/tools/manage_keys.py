#!/usr/bin/env python3
"""
API密钥管理工具
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from musicgen_api.database import APIKeyManager


def create_key(args):
    """创建新的API密钥"""
    manager = APIKeyManager(args.db_path)
    
    api_key = manager.generate_api_key(
        name=args.name,
        usage_limit=args.usage_limit,
        daily_limit=args.daily_limit
    )
    
    print(f"API密钥创建成功!")
    print(f"   名称: {args.name}")
    print(f"   密钥: {api_key}")
    print(f"   使用限制: {args.usage_limit if args.usage_limit > 0 else '无限制'}")
    print(f"   每日限制: {args.daily_limit}")
    print()
    print("请妥善保管您的API密钥，它将不会再次显示！")


def list_keys(args):
    """列出所有API密钥"""
    manager = APIKeyManager(args.db_path)
    keys = manager.list_api_keys()
    
    if not keys:
        print("没有找到任何API密钥")
        return
    
    print(f"API密钥列表 (共 {len(keys)} 个):")
    print()
    
    for i, key in enumerate(keys, 1):
        status = "🟢 活跃" if key["is_active"] else "🔴 禁用"
        usage_limit = key["usage_limit"] if key["usage_limit"] > 0 else "无限制"
        
        print(f"{i}. {key['key_name']}")
        print(f"   状态: {status}")
        print(f"   创建时间: {key['created_at']}")
        print(f"   最后使用: {key['last_used'] or '从未使用'}")
        print(f"   使用次数: {key['usage_count']} / {usage_limit}")
        print(f"   今日使用: {key['daily_usage']} / {key['daily_limit']}")
        print(f"   速率限制: {key['rate_limit']} 次/分钟")
        print()


def deactivate_key(args):
    """禁用API密钥"""
    manager = APIKeyManager(args.db_path)
    
    success = manager.deactivate_key(args.api_key)
    
    if success:
        print("API密钥已成功禁用")
    else:
        print("API密钥不存在或已经被禁用")


def show_usage(args):
    """显示API密钥使用统计"""
    manager = APIKeyManager(args.db_path)
    
    key_info = manager.get_key_info(args.api_key)
    
    if not key_info:
        print("API密钥不存在")
        return
    
    print(f"API密钥使用统计:")
    print(f"   名称: {key_info['key_name']}")
    print(f"   创建时间: {key_info['created_at']}")
    print(f"   最后使用: {key_info['last_used'] or '从未使用'}")
    print(f"   总使用次数: {key_info['usage_count']}")
    print(f"   使用限制: {key_info['usage_limit'] if key_info['usage_limit'] > 0 else '无限制'}")
    print(f"   今日使用: {key_info['daily_usage']} / {key_info['daily_limit']}")
    print(f"   速率限制: {key_info['rate_limit']} 次/分钟")


def main():
    parser = argparse.ArgumentParser(description="API密钥管理工具")
    parser.add_argument("--db-path", default="musicgen_api.db", help="数据库文件路径")
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 创建密钥
    create_parser = subparsers.add_parser("create", help="创建新的API密钥")
    create_parser.add_argument("--name", required=True, help="密钥名称")
    create_parser.add_argument("--usage-limit", type=int, default=-1, help="使用次数限制（-1表示无限制）")
    create_parser.add_argument("--daily-limit", type=int, default=100, help="每日使用限制")
    
    # 列出密钥
    list_parser = subparsers.add_parser("list", help="列出所有API密钥")
    
    # 禁用密钥
    deactivate_parser = subparsers.add_parser("deactivate", help="禁用API密钥")
    deactivate_parser.add_argument("--api-key", required=True, help="要禁用的API密钥")
    
    # 显示使用统计
    usage_parser = subparsers.add_parser("usage", help="显示API密钥使用统计")
    usage_parser.add_argument("--api-key", required=True, help="API密钥")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "create":
            create_key(args)
        elif args.command == "list":
            list_keys(args)
        elif args.command == "deactivate":
            deactivate_key(args)
        elif args.command == "usage":
            show_usage(args)
    except Exception as e:
        print(f"操作失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 