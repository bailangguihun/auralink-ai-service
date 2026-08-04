"""
数据库管理模块
"""
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class APIKeyManager:
    """API密钥管理器"""
    
    def __init__(self, db_path: str = "musicgen_api.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 创建API密钥表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_hash TEXT UNIQUE NOT NULL,
                    key_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    usage_limit INTEGER DEFAULT -1,  -- -1表示无限制
                    usage_count INTEGER DEFAULT 0,
                    daily_limit INTEGER DEFAULT 100,
                    rate_limit INTEGER DEFAULT 10  -- 每分钟限制
                )
            """)
            
            # 创建使用记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS usage_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_hash TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_name TEXT NOT NULL,
                    duration INTEGER NOT NULL,
                    prompt TEXT,
                    success BOOLEAN DEFAULT 1,
                    error_message TEXT,
                    FOREIGN KEY (key_hash) REFERENCES api_keys (key_hash)
                )
            """)
            
            # 创建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_key_hash ON api_keys(key_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_logs(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_usage_key_hash ON usage_logs(key_hash)")
            
            conn.commit()
            logger.info("数据库初始化完成")
    
    def generate_api_key(self, name: str, usage_limit: int = -1, daily_limit: int = 100) -> str:
        """生成新的API密钥"""
        # 生成32位随机密钥
        api_key = f"mgapi_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO api_keys (key_hash, key_name, usage_limit, daily_limit)
                VALUES (?, ?, ?, ?)
            """, (key_hash, name, usage_limit, daily_limit))
            conn.commit()
        
        logger.info(f"生成新的API密钥: {name}")
        return api_key
    
    def validate_api_key(self, api_key: str) -> Tuple[bool, Optional[str]]:
        """验证API密钥"""
        if not api_key.startswith("mgapi_"):
            return False, "无效的API密钥格式"
        
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT key_name, is_active, usage_limit, usage_count, daily_limit
                FROM api_keys WHERE key_hash = ?
            """, (key_hash,))
            
            result = cursor.fetchone()
            if not result:
                return False, "API密钥不存在"
            
            key_name, is_active, usage_limit, usage_count, daily_limit = result
            
            if not is_active:
                return False, "API密钥已被禁用"
            
            # 检查总使用限制
            if usage_limit > 0 and usage_count >= usage_limit:
                return False, "API密钥使用次数已达上限"
            
            # 检查每日使用限制
            daily_count = self.get_daily_usage_count(key_hash)
            if daily_count >= daily_limit:
                return False, "API密钥今日使用次数已达上限"
            
            # 检查速率限制
            recent_count = self.get_recent_usage_count(key_hash, minutes=1)
            cursor.execute("SELECT rate_limit FROM api_keys WHERE key_hash = ?", (key_hash,))
            rate_limit = cursor.fetchone()[0]
            
            if recent_count >= rate_limit:
                return False, "请求过于频繁，请稍后再试"
        
        return True, None
    
    def get_daily_usage_count(self, key_hash: str) -> int:
        """获取今日使用次数"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM usage_logs 
                WHERE key_hash = ? AND date(timestamp) = ? AND success = 1
            """, (key_hash, today))
            
            return cursor.fetchone()[0]
    
    def get_recent_usage_count(self, key_hash: str, minutes: int = 1) -> int:
        """获取最近N分钟的使用次数"""
        since_time = datetime.now() - timedelta(minutes=minutes)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM usage_logs 
                WHERE key_hash = ? AND timestamp > ? AND success = 1
            """, (key_hash, since_time))
            
            return cursor.fetchone()[0]
    
    def log_usage(self, api_key: str, model_name: str, duration: int, 
                  prompt: str, success: bool = True, error_message: str = None):
        """记录使用日志"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 记录使用日志
            cursor.execute("""
                INSERT INTO usage_logs (key_hash, model_name, duration, prompt, success, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (key_hash, model_name, duration, prompt, success, error_message))
            
            # 更新API密钥使用统计
            if success:
                cursor.execute("""
                    UPDATE api_keys 
                    SET usage_count = usage_count + 1, last_used = CURRENT_TIMESTAMP
                    WHERE key_hash = ?
                """, (key_hash,))
            
            conn.commit()
    
    def get_key_info(self, api_key: str) -> Optional[dict]:
        """获取API密钥信息"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT key_name, created_at, last_used, usage_count, usage_limit, daily_limit, rate_limit
                FROM api_keys WHERE key_hash = ?
            """, (key_hash,))
            
            result = cursor.fetchone()
            if result:
                key_name, created_at, last_used, usage_count, usage_limit, daily_limit, rate_limit = result
                daily_usage = self.get_daily_usage_count(key_hash)
                
                return {
                    "key_name": key_name,
                    "created_at": created_at,
                    "last_used": last_used,
                    "usage_count": usage_count,
                    "usage_limit": usage_limit,
                    "daily_usage": daily_usage,
                    "daily_limit": daily_limit,
                    "rate_limit": rate_limit
                }
        
        return None
    
    def list_api_keys(self) -> List[dict]:
        """列出所有API密钥"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT key_hash, key_name, created_at, last_used, is_active, 
                       usage_count, usage_limit, daily_limit, rate_limit
                FROM api_keys ORDER BY created_at DESC
            """)
            
            keys = []
            for row in cursor.fetchall():
                key_hash, key_name, created_at, last_used, is_active, usage_count, usage_limit, daily_limit, rate_limit = row
                daily_usage = self.get_daily_usage_count(key_hash)
                
                keys.append({
                    "key_name": key_name,
                    "created_at": created_at,
                    "last_used": last_used,
                    "is_active": bool(is_active),
                    "usage_count": usage_count,
                    "usage_limit": usage_limit,
                    "daily_usage": daily_usage,
                    "daily_limit": daily_limit,
                    "rate_limit": rate_limit
                })
            
            return keys
    
    def deactivate_key(self, api_key: str) -> bool:
        """禁用API密钥"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE api_keys SET is_active = 0 WHERE key_hash = ?", (key_hash,))
            conn.commit()
            
            return cursor.rowcount > 0 