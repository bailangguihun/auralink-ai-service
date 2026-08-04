"""
MusicGen模型包装器 - 使用HuggingFace Transformers
"""
import os
import torch
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import io
import base64
import numpy as np
import wave

# 设置环境变量使用镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logger = logging.getLogger(__name__)


@dataclass
class GenerationRequest:
    """音乐生成请求"""
    prompt: str
    model_name: str = "medium"
    duration: float = 8.0
    temperature: float = 1.0
    top_k: int = 250
    top_p: float = 0.0
    cfg_coef: float = 3.0
    use_sampling: bool = True
    two_step_cfg: bool = False


@dataclass
class GenerationResponse:
    """音乐生成响应"""
    success: bool
    audio_data: Optional[str] = None  # base64编码的音频数据
    sample_rate: int = 32000
    duration: float = 0.0
    error_message: Optional[str] = None
    model_used: Optional[str] = None


class MusicGenModelManager:
    """MusicGen模型管理器 - 使用HuggingFace Transformers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models: Dict[str, Any] = {}
        self.processors: Dict[str, Any] = {}
        self.model_configs = config.get("models", {})
        
        # 检查CUDA可用性
        self.device_count = torch.cuda.device_count()
        if self.device_count == 0:
            logger.warning("未检测到CUDA设备，将使用CPU运行（速度较慢）")
            self.device = "cpu"
        else:
            logger.info(f"检测到 {self.device_count} 个CUDA设备")
            self.device = "cuda"
    
    def load_model(self, model_name: str) -> bool:
        """使用HuggingFace Transformers加载指定的模型"""
        try:
            from transformers import MusicgenForConditionalGeneration, AutoProcessor
        except ImportError as e:
            logger.error(f"Transformers导入失败: {e}")
            logger.error("请确保已正确安装 transformers")
            return False
        
        if model_name in self.models:
            logger.info(f"使用已预加载的模型: {model_name}")
            return True
        
        if model_name not in self.model_configs:
            logger.error(f"未知的模型名称: {model_name}")
            return False
        
        try:
            model_config = self.model_configs[model_name]
            logger.info(f"正在加载模型: {model_name} ({model_config.size})")
            
            # 确定设备
            if self.device == "cuda" and model_config.gpu_id < self.device_count:
                device = f"cuda:{model_config.gpu_id}"
            else:
                device = "cpu"
            
            logger.info(f"使用设备: {device}")
            
            # 按照former-example-app.py的方式加载模型
            model_path = f"./models/musicgen-stereo-{model_name}"
            if os.path.exists(model_path):
                logger.info(f"从本地路径加载模型: {model_path}")
                model = MusicgenForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16
                ).to(device)
                processor = AutoProcessor.from_pretrained(model_path)
            else:
                logger.info(f"从HuggingFace加载模型: {model_config.name}")
                # 使用transformers模型名称格式
                hf_model_name = model_config.name.replace("facebook/musicgen-", "facebook/musicgen-stereo-")
                model = MusicgenForConditionalGeneration.from_pretrained(
                    hf_model_name,
                    torch_dtype=torch.float16
                ).to(device)
                processor = AutoProcessor.from_pretrained(hf_model_name)
            
            self.models[model_name] = model
            self.processors[model_name] = processor
            logger.info(f"✅ 模型 {model_name} 加载完成，设备: {device}")
            return True
            
        except Exception as e:
            logger.error(f"加载模型 {model_name} 失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def unload_model(self, model_name: str):
        """卸载指定的模型"""
        if model_name in self.models:
            del self.models[model_name]
            del self.processors[model_name]
            torch.cuda.empty_cache()
            logger.info(f"模型 {model_name} 已卸载")
    
    def generate_music(self, request: GenerationRequest) -> GenerationResponse:
        """按照former-example-app.py的方式生成音乐"""
        try:
            # 验证模型名称
            if request.model_name not in self.model_configs:
                return GenerationResponse(
                    success=False,
                    error_message=f"不支持的模型: {request.model_name}"
                )
            
            # 验证生成时长
            max_duration = self.model_configs[request.model_name].max_duration
            if request.duration > max_duration:
                return GenerationResponse(
                    success=False,
                    error_message=f"请求的时长 ({request.duration}s) 超过模型最大限制 ({max_duration}s)"
                )
            
            # 加载模型
            if not self.load_model(request.model_name):
                return GenerationResponse(
                    success=False,
                    error_message=f"加载模型 {request.model_name} 失败"
                )
            
            model = self.models[request.model_name]
            processor = self.processors[request.model_name]
            
            # 获取设备
            model_config = self.model_configs[request.model_name]
            if self.device == "cuda" and model_config.gpu_id < self.device_count:
                device = f"cuda:{model_config.gpu_id}"
            else:
                device = "cpu"
            
            logger.info(f"开始生成音乐: prompt='{request.prompt}', duration={request.duration}s")
            
            # 按照former-example-app.py的方式生成音乐
            with torch.no_grad():
                # 设定生成时长与 token 数量
                max_duration = request.duration  # 秒
                max_new_tokens = int(max_duration * 50)  # 大约50 tokens/秒
                
                # 准备输入
                inputs = processor(
                    text=request.prompt,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                
                # 生成音频
                audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)
                
                # 获取采样率和音频数据
                sample_rate = model.config.audio_encoder.sampling_rate
                audio_data = audio_values[0, 0].cpu().numpy().astype("float32")
                
                # 转换为base64编码
                audio_base64 = self._audio_to_base64_numpy(audio_data, sample_rate)
                
                logger.info(f"音乐生成完成: duration={request.duration}s, sample_rate={sample_rate}")
                
                return GenerationResponse(
                    success=True,
                    audio_data=audio_base64,
                    sample_rate=sample_rate,
                    duration=request.duration,
                    model_used=request.model_name
                )
                
        except Exception as e:
            logger.error(f"生成音乐失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return GenerationResponse(
                success=False,
                error_message=f"生成失败: {str(e)}"
            )
    
    def _audio_to_base64_numpy(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """使用numpy和wave模块转换音频为base64"""
        try:
            # 确保音频数据在合理范围内
            audio_data = np.clip(audio_data, -1.0, 1.0)
            
            # 转换为16位整数
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # 创建WAV文件
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # 单声道
                wav_file.setsampwidth(2)  # 16位 = 2字节
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            # 编码为base64
            buffer.seek(0)
            audio_bytes = buffer.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            return audio_base64
            
        except Exception as e:
            logger.error(f"音频转换失败: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            "loaded_models": list(self.models.keys()),
            "available_models": list(self.model_configs.keys()),
            "device_count": self.device_count,
            "device": self.device
        }
        
        for model_name, model_config in self.model_configs.items():
            info[f"{model_name}_config"] = {
                "name": model_config.name,
                "size": model_config.size,
                "gpu_id": model_config.gpu_id,
                "memory_requirement": model_config.memory_requirement,
                "max_duration": model_config.max_duration
            }
        
        return info
    
    def cleanup(self):
        """清理资源"""
        for model_name in list(self.models.keys()):
            self.unload_model(model_name)
    
    def preload_models(self, model_names: List[str]) -> Dict[str, bool]:
        """预加载指定的模型"""
        logger.info(f"开始预加载模型: {model_names}")
        results = {}
        
        for model_name in model_names:
            logger.info(f"预加载模型: {model_name}")
            success = self.load_model(model_name)
            results[model_name] = success
            
            if success:
                logger.info(f"✅ 模型 {model_name} 预加载成功")
            else:
                logger.error(f"❌ 模型 {model_name} 预加载失败")
        
        success_count = sum(results.values())
        total_count = len(model_names)
        logger.info(f"预加载完成: {success_count}/{total_count} 个模型成功加载")
        
        return results


# 全局模型管理器实例
_model_manager: Optional[MusicGenModelManager] = None


def get_model_manager(config: Optional[Dict[str, Any]] = None) -> MusicGenModelManager:
    """获取全局模型管理器实例"""
    global _model_manager
    if _model_manager is None:
        if config is None:
            raise ValueError("首次调用需要提供配置")
        _model_manager = MusicGenModelManager(config)
    return _model_manager


def cleanup_models():
    """清理所有模型"""
    global _model_manager
    if _model_manager is not None:
        _model_manager.cleanup()
        _model_manager = None 