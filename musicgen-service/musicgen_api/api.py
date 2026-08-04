"""
FastAPI服务器
"""
import logging
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import time

from .config import load_config, APIConfig
from .database import APIKeyManager
from .models import MusicGenModelManager, GenerationRequest, GenerationResponse

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="MusicGen API Server",
    description="基于AudioCraft MusicGen的音乐生成API服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
config: APIConfig = None
api_key_manager: APIKeyManager = None
model_manager: MusicGenModelManager = None
security = HTTPBearer()


class MusicGenerationRequest(BaseModel):
    """音乐生成请求模型"""
    prompt: str = Field(..., description="音乐生成的文本提示")
    model: str = Field("medium", description="使用的模型 (small/medium/large)")
    duration: float = Field(8.0, ge=1.0, le=60.0, description="生成音乐的时长（秒）")
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="生成的随机性")
    top_k: int = Field(250, ge=1, le=1000, description="Top-k采样参数")
    top_p: float = Field(0.0, ge=0.0, le=1.0, description="Top-p采样参数")
    cfg_coef: float = Field(3.0, ge=1.0, le=10.0, description="分类器引导系数")
    use_sampling: bool = Field(True, description="是否使用采样")
    two_step_cfg: bool = Field(False, description="是否使用两步分类器引导")


class APIKeyCreateRequest(BaseModel):
    """API密钥创建请求"""
    name: str = Field(..., description="API密钥名称")
    usage_limit: int = Field(-1, description="使用次数限制，-1表示无限制")
    daily_limit: int = Field(100, description="每日使用限制")


class APIResponse(BaseModel):
    """API响应基类"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """验证API密钥"""
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少API密钥"
        )
    
    api_key = credentials.credentials
    is_valid, error_message = api_key_manager.validate_api_key(api_key)
    
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error_message
        )
    
    return api_key


async def verify_admin_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """验证管理员认证token"""
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少管理员认证token"
        )
    
    token = credentials.credentials
    if token != config.admin_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无效的管理员认证token"
        )
    
    return token


@app.on_event("startup")
async def startup_event():
    """启动事件"""
    global config, api_key_manager, model_manager
    
    # 加载配置
    config = load_config("config.yaml")
    logger.info("配置加载完成")
    logger.info(f"预加载模型列表: {config.preload_models}")
    logger.info(f"预加载开关: {config.preload_on_startup}")
    
    # 初始化API密钥管理器
    api_key_manager = APIKeyManager()
    logger.info("API密钥管理器初始化完成")
    
    # 初始化模型管理器
    model_manager = MusicGenModelManager(config.__dict__)
    logger.info("模型管理器初始化完成")
    
    # 预加载模型（如果启用）
    if config.preload_on_startup and config.preload_models:
        logger.info("开始预加载模型...")
        try:
            preload_results = model_manager.preload_models(config.preload_models)
            
            # 报告预加载结果
            successful_models = [name for name, success in preload_results.items() if success]
            failed_models = [name for name, success in preload_results.items() if not success]
            
            if successful_models:
                logger.info(f"成功预加载模型: {successful_models}")
            if failed_models:
                logger.warning(f"预加载失败的模型: {failed_models}")
                logger.warning("模型将在首次使用时加载（可能会有延迟）")
        except Exception as e:
            logger.error(f"预加载过程出现异常: {e}")
            logger.warning("跳过预加载，模型将在首次使用时加载")
    else:
        logger.info("跳过模型预加载 (配置已禁用或无指定模型)")
    
    # 创建默认API密钥（仅用于演示）
    try:
        demo_key = api_key_manager.generate_api_key("演示密钥", daily_limit=50)
        logger.info(f"演示API密钥已创建: {demo_key}")
    except Exception as e:
        logger.warning(f"创建演示API密钥失败: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """关闭事件"""
    if model_manager:
        model_manager.cleanup()
    logger.info("服务器关闭完成")


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """添加处理时间头部"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.get("/", response_model=APIResponse)
async def root():
    """根端点"""
    return APIResponse(
        success=True,
        message="MusicGen API Server 运行正常",
        data={
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health"
        }
    )


@app.get("/health")
async def health_check():
    """健康检查"""
    model_info = model_manager.get_model_info()
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "models": model_info
    }


@app.post("/generate", response_model=APIResponse)
async def generate_music(
    request: MusicGenerationRequest,
    api_key: str = Depends(verify_api_key)
):
    """生成音乐"""
    try:
        # 创建生成请求
        gen_request = GenerationRequest(
            prompt=request.prompt,
            model_name=request.model,
            duration=request.duration,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            cfg_coef=request.cfg_coef,
            use_sampling=request.use_sampling,
            two_step_cfg=request.two_step_cfg
        )
        
        # 生成音乐
        response = model_manager.generate_music(gen_request)
        
        # 记录使用日志
        api_key_manager.log_usage(
            api_key=api_key,
            model_name=request.model,
            duration=int(request.duration),
            prompt=request.prompt,
            success=response.success,
            error_message=response.error_message
        )
        
        if response.success:
            return APIResponse(
                success=True,
                message="音乐生成成功",
                data={
                    "audio_data": response.audio_data,
                    "sample_rate": response.sample_rate,
                    "duration": response.duration,
                    "model_used": response.model_used,
                    "prompt": request.prompt
                }
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=response.error_message
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"生成音乐时发生错误: {str(e)}")
        
        # 记录失败日志
        api_key_manager.log_usage(
            api_key=api_key,
            model_name=request.model,
            duration=int(request.duration),
            prompt=request.prompt,
            success=False,
            error_message=str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"服务器内部错误: {str(e)}"
        )


@app.get("/models", response_model=APIResponse)
async def get_models(api_key: str = Depends(verify_api_key)):
    """获取可用模型信息"""
    model_info = model_manager.get_model_info()
    return APIResponse(
        success=True,
        message="模型信息获取成功",
        data=model_info
    )


@app.get("/usage", response_model=APIResponse)
async def get_usage_info(api_key: str = Depends(verify_api_key)):
    """获取API密钥使用信息"""
    key_info = api_key_manager.get_key_info(api_key)
    if key_info:
        return APIResponse(
            success=True,
            message="使用信息获取成功",
            data=key_info
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API密钥信息未找到"
        )


@app.post("/admin/keys", response_model=APIResponse)
async def create_api_key(
    request: APIKeyCreateRequest,
    admin_token: str = Depends(verify_admin_token)
):
    """创建新的API密钥（管理员功能）"""
    try:
        new_key = api_key_manager.generate_api_key(
            name=request.name,
            usage_limit=request.usage_limit,
            daily_limit=request.daily_limit
        )
        
        return APIResponse(
            success=True,
            message="API密钥创建成功",
            data={
                "api_key": new_key,
                "name": request.name,
                "usage_limit": request.usage_limit,
                "daily_limit": request.daily_limit
            }
        )
    except Exception as e:
        logger.error(f"创建API密钥时发生错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建API密钥失败: {str(e)}"
        )


@app.get("/admin/keys", response_model=APIResponse)
async def list_api_keys(admin_token: str = Depends(verify_admin_token)):
    """列出所有API密钥（管理员功能）"""
    try:
        keys = api_key_manager.list_api_keys()
        return APIResponse(
            success=True,
            message="API密钥列表获取成功",
            data={"keys": keys}
        )
    except Exception as e:
        logger.error(f"获取API密钥列表时发生错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取API密钥列表失败: {str(e)}"
        )


class PreloadRequest(BaseModel):
    """预加载请求模型"""
    models: List[str] = Field(default=None, description="要预加载的模型列表，为空则使用配置文件中的默认列表")


@app.post("/admin/preload", response_model=APIResponse)
async def preload_models_manual(
    request: PreloadRequest = None,
    admin_token: str = Depends(verify_admin_token)
):
    """手动预加载模型（管理员功能）"""
    try:
        # 确定要预加载的模型列表
        models_to_preload = request.models if request and request.models else config.preload_models
        
        if not models_to_preload:
            return APIResponse(
                success=False,
                message="没有指定要预加载的模型"
            )
        
        logger.info(f"手动触发预加载: {models_to_preload}")
        preload_results = model_manager.preload_models(models_to_preload)
        
        # 统计结果
        successful_models = [name for name, success in preload_results.items() if success]
        failed_models = [name for name, success in preload_results.items() if not success]
        
        return APIResponse(
            success=True,
            message="预加载完成",
            data={
                "requested_models": models_to_preload,
                "successful_models": successful_models,
                "failed_models": failed_models,
                "success_count": len(successful_models),
                "total_count": len(models_to_preload)
            }
        )
        
    except Exception as e:
        logger.error(f"手动预加载时发生错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"预加载失败: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    logger.error(f"未处理的异常: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"success": False, "message": "服务器内部错误", "detail": str(exc)}
    ) 