from flask import Flask, request, jsonify
import os
import time
import torch
import logging
from flask_cors import CORS
from PIL import Image
import scipy.io.wavfile
import base64
import io
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from qwen_vl_utils import process_vision_info

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 公共音频输出目录
PUBLIC_AUDIO_FOLDER = os.getenv("PUBLIC_AUDIO_FOLDER", "/data/auralink/audio")
os.makedirs(PUBLIC_AUDIO_FOLDER, exist_ok=True)

# 临时上传目录
UPLOAD_FOLDER = os.getenv("VISION_UPLOAD_FOLDER", "/data/auralink/uploads/vision")

QWEN_MODEL_PATH = os.getenv(
    "QWEN_MODEL_PATH",
    "/data/auralink/models/qwen-vl/checkpoint-400-merged-new",
)

VISION_HOST = os.getenv("VISION_HOST", "0.0.0.0")
VISION_PORT = int(os.getenv("VISION_PORT", "5002"))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# GPU分配策略
GPU_MAPPING = {
    'qwen': 3,  # Qwen2.5-VL 放在 GPU 1
    'small': 3,  # small 模型放在 GPU 2
    'medium': 2,  # medium 模型放在 GPU 2
    'large': 2  # large 模型放在 GPU 3
}

# MusicGen 模型加载/启用配置（显存不足时可只启用部分模型）
# - True: 启用并在启动时尝试加载
# - False: 不加载且不可被调用（调用会返回明确错误码）
MUSIC_MODEL_LOADING = {
    # 音乐生成由 5001 端口的 VMM 服务负责；5002 仅加载视觉模型，避免重复占用显存。
    'small': False,
    'medium': False,
    'large': False,
}

# 错误码约定（客户端可据此做稳定处理）
ERROR_CODES = {
    "INVALID_MODEL_SIZE": 2000,     # 请求的模型规模不合法
    "MODEL_DISABLED": 2001,         # 模型在配置中未启用加载
    "MODEL_NOT_LOADED": 2002,       # 模型已启用，但启动时未成功加载/当前不可用
    "MODELS_NOT_READY": 2003,       # 服务模型尚未初始化完成
    "BAD_REQUEST": 2100,            # 请求参数/格式错误
    "INTERNAL_ERROR": 2500,         # 服务内部错误
}

# 全局模型实例
qwen_model = None
qwen_processor = None
musicgen_models = {}  # 存储不同规模的模型
musicgen_processor = None
models_initialized = False
available_models = []
music_model_load_errors = {}  # 记录启用但加载失败的模型原因
REMOTE_IMAGE_TIMEOUT_SEC = 30
MAX_REMOTE_IMAGE_BYTES = 15 * 1024 * 1024


def _get_enabled_music_models():
    """返回配置中启用的音乐模型列表（按 small/medium/large 固定顺序）"""
    order = ['small', 'medium', 'large']
    return [m for m in order if MUSIC_MODEL_LOADING.get(m, False)]


def _is_valid_music_model_size(model_size: str) -> bool:
    return model_size in ('small', 'medium', 'large')


def _is_http_url(value: str) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value.strip())
    return parsed.scheme in ('http', 'https') and bool(parsed.netloc)


def _read_remote_image_bytes(url: str) -> bytes:
    req = Request(
        url,
        headers={
            "User-Agent": "Auralink-Backend/1.0",
            "Accept": "image/*,*/*;q=0.8",
        },
    )
    with urlopen(req, timeout=REMOTE_IMAGE_TIMEOUT_SEC) as resp:
        status = getattr(resp, 'status', 200)
        if status and int(status) >= 400:
            raise ValueError(f"远程图像请求失败: HTTP {status}")
        content_type = str(resp.headers.get("Content-Type", "")).lower()
        payload = resp.read(MAX_REMOTE_IMAGE_BYTES + 1)
        if len(payload) > MAX_REMOTE_IMAGE_BYTES:
            raise ValueError(f"远程图像过大，超过 {MAX_REMOTE_IMAGE_BYTES} 字节限制")
        if content_type and "image" not in content_type:
            # 某些服务端不返回标准类型；如果 PIL 可正常解码，依然允许通过
            logger.warning("远程资源 Content-Type 非 image: %s", content_type)
        return payload


def _parse_image_input(image_input):
    """解析 image/imageUrl 输入，支持 base64、本地路径、HTTP(S) URL。"""
    if not isinstance(image_input, str):
        raise ValueError("图像数据类型无效，必须是字符串")

    value = image_input.strip()
    if not value:
        raise ValueError("图像数据为空")

    if value.startswith('data:image'):
        _, imgstr = value.split(';base64,', 1)
        image_bytes = base64.b64decode(imgstr)
        return Image.open(io.BytesIO(image_bytes)).convert('RGB')

    decoded_value = unquote(value)
    if os.path.exists(value):
        return Image.open(value).convert('RGB')
    if decoded_value != value and os.path.exists(decoded_value):
        return Image.open(decoded_value).convert('RGB')

    if _is_http_url(value):
        remote_bytes = _read_remote_image_bytes(value)
        return Image.open(io.BytesIO(remote_bytes)).convert('RGB')

    raise ValueError("无效的图像数据，需为 base64、本地路径或可访问的 HTTP(S) 图片 URL")


def initialize_models():
    """初始化模型"""
    global qwen_model, qwen_processor, musicgen_models, musicgen_processor, models_initialized, available_models, music_model_load_errors
    try:
        logger.info("开始加载非VMM模型...")

        # 导入必要的库
        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
            MusicgenForConditionalGeneration,
            AutoProcessor
        )
        from qwen_vl_utils import process_vision_info

        # 检查可用的GPU数量
        gpu_count = torch.cuda.device_count()
        logger.info(f"检测到 {gpu_count} 个GPU")

        if gpu_count < 2:
            logger.warning("GPU数量不足，将使用单GPU模式")
            # 调整GPU映射
            for key in GPU_MAPPING:
                GPU_MAPPING[key] = 0

        # 加载Qwen视觉模型
        qwen_device = f'cuda:{GPU_MAPPING["qwen"]}' if torch.cuda.is_available() else 'cpu'
        logger.info(f"正在初始化视觉模型 on {qwen_device}")
        qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_PATH,  # 或 "./models/Qwen2.5-VL-7B-Instruct"
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(qwen_device)

        logger.info("正在初始化视觉模型处理器")
        qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_PATH)

        # 当前 5002 服务仅负责图像描述。所有 MusicGen 模型关闭时，
        # Qwen 加载完成即视为初始化完成，不再重复加载 5001 已有的音乐模型。
        if not any(MUSIC_MODEL_LOADING.values()):
            models_initialized = True
            available_models = []
            logger.info("Vision-only mode: Qwen model loaded; MusicGen is served by port 5001")
            return

        # 加载不同规模的MusicGen模型
        model_paths = {
            'small': "./models/musicgen-stereo-small",
            'medium': "./models/musicgen-stereo-medium",
            'large': "./models/musicgen-stereo-large"
        }

        # 重新初始化（避免进程内重复初始化残留旧模型）
        musicgen_models = {}
        available_models = []
        music_model_load_errors = {}

        enabled_music_models = _get_enabled_music_models()
        logger.info("MusicGen 启用配置: %s", enabled_music_models)

        # 初始化音乐生成模型处理器：优先用“已启用”的任意一个模型目录初始化
        processor_source = enabled_music_models[0] if len(enabled_music_models) > 0 else 'large'
        logger.info("正在初始化音乐生成模型处理器（来源=%s）", processor_source)
        musicgen_processor = AutoProcessor.from_pretrained(model_paths[processor_source])

        # 尝试加载small模型
        if MUSIC_MODEL_LOADING.get('small', False):
            try:
                small_device = f'cuda:{GPU_MAPPING["small"]}' if torch.cuda.is_available() else 'cpu'
                logger.info(f"正在初始化small规模音乐生成模型 on {small_device}")
                musicgen_models['small'] = MusicgenForConditionalGeneration.from_pretrained(
                    model_paths['small'],
                    torch_dtype=torch.float16
                ).to(small_device)
                available_models.append('small')
                logger.info("small规模模型加载成功")
            except Exception as e:
                music_model_load_errors['small'] = str(e)
                logger.warning(f"加载small规模模型失败: {e}")
        else:
            logger.info("small规模模型未启用，跳过加载")

        # 尝试加载medium模型
        if MUSIC_MODEL_LOADING.get('medium', False):
            try:
                medium_device = f'cuda:{GPU_MAPPING["medium"]}' if torch.cuda.is_available() else 'cpu'
                logger.info(f"正在初始化medium规模音乐生成模型 on {medium_device}")
                musicgen_models['medium'] = MusicgenForConditionalGeneration.from_pretrained(
                    model_paths['medium'],
                    torch_dtype=torch.float16
                ).to(medium_device)
                available_models.append('medium')
                logger.info("medium规模模型加载成功")
            except Exception as e:
                music_model_load_errors['medium'] = str(e)
                logger.warning(f"加载medium规模模型失败: {e}")
        else:
            logger.info("medium规模模型未启用，跳过加载")

        # 尝试加载large模型
        if MUSIC_MODEL_LOADING.get('large', False):
            try:
                large_device = f'cuda:{GPU_MAPPING["large"]}' if torch.cuda.is_available() else 'cpu'
                logger.info(f"正在初始化large规模音乐生成模型 on {large_device}")
                musicgen_models['large'] = MusicgenForConditionalGeneration.from_pretrained(
                    model_paths['large'],
                    torch_dtype=torch.float16
                ).to(large_device)
                available_models.append('large')
                logger.info("large规模模型加载成功")
            except Exception as e:
                music_model_load_errors['large'] = str(e)
                logger.warning(f"加载large规模模型失败: {e}")
        else:
            logger.info("large规模模型未启用，跳过加载")

        if len(available_models) > 0:
            models_initialized = True
            logger.info(f"非VMM模型加载完成，可用模型: {available_models}")
        else:
            logger.error("没有成功加载任何模型")
            models_initialized = False

    except Exception as e:
        logger.exception(f"初始化非VMM模型失败: {e}")
        models_initialized = False


def get_description(image, vision_device=None):
    """使用Qwen模型生成图像描述"""
    if vision_device is None:
        vision_device = f'cuda:{GPU_MAPPING["qwen"]}' if torch.cuda.is_available() else 'cpu'

    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")

    # 构造对话消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": """
                Analyze the given Chinese painting and generate a description of the type of music that best matches its style and artistic conception. Consider the following aspects:
                1.Artistic Conception (Atmosphere): What kind of mood or emotional tone should the music convey based on the painting?
                e.g., tranquil, poetic, serene, majestic, meditative, mysterious, solemn
                2.Instruments: Identify key elements in the painting. What traditional Chinese instruments best represent them in the music?
                e.g., guqin (zither), xiao (bamboo flute), pipa (lute), erhu (two-string fiddle), dizi (flute), bianzhong (bronze bells), yunluo (gong chimes), guzheng (zither), war drums
                3.Music Style: What style of music would best align with the artistic style of the painting?
                e.g., traditional Chinese classical, modern Chinese fusion, imperial court music, Jiangnan silk and bamboo ensemble, Daoist meditative music, epic battle music
                4.Rhythm: Should the music have a fast-paced or slow tempo to match the painting's atmosphere?
                e.g., slow and flowing, moderate and balanced, grand and powerful, light and agile
                5.Scene Representation: What is the overall theme of the painting? What musical details should be included to reflect the painting accurately?
                e.g., misty mountain paradise, a boat drifting in Jiangnan rain, desolate desert with the Great Wall, majestic imperial court, serene Buddhist temple in the mountains
                PLEASE RESPONSE IN THIS FORMAT:
                The artistic conception of this music is [XXX]. It should use [XXX] instruments, adopting a [XXX] music style with a [XXX] rhythm, to express a [XXX] scene.
                """},
            ],
        }
    ]

    # 构造文本与多模态输入
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = qwen_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(vision_device)

    logger.info("正在使用视觉模型生成描述")
    with torch.no_grad():
        generated_ids = qwen_model.generate(**inputs, max_new_tokens=128)

    # 剪去输入部分，保留新生成的 token
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = qwen_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    logger.info(f"视觉模型输出: {output_text}")
    return output_text[0]  # 返回字符串


def generate_music(description, model_size='large', duration=30):
    """使用MusicGen模型根据描述生成音乐"""
    if not _is_valid_music_model_size(model_size):
        raise ValueError(f"请求的模型规模不合法: {model_size}，仅支持 small/medium/large")
    if not MUSIC_MODEL_LOADING.get(model_size, False):
        raise ValueError(f"模型 {model_size} 未启用加载，请在 MUSIC_MODEL_LOADING 中开启后重启服务")
    if model_size not in musicgen_models:
        # 已启用但未加载成功
        reason = music_model_load_errors.get(model_size, "未知原因")
        raise RuntimeError(f"模型 {model_size} 已启用但未成功加载: {reason}")

    # 获取对应的模型和设备
    model = musicgen_models[model_size]
    device = f'cuda:{GPU_MAPPING[model_size]}' if torch.cuda.is_available() else 'cpu'

    # 设定生成时长与 token 数量
    max_duration = duration  # 秒
    max_new_tokens = max_duration * 50

    # 准备输入
    inputs = musicgen_processor(
        text=description,
        padding=True,
        return_tensors="pt",
    ).to(device)

    logger.info(f"正在使用 {model_size} 规模的音乐生成模型生成音频")
    with torch.no_grad():
        audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # 获取采样率
    sampling_rate = model.config.audio_encoder.sampling_rate
    timestamp = int(time.time() * 1000)
    output_filename = f"nonvmm_{model_size}_{timestamp}.wav"
    output_path = os.path.join(PUBLIC_AUDIO_FOLDER, output_filename)

    audio_data = audio_values[0, 0].cpu().numpy().astype("float32")
    scipy.io.wavfile.write(output_path, rate=sampling_rate, data=audio_data)
    logger.info(f"音频生成完成: {output_path}")

    return output_filename, output_path


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "ok",
        "model_ready": models_initialized,
        "available_models": available_models,
        "enabled_music_models": _get_enabled_music_models(),
        "music_model_load_errors": music_model_load_errors
    })


@app.route('/models', methods=['GET'])
def get_models():
    """获取可用模型列表"""
    return jsonify({
        "available": models_initialized,
        "models": available_models,
        "enabled_models": _get_enabled_music_models(),
        "disabled_models": [m for m in ('small', 'medium', 'large') if not MUSIC_MODEL_LOADING.get(m, False)],
        "load_errors": music_model_load_errors
    })


@app.route('/describe_image', methods=['POST'])
@app.route('/describe-image', methods=['POST'])
def describe_image_api():
    """仅获取图像描述接口"""
    if qwen_model is None or qwen_processor is None:
        return jsonify({
            "success": False,
            "error_code": ERROR_CODES["MODELS_NOT_READY"],
            "message": "模型尚未初始化完成，请稍后再试"
        }), 503

    try:
        data = request.json or {}
        image_data = data.get('image') or data.get('imageUrl')

        if not image_data:
            return jsonify({
                "success": False,
                "error_code": ERROR_CODES["BAD_REQUEST"],
                "message": "缺少 image/imageUrl 字段"
            }), 400

        try:
            image = _parse_image_input(image_data)
        except Exception as parse_error:
            return jsonify({
                "success": False,
                "error_code": ERROR_CODES["BAD_REQUEST"],
                "message": f"无效的图像数据: {parse_error}"
            }), 400

        logger.info("开始生成图像描述")

        # 获取图像描述
        vision_device = f'cuda:{GPU_MAPPING["qwen"]}' if torch.cuda.is_available() else 'cpu'
        description = get_description(image, vision_device)

        return jsonify({
            "success": True,
            "description": description,
            "message": "图像描述生成成功"
        })

    except Exception as e:
        error_msg = f"生成图像描述失败: {str(e)}"
        logger.exception(error_msg)
        return jsonify({
            "success": False,
            "error_code": ERROR_CODES["INTERNAL_ERROR"],
            "message": error_msg
        }), 500


@app.route('/generate', methods=['POST'])
def generate_music_api():
    """生成音乐接口"""
    if not models_initialized:
        return jsonify({
            "success": False,
            "error_code": ERROR_CODES["MODELS_NOT_READY"],
            "message": "模型尚未初始化完成，请稍后再试"
        }), 503

    try:
        data = request.json
        if not data:
            return jsonify({
                "success": False,
                "error_code": ERROR_CODES["BAD_REQUEST"],
                "message": "请求体为空或不是JSON"
            }), 400

        image_data = data.get('image') or data.get('imageUrl')
        model_size = str(data.get('modelSize', 'large')).lower()  # 默认使用large模型
        duration = int(data.get('duration', 30))

        logger.info(
            "收到音乐生成请求 model_size=%s duration=%s image_provided=%s",
            model_size, duration, bool(image_data)
        )

        if not image_data:
            logger.warning("请求缺少 image 字段")
            return jsonify({
                "success": False,
                "error_code": ERROR_CODES["BAD_REQUEST"],
                "message": "缺少 image/imageUrl 字段，需提供 base64、本地路径或 HTTP(S) 图片 URL"
            }), 400

        # 检查模型规模合法性/启用状态/加载状态
        if not _is_valid_music_model_size(model_size):
            return jsonify({
                "success": False,
                "error_code": ERROR_CODES["INVALID_MODEL_SIZE"],
                "message": f"请求的模型规模不合法: {model_size}，仅支持 small/medium/large"
            }), 400

        if not MUSIC_MODEL_LOADING.get(model_size, False):
            return jsonify({
                "success": False,
                "error_code": ERROR_CODES["MODEL_DISABLED"],
                "message": f"模型 {model_size} 未启用加载（配置 MUSIC_MODEL_LOADING 中为 False），不可被调用"
            }), 400

        if model_size not in available_models:
            reason = music_model_load_errors.get(model_size, "模型未成功加载（可能显存不足或权重路径问题）")
            return jsonify({
                "success": False,
                "error_code": ERROR_CODES["MODEL_NOT_LOADED"],
                "message": f"模型 {model_size} 已启用但当前不可用: {reason}",
                "available_models": available_models
            }), 503

        try:
            image = _parse_image_input(image_data)
        except Exception as parse_error:
            logger.warning("无效的图像数据，无法解析: %s", str(image_data)[:200])
            return jsonify({
                "success": False,
                "error_code": ERROR_CODES["BAD_REQUEST"],
                "message": f"无效的图像数据: {parse_error}"
            }), 400

        logger.info(f"开始生成音乐，模型: {model_size}, 时长: {duration}秒")

        # 1. 获取图像描述
        vision_device = f'cuda:{GPU_MAPPING["qwen"]}' if torch.cuda.is_available() else 'cpu'
        description = get_description(image, vision_device)

        # 2. 根据描述生成音乐
        output_filename, output_path = generate_music(description, model_size, duration)

        return jsonify({
            "success": True,
            "fileName": output_filename,
            "full_path": output_path,
            "description": description,
            "model": model_size,
            "message": "音乐生成成功"
        })

    except Exception as e:
        error_msg = f"生成音乐失败: {str(e)}"
        logger.exception(error_msg)
        return jsonify({
            "success": False,
            "error_code": ERROR_CODES["INTERNAL_ERROR"],
            "message": error_msg
        }), 500


if __name__ == '__main__':
    # 初始化模型
    initialize_models()

    # 启动服务
    app.run(host=VISION_HOST, port=VISION_PORT, debug=False)
