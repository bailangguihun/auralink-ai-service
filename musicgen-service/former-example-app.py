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

from qwen_vl_utils import process_vision_info

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 公共音频输出目录
PUBLIC_AUDIO_FOLDER = '../VMM-frontend/project/public/audios'
os.makedirs(PUBLIC_AUDIO_FOLDER, exist_ok=True)

# 临时上传目录
UPLOAD_FOLDER = './temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# GPU分配策略
GPU_MAPPING = {
    'qwen': 1,  # Qwen2.5-VL 放在 GPU 1
    'small': 2,  # small 模型放在 GPU 2
    'medium': 2,  # medium 模型放在 GPU 2
    'large': 3  # large 模型放在 GPU 3
}

# 全局模型实例
qwen_model = None
qwen_processor = None
musicgen_models = {}  # 存储不同规模的模型
musicgen_processor = None
models_initialized = False
available_models = []


def initialize_models():
    """初始化模型"""
    global qwen_model, qwen_processor, musicgen_models, musicgen_processor, models_initialized, available_models
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
            "./checkpoint-400-merged-new",  # 或 "./models/Qwen2.5-VL-7B-Instruct"
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(qwen_device)

        logger.info("正在初始化视觉模型处理器")
        qwen_processor = AutoProcessor.from_pretrained("./checkpoint-400-merged-new")

        # 初始化音乐生成模型处理器
        logger.info("正在初始化音乐生成模型处理器")
        musicgen_processor = AutoProcessor.from_pretrained("./models/musicgen-stereo-large")

        # 加载不同规模的MusicGen模型
        model_paths = {
            'small': "./models/musicgen-stereo-small",
            'medium': "./models/musicgen-stereo-medium",
            'large': "./models/musicgen-stereo-large"
        }

        available_models = []

        # 尝试加载small模型
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
            logger.warning(f"加载small规模模型失败: {e}")

        # 尝试加载medium模型
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
            logger.warning(f"加载medium规模模型失败: {e}")

        # 尝试加载large模型
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
            logger.warning(f"加载large规模模型失败: {e}")

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
    if model_size not in musicgen_models:
        raise ValueError(f"模型规模 {model_size} 不可用，可用模型: {available_models}")

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
        "available_models": available_models
    })


@app.route('/models', methods=['GET'])
def get_models():
    """获取可用模型列表"""
    return jsonify({
        "available": models_initialized,
        "models": available_models
    })


@app.route('/describe_image', methods=['POST'])
def describe_image_api():
    """仅获取图像描述接口"""
    if not models_initialized:
        return jsonify({
            "success": False,
            "message": "模型尚未初始化完成，请稍后再试"
        }), 503

    try:
        data = request.json
        image_data = data.get('image')

        # 处理图像数据
        if image_data.startswith('data:image'):
            # 如果是base64编码的图片，解码
            format, imgstr = image_data.split(';base64,')
            image_bytes = base64.b64decode(imgstr)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        elif os.path.exists(image_data):
            # 如果是文件路径
            image = Image.open(image_data).convert('RGB')
        else:
            return jsonify({
                "success": False,
                "message": "无效的图像数据"
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
            "message": error_msg
        }), 500


@app.route('/generate', methods=['POST'])
def generate_music_api():
    """生成音乐接口"""
    if not models_initialized:
        return jsonify({
            "success": False,
            "message": "模型尚未初始化完成，请稍后再试"
        }), 503

    try:
        data = request.json
        image_data = data.get('image')
        model_size = data.get('modelSize', 'large').lower()  # 默认使用large模型
        duration = int(data.get('duration', 30))

        # 检查模型是否可用
        if model_size not in available_models:
            return jsonify({
                "success": False,
                "message": f"请求的模型规模 {model_size} 不可用，可用模型: {available_models}"
            }), 400

        # 处理图像数据
        if image_data.startswith('data:image'):
            # 如果是base64编码的图片，解码
            format, imgstr = image_data.split(';base64,')
            image_bytes = base64.b64decode(imgstr)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        elif os.path.exists(image_data):
            # 如果是文件路径
            image = Image.open(image_data).convert('RGB')
        else:
            return jsonify({
                "success": False,
                "message": "无效的图像数据"
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
            "message": error_msg
        }), 500


if __name__ == '__main__':
    # 初始化模型
    initialize_models()

    # 启动服务
    app.run(host='0.0.0.0', port=5002, debug=False)