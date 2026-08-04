import os
import sys
from pathlib import Path

THIRD_PARTY_AUDIOCRAFT = Path(
    os.getenv(
        "AUDIOCRAFT_SOURCE_DIR",
        Path(__file__).resolve().parent / "third_party" / "audiocraft",
    )
).resolve()

if str(THIRD_PARTY_AUDIOCRAFT) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_AUDIOCRAFT))


# 设置 Hugging Face 镜像站
os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

from flask import Flask, request, jsonify
import time
import torch
import torch.nn as nn
import logging
from flask_cors import CORS
from PIL import Image
import clip
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from audiocraft.modules.conditioners import ConditioningAttributes
import base64
import io
from urllib.parse import unquote
from urllib.request import Request, urlopen

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
UPLOAD_FOLDER = os.getenv("VMM_UPLOAD_FOLDER", "/data/auralink/uploads/vmm")

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VMM_DEVICE = os.getenv("VMM_DEVICE", DEFAULT_DEVICE)

VMM_MODEL_PATH = os.getenv(
    "VMM_MODEL_PATH",
    "/data/auralink/models/vmm/musicgen-small",
)

VMM_CHECKPOINT_PATH = os.getenv(
    "VMM_CHECKPOINT_PATH",
    "/data/auralink/models/vmm/final_model.pth",
)

VMM_HOST = os.getenv("VMM_HOST", "0.0.0.0")
VMM_PORT = int(os.getenv("VMM_PORT", "5001"))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class VisualMusicGen(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载CLIP视觉编码器
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=VMM_DEVICE)

        # 加载MusicGen模型
        self.musicgen = MusicGen.get_pretrained(VMM_MODEL_PATH)   # Model path is configured by VMM_MODEL_PATH
        self.musicgen.lm = self.musicgen.lm.float()
        self.musicgen.compression_model = self.musicgen.compression_model.float()

        for param in self.musicgen.compression_model.parameters():  # 音频压缩模型参数
            param.requires_grad = False

        if hasattr(self.musicgen.lm, 'dim'):
            hidden_dim = self.musicgen.lm.dim
        else:
            hidden_dim = self.musicgen.lm.emb.weight.shape[1]

        # 定义视觉适配器
        self.visual_adapter = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        for layer in self.visual_adapter:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.02)
                nn.init.constant_(layer.bias, 0.0)

        # 开启条件投影层梯度
        if hasattr(self.musicgen.lm.condition_provider.conditioners['description'], 'output_proj'):
            self.musicgen.lm.condition_provider.conditioners['description'].output_proj.requires_grad_(True)

        self.generation_params = {
            'use_sampling': True,
            'temp': 1.0,
            'top_k': 250,
            'top_p': 0,
            'max_gen_len': int(30 * self.musicgen.frame_rate)
        }

    def forward(self, images, audio_tokens):
        # 提取视觉特征
        with torch.no_grad():
            clip_features = self.clip_model.encode_image(images)

        # 特征适配
        condition = self.visual_adapter(clip_features.float())

        condition_attributes = ConditioningAttributes(
            text={'description': condition},
            # joint_embed={'visual': condition}
        )

        # 生成音乐token
        return self.musicgen.lm.compute_predictions(
            audio_tokens,
            conditions=[condition_attributes],
            keep_only_valid_steps=False
        )

    def generate_music_from_image(self, image, duration=30, progress=True, text_description=None, **gen_kwargs):
        """
        从输入图像生成音乐。
        该函数直接调用musicgen.lm.generate来绕过官方generate接口，
        使用视觉条件（condition）替代文本条件。
        
        参数:
            image: PIL图像或预处理的图像张量
            duration: 生成音频的时长（秒）
            progress: 是否显示进度
            text_description: 可选的文字描述，用于增强生成效果
            **gen_kwargs: 其他生成参数
        """
        # 设置生成参数
        generation_params = self.generation_params.copy()
        generation_params.update(gen_kwargs)
        self.musicgen.set_generation_params(
            duration=duration,
            use_sampling=generation_params['use_sampling'],
            top_k=generation_params['top_k'],
            top_p=generation_params['top_p']
        )

        if not isinstance(image, torch.Tensor):
            image = self.clip_preprocess(image).unsqueeze(0).to(next(self.parameters()).device)
        else:
            image = image.to(next(self.parameters()).device)

        # 提取视觉特征及条件向量
        with torch.no_grad():
            clip_features = self.clip_model.encode_image(image)
            condition = self.visual_adapter(clip_features.float())

        # 进度回调函数
        def progress_callback(generated, total):
            print(f"{generated:6d} / {total:6d}", end='\r')

        callback = progress_callback if progress else None

        condition_attributes = ConditioningAttributes(
            text={'description': text_description},  # 使用提供的文字描述，如果没有则为None
            joint_embed={'visual': condition}
        )

        # 直接调用LM的生成函数：传入 prompt_tokens=None 和我们构造的视觉条件
        with torch.no_grad():
            tokens = self.musicgen.lm.generate(
                conditions=[condition_attributes],
                callback=callback,
                **generation_params
            )
            # 解码生成的tokens为音频
            generated_audio = self.musicgen.generate_audio(tokens)

        return generated_audio

# 全局模型实例
model = None
model_initialized = False


def load_image_from_input(image_data):
    """Load image from data URL, local path, or HTTP(S) URL."""
    if not image_data or not isinstance(image_data, str):
        raise ValueError("invalid image data")

    image_source = image_data.strip()
    if not image_source:
        raise ValueError("invalid image data")

    if image_source.startswith('data:image'):
        try:
            _, imgstr = image_source.split(';base64,', 1)
            image_bytes = base64.b64decode(imgstr)
            return Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as exc:
            raise ValueError(f"invalid base64 image data: {exc}") from exc

    if image_source.startswith('http://') or image_source.startswith('https://'):
        try:
            req = Request(image_source, headers={'User-Agent': 'Mozilla/5.0'})
            with urlopen(req, timeout=20) as response:
                image_bytes = response.read()
            return Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as exc:
            raise ValueError(f"failed to load image from url: {exc}") from exc

    local_path = unquote(image_source)
    if os.path.exists(local_path):
        try:
            return Image.open(local_path).convert('RGB')
        except Exception as exc:
            raise ValueError(f"failed to read image file: {exc}") from exc

    raise ValueError("invalid image data")


def initialize_model():
    """初始化模型"""
    global model, model_initialized
    try:
        logger.info("开始加载VMM模型...")
        device = VMM_DEVICE
        model = VisualMusicGen().to(device)

        # 加载预训练权重
        checkpoint_path = VMM_CHECKPOINT_PATH
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            logger.info(f"已加载预训练权重：{checkpoint_path}")
        else:
            logger.warning(f"预训练权重文件不存在: {checkpoint_path}")

        model_initialized = True
        logger.info("VMM模型加载完成")
    except Exception as e:
        logger.exception(f"初始化VMM模型失败: {e}")
        model_initialized = False


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "ok",
        "model_ready": model_initialized
    })


@app.route('/models', methods=['GET'])
def get_models():
    """获取可用模型列表"""
    return jsonify({
        "available": model_initialized,
        "models": ["small"] if model_initialized else []
    })


@app.route('/generate', methods=['POST'])
def generate_music():
    """Music generation endpoint."""
    if not model_initialized:
        return jsonify({
            "success": False,
            "message": "model is not initialized"
        }), 503

    try:
        data = request.get_json(silent=True) or {}
        image_data = data.get('image') or data.get('imageUrl')
        duration = int(data.get('duration', 30))

        try:
            image = load_image_from_input(image_data)
        except ValueError as image_error:
            return jsonify({
                "success": False,
                "message": str(image_error)
            }), 400

        logger.info(f"Start generating music, duration: {duration}s")

        with torch.no_grad():
            generated_audio = model.generate_music_from_image(image, duration=duration, progress=True)

        generated_audio = generated_audio.squeeze(0)
        timestamp = int(time.time() * 1000)
        output_filename = f"vmm_generated_{timestamp}.wav"
        output_path = os.path.join(PUBLIC_AUDIO_FOLDER, output_filename)

        audio_write(
            output_path.replace('.wav', ''),
            generated_audio.cpu(),
            model.musicgen.sample_rate,
            strategy="loudness",
            loudness_compressor=True
        )

        return jsonify({
            "success": True,
            "fileName": output_filename,
            "full_path": output_path,
            "message": "music generated successfully"
        })

    except Exception as e:
        error_msg = f"music generation failed: {str(e)}"
        logger.exception(error_msg)
        return jsonify({
            "success": False,
            "message": error_msg
        }), 500

@app.route('/api/generate_with_image', methods=['POST'])
def generate_with_image():
    """Generate music from image only."""
    try:
        data = request.get_json(silent=True) or {}
        image_data = data.get('image') or data.get('imageUrl')
        duration = int(data.get('duration', 30))

        try:
            image = load_image_from_input(image_data)
        except ValueError as image_error:
            return jsonify({
                "success": False,
                "message": str(image_error)
            }), 400

        logger.info(f"Start generating music, duration: {duration}s")
        global model, model_initialized
        if not model_initialized:
            model = VisualMusicGen()
            model.eval()

        with torch.no_grad():
            generated_audio = model.generate_music_from_image(
                image, duration=duration, progress=True, text_description="Chinese traditional music"
            )

        generated_audio = generated_audio.squeeze(0)
        timestamp = int(time.time() * 1000)
        output_filename = f"vmm_generated_{timestamp}.wav"
        output_path = os.path.join(PUBLIC_AUDIO_FOLDER, output_filename)

        audio_write(
            output_path.replace('.wav', ''),
            generated_audio.cpu(),
            model.musicgen.sample_rate,
            strategy="loudness",
            loudness_compressor=True
        )

        return jsonify({
            "success": True,
            "fileName": output_filename,
            "full_path": output_path,
            "message": "music generated successfully"
        })

    except Exception as e:
        logger.error(f"music generation failed: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"music generation failed: {str(e)}"
        }), 500

@app.route('/api/generate_with_image_and_text', methods=['POST'])
def generate_with_image_and_text():
    """Generate music from image and text."""
    try:
        data = request.get_json(silent=True) or {}
        image_data = data.get('image') or data.get('imageUrl')
        duration = int(data.get('duration', 30))

        try:
            image = load_image_from_input(image_data)
        except ValueError as image_error:
            return jsonify({
                "success": False,
                "message": str(image_error)
            }), 400

        logger.info(f"Start generating music, duration: {duration}s")
        text_description = data.get('text_description', '')
        if not text_description.strip():
            return jsonify({
                "success": False,
                "message": "missing text_description"
            }), 400

        global model, model_initialized
        if not model_initialized:
            model = VisualMusicGen()
            model.eval()

        with torch.no_grad():
            generated_audio = model.generate_music_from_image(
                image, duration=duration, progress=True, text_description=text_description
            )

        generated_audio = generated_audio.squeeze(0)
        timestamp = int(time.time() * 1000)
        output_filename = f"vmm_generated_{timestamp}.wav"
        output_path = os.path.join(PUBLIC_AUDIO_FOLDER, output_filename)

        audio_write(
            output_path.replace('.wav', ''),
            generated_audio.cpu(),
            model.musicgen.sample_rate,
            strategy="loudness",
            loudness_compressor=True
        )

        return jsonify({
            "success": True,
            "fileName": output_filename,
            "full_path": output_path,
            "message": "music generated successfully"
        })

    except Exception as e:
        logger.error(f"music generation failed: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"music generation failed: {str(e)}"
        }), 500

if __name__ == '__main__':
    # 初始化模型
    initialize_model()

    # 启动服务
    app.run(host=VMM_HOST, port=VMM_PORT, debug=False)
