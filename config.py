import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """配置类"""
    # Whisper模型配置
    WHISPER_MODELS = {
        'tiny': 'tiny',
        'base': 'base',
        'small': 'small',
        'medium': 'medium',
        'large': 'large',
        'large-v2': 'large-v2',
        'large-v3': 'large-v3'
    }

    # 设备配置
    DEVICE = 'cuda'  # 默认使用cuda
    CUDA_VERSION = '13.0'  # CUDA版本

    # 服务器配置
    HOST = '0.0.0.0'
    PORT = 7860
    DEBUG = False

    # 文件上传配置
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg', 'webm'}

    # 文本修正配置
    ENABLE_TEXT_CORRECTION = True
    CORRECTION_RULES = {
        '，': ',',
        '。': '.',
        '？': '?',
        '！': '!',
        '；': ';',
        '：': ':',
        '“': '"',
        '”': '"',
        '‘': "'",
        '’': "'",
        '【': '[',
        '】': ']',
        '（': '(',
        '）': ')',
        '《': '<',
        '》': '>'
    }
    # 说话人识别配置
    ENABLE_SPEAKER_DIARIZATION = True
    # HuggingFace Token (用于pyannote模型)
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

    @classmethod
    def get_device(cls) -> str:
        """获取可用的设备"""
        if cls.DEVICE == 'cuda':
            try:
                import torch
                if torch.cuda.is_available():
                    # 检查CUDA版本
                    cuda_version = torch.version.cuda
                    if cuda_version:
                        print(f"检测到CUDA版本: {cuda_version}")
                    return 'cuda'
                else:
                    print("CUDA不可用，将使用CPU")
                    return 'cpu'
            except Exception as e:
                print(f"CUDA检测失败: {e}, 将使用CPU")
                return 'cpu'
        return 'cpu'

    @classmethod
    def create_upload_folder(cls):
        """创建上传文件夹"""
        if not os.path.exists(cls.UPLOAD_FOLDER):
            os.makedirs(cls.UPLOAD_FOLDER)
