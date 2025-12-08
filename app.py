import json
import os
import re

import whisper
import torchaudio
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import logging
from typing import Dict, Any, Optional

from config import Config
from utils.audio_processor import AudioProcessor
from utils.text_corrector import TextCorrector

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperASR:
    """Whisper语音识别器"""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.current_device = Config.get_device()
        logger.info(f"使用设备: {self.current_device}")

        # 初始化文本修正器
        self.corrector = TextCorrector(
            enable_correction=Config.ENABLE_TEXT_CORRECTION,
            rules=Config.CORRECTION_RULES
        )

        # 初始化多说话人识别模型
        if Config.ENABLE_SPEAKER_DIARIZATION:
            from pyannote.audio import Pipeline

            token=os.getenv("HUGGINGFACE_TOKEN")
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token
            )
            print(f"加载pyannote/speaker-diarization-3.1模型成功")

    def load_model(self, model_size: str) -> Any:
        """加载指定大小的模型"""
        if model_size not in Config.WHISPER_MODELS:
            raise ValueError(f"不支持的模型大小: {model_size}")

        model_name = Config.WHISPER_MODELS[model_size]

        # 如果模型已加载，直接返回
        if model_name in self.models:
            return self.models[model_name]

        try:
            logger.info(f"正在加载模型: {model_name}")
            start_time = time.time()

            # 加载模型
            model = whisper.load_model(
                model_name,
                device=self.current_device,
                download_root="./models"
            )

            load_time = time.time() - start_time
            logger.info(f"模型 {model_name} 加载完成，耗时: {load_time:.2f}秒")

            # 缓存模型
            self.models[model_name] = model
            return model

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def transcribe(
            self,
            audio_path: str,
            model_size: str = "base",
            language: Optional[str] = None,
            task: str = "transcribe"
    ) -> Dict[str, Any]:
        """转录音频文件"""
        try:
            # 加载模型
            model = self.load_model(model_size)

            # 转录选项
            options = {
                "task": task,
                "fp16": False if self.current_device == "cpu" else True,
                "verbose": False
            }

            if language:
                options["language"] = language

            # 执行转录
            logger.info(f"开始转录，模型: {model_size}, 语言: {language or 'auto'}")
            start_time = time.time()

            result = model.transcribe(audio_path, **options)

            transcription_time = time.time() - start_time
            logger.info(f"转录完成，耗时: {transcription_time:.2f}秒")

            # 获取原始文本
            text = result.get("text", "").strip()

            # 应用文本修正
            corrected_text = self.corrector.correct_text(text, language or "zh")
            print(f"文本修正：" + corrected_text)
            # corrected_text = self.corrector.correct_with_ai(corrected_text)
            # print(f"模型修正：" + corrected_text)

            speaker_list=[]
            # 多说话人检测
            if Config.ENABLE_SPEAKER_DIARIZATION:
                waveform, sample_rate = torchaudio.load(audio_path)

                diarization = self.diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate})
                diarization_json=self.diarization_to_json(diarization_text=str(diarization))
                speaker_list=self.find_overlap_and_merge(diarization_json['segments'], result['segments'])

            return {
                "text": corrected_text,
                "original_text": corrected_text,
                "language": result.get("language", ""),
                "duration": result.get("duration", 0),
                "processing_time": transcription_time,
                "model": model_size,
                "speakers": speaker_list
            }

        except Exception as e:
            logger.error(f"转录失败: {e}")
            raise

    def diarization_to_json(self, diarization_text):
        entries = []

        # 使用正则表达式匹配每一行
        pattern = r'\[ (.*?) -->  (.*?)\] (.*?) SPEAKER_(\d+)'

        for line in diarization_text.strip().split('\n'):
            match = re.match(pattern, line.strip())
            if match:
                start, end, text, speaker_id = match.groups()
                entry = {
                    "start": start.strip(),
                    "end": end.strip(),
                    "text": text.strip(),
                    "speaker": f"SPEAKER_{speaker_id}"
                }
                entries.append(entry)

        return {"segments": entries}

    def find_overlap_and_merge(self, arr1, arr2, time_offset=0.1):
        """找出两个数组时间段的重合部分并合并，允许时间偏移"""
        result = []
        for item2 in arr2:
            start2 = item2['start']
            end2 = item2['end']
            matched = False

            for item1 in arr1:
                start1_str = item1['start']
                end1_str = item1['end']

                # 解析带毫秒的时间
                if '.' in start1_str:
                    h1, m1, s1_ms = start1_str.split(':')
                    s1, ms1 = s1_ms.split('.')
                    start1 = float(h1) * 3600 + float(m1) * 60 + float(s1) + float('0.' + ms1)
                else:
                    h1, m1, s1 = start1_str.split(':')
                    start1 = float(h1) * 3600 + float(m1) * 60 + float(s1)

                if '.' in end1_str:
                    h2, m2, s2_ms = end1_str.split(':')
                    s2, ms2 = s2_ms.split('.')
                    end1 = float(h2) * 3600 + float(m2) * 60 + float(s2) + float('0.' + ms2)
                else:
                    h2, m2, s2 = end1_str.split(':')
                    end1 = float(h2) * 3600 + float(m2) * 60 + float(s2)

                if end1 > start2 + time_offset and start1 < end2 - time_offset:
                    speaker_name = None
                    if not speaker_name:
                        speaker_name = item1['speaker']

                    result_item = {
                        'start': start2,
                        'end': end2,
                        'text': item2['text'],
                        'speaker': speaker_name
                    }
                    result.append(result_item)
                    matched = True
                    break

            # 如果仍然没有匹配到，使用默认的说话人
            if not matched and arr1:
                result_item = {
                    'start': start2,
                    'end': end2,
                    'text': item2['text'],
                    'speaker': 'Unknown'
                }
                result.append(result_item)

        # 按开始时间排序结果
        result.sort(key=lambda x: x['start'])
        return result

app = Flask(__name__)
CORS(app)

# 配置应用
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER

# 创建必要文件夹
Config.create_upload_folder()

# 初始化ASR系统
asr_system = WhisperASR()

@app.route('/v1/audio/transcriptions', methods=['POST'])
def transcribe_audio():
    """转录音频端点（兼容OpenAI API格式）"""
    try:
        # 检查文件上传
        if 'file' not in request.files:
            return jsonify({
                "error": {
                    "message": "没有上传文件",
                    "type": "invalid_request_error",
                    "code": "no_file"
                }
            }), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({
                "error": {
                    "message": "没有选择文件",
                    "type": "invalid_request_error",
                    "code": "empty_filename"
                }
            }), 400

        # 检查文件扩展名
        if not AudioProcessor.allowed_file(file.filename, Config.ALLOWED_EXTENSIONS):
            return jsonify({
                "error": {
                    "message": f"不支持的文件格式。允许的格式: {', '.join(Config.ALLOWED_EXTENSIONS)}",
                    "type": "invalid_request_error",
                    "code": "invalid_file_format"
                }
            }), 400

        # 读取文件数据
        file_data = file.read()

        # 获取参数
        model = request.form.get('model', 'whisper-1')
        language = request.form.get('language', None)
        task = request.form.get('task', 'transcribe')

        # 解析模型大小（支持OpenAI格式和直接指定）
        if model.startswith('whisper-'):
            model_map = {
                'whisper-1': 'base',  # OpenAI的默认模型
            }
            model_size = model_map.get(model, 'base')
        else:
            model_size = model.lower()

        # 保存上传的文件
        filename = f"upload_{int(time.time())}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        with open(filepath, 'wb') as f:
            f.write(file_data)

        try:
            # 执行转录
            result = asr_system.transcribe(
                audio_path=filepath,
                model_size=model_size,
                language=language,
                task=task
            )

            # 返回OpenAI兼容格式
            response = {
                "text": result["text"],
                "language": result["language"],
                "model": model,
                "speakers": result["speakers"]
            }

            return jsonify(response)

        finally:
            # 清理上传的文件
            try:
                os.remove(filepath)
            except:
                pass

    except Exception as e:
        logger.error(f"API错误: {e}")
        return jsonify({
            "error": {
                "message": str(e),
                "type": "internal_server_error",
                "code": "transcription_failed"
            }
        }), 500

@app.route('/v1/models', methods=['GET'])
def list_models():
    """列出可用模型（兼容OpenAI API格式）"""
    models = []
    for model_id, model_name in Config.WHISPER_MODELS.items():
        models.append({
            "id": f"whisper-{model_id}",
            "object": "model",
            "created": 1677610602,
            "owned_by": "openai",
            "permission": [],
            "root": f"whisper-{model_id}",
            "parent": None
        })

    return jsonify({
        "object": "list",
        "data": models
    })

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        "status": "healthy",
        "device": asr_system.current_device,
        "loaded_models": list(asr_system.models.keys())
    })

if __name__ == '__main__':
    # 预加载基础模型
    try:
        logger.info("预加载基础模型...")
        asr_system.load_model('base')
    except Exception as e:
        logger.warning(f"预加载模型失败: {e}")

    # 启动服务器
    logger.info(f"启动服务器: http://{Config.HOST}:{Config.PORT}")
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        threaded=True
    )
