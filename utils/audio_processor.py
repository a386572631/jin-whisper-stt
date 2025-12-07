import os
import io
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import tempfile
from typing import Optional, Tuple

class AudioProcessor:
    """音频处理器"""

    @staticmethod
    def allowed_file(filename: str, allowed_extensions: set) -> bool:
        """检查文件扩展名是否允许"""
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in allowed_extensions

    @staticmethod
    def convert_to_wav(audio_data: bytes, original_format: str) -> bytes:
        """将音频转换为WAV格式"""
        try:
            # 使用临时文件处理
            with tempfile.NamedTemporaryFile(suffix=f'.{original_format}', delete=False) as tmp_input:
                tmp_input.write(audio_data)
                tmp_input_path = tmp_input.name

            # 使用pydub加载音频
            audio = AudioSegment.from_file(tmp_input_path, format=original_format)

            # 转换为WAV
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_output:
                audio.export(tmp_output.name, format='wav')
                with open(tmp_output.name, 'rb') as f:
                    wav_data = f.read()

            # 清理临时文件
            os.unlink(tmp_input_path)
            os.unlink(tmp_output.name)

            return wav_data

        except Exception as e:
            print(f"音频转换失败: {e}")
            raise

    @staticmethod
    def load_audio(audio_bytes: bytes, sr: int = 16000) -> np.ndarray:
        """加载音频数据"""
        try:
            # 尝试直接读取
            with io.BytesIO(audio_bytes) as audio_file:
                data, sample_rate = sf.read(audio_file)

                # 转换为单声道
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)

                # 重采样
                if sample_rate != sr:
                    import librosa
                    data = librosa.resample(data, orig_sr=sample_rate, target_sr=sr)

                return data

        except Exception as e:
            print(f"音频加载失败: {e}")
            # 尝试使用pydub处理
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name

                audio = AudioSegment.from_file(tmp_path)
                audio = audio.set_frame_rate(sr).set_channels(1)

                # 转换为numpy数组
                samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0

                os.unlink(tmp_path)
                return samples

            except Exception as e2:
                print(f"备用音频加载也失败: {e2}")
                raise

    @staticmethod
    def get_audio_duration(audio_bytes: bytes) -> float:
        """获取音频时长（秒）"""
        try:
            with io.BytesIO(audio_bytes) as audio_file:
                data, sample_rate = sf.read(audio_file)
                return len(data) / sample_rate
        except:
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name

                audio = AudioSegment.from_file(tmp_path)
                duration = len(audio) / 1000.0  # 转换为秒

                os.unlink(tmp_path)
                return duration
            except:
                return 0.0