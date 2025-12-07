# Whisper语音识别API服务

基于OpenAI Whisper和Flask构建的语音识别API服务，提供兼容OpenAI API的接口。

## 功能特性

- ✅ 支持Whisper所有模型级别：tiny/base/small/medium/large/large-v2/large-v3
- ✅ 兼容OpenAI API接口格式
- ✅ 支持CUDA 13.0驱动
- ✅ 智能文本后处理修正，调用大模型修正
- ✅ 支持多种音频格式
- ✅ 提供健康检查端点

## 安装要求

### Python版本
- Python 3.10

### 系统依赖
- 对于CUDA支持：NVIDIA GPU + CUDA 13.0 + cuDNN（有待验证中...）

## 安装步骤

1. 克隆项目并进入目录：
```bash
git clone <repository-url>

cd whisper-tts
```

2. 安装虚拟环境，并激活
```bash
conda create -n whisper-stt python=3.10 -y

conda activate whisper-stt
```
3. 安装项目所需依赖
```bash
pip install -r requirements.txt

```
4. 运行
```bash
python app.py
```