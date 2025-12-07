import os
import re
import json
from typing import Dict, List, Optional
import requests

class TextCorrector:
    """文本修正器"""

    def __init__(self, enable_correction: bool = True, rules: Optional[Dict] = None):
        self.enable_correction = enable_correction
        self.rules = rules or {}

        # 常见错误模式
        self.common_errors = {
            r'(\w)\s+([,.!?;:])': r'\1\2',  # 移除标点前的空格
            r'([,.!?;:])\s*,\s*': r'\1 ',   # 修正重复标点
            r'\s+': ' ',                     # 多个空格合并为一个
            r'^\\s+': '',                    # 移除开头的空格
            r'\\s+$': '',                    # 移除结尾的空格
        }

        # 中文常见错误修正
        self.chinese_errors = {
            '嗯啊': '嗯，',
            '啊哈': '啊，',
            '那个': '',  # 移除填充词
            '这个': '',
            '就是': '',
        }

    def correct_text(self, text: str, language: str = 'zh') -> str:
        """修正文本"""
        if not self.enable_correction:
            return text

        # 应用自定义规则
        for old, new in self.rules.items():
            text = text.replace(old, new)

        # 应用常见错误修正
        for pattern, replacement in self.common_errors.items():
            text = re.sub(pattern, replacement, text)

        # 语言特定的修正
        if 'zh' in language.lower():
            text = self._correct_chinese(text)
        elif 'en' in language.lower():
            text = self._correct_english(text)

        # 句子首字母大写（英文）
        if 'en' in language.lower():
            sentences = re.split(r'([.!?]\\s+)', text)
            text = ''.join([
                s.capitalize() if i % 2 == 0 else s
                for i, s in enumerate(sentences)
            ])

        return text.strip()

    def _correct_chinese(self, text: str) -> str:
        """修正中文文本"""
        # 移除常见填充词
        for error, correction in self.chinese_errors.items():
            text = text.replace(error, correction)

        # 修正标点符号
        text = re.sub(r'([，。！？；：])\1+', r'\1', text)  # 重复标点

        return text

    def _correct_english(self, text: str) -> str:
        """修正英文文本"""
        # 修正常见拼写错误
        common_mistakes = {
            'teh': 'the',
            'adn': 'and',
            'thier': 'their',
            'recieve': 'receive',
            'seperate': 'separate',
        }

        words = text.split()
        corrected_words = []
        for word in words:
            lower_word = word.lower()
            if lower_word in common_mistakes:
                # 保持原单词的大小写格式
                if word.isupper():
                    corrected_words.append(common_mistakes[lower_word].upper())
                elif word[0].isupper():
                    corrected_words.append(common_mistakes[lower_word].capitalize())
                else:
                    corrected_words.append(common_mistakes[lower_word])
            else:
                corrected_words.append(word)

        return ' '.join(corrected_words)

    def correct_with_ai(self, text: str) -> str:
        """使用大模型进行高级修正（可选）"""
        from dotenv import load_dotenv
        load_dotenv(override=True)

        api_key=os.getenv("DEEPSEEK_API_KEY")
        url=os.getenv("DEEPSEEK_URL")

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            data = {
                    "inputs": {
                        "content": text
                    },
                    "response_mode": "blocking",
                    "user": "0"
            }

            response = requests.post(url, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                print(f"Deepseek修正：" + str(result))
                return result['data']['outputs']['text'].strip()
            else:
                print(f"大模型纠正错误: {response.status_code}")
                return text

        except Exception as e:
            print(f"大模型纠正错误: {e}")
            return text
