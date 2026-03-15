"""文本清洗工具"""
import re

def clean_legal_text(text: str) -> str:
    """
    清洗法律文档文本中的噪音数据
    
    Args:
        text: 原始文本
    
    Returns:
        清洗后的文本
    """
    # 1. 统一处理特殊空格（如 \xa0 全角空格）
    text = text.replace('\xa0', ' ')
    
    # 2. 匹配并去除各类页码标识：如 "- 1 -", "— 1 —", "-1-", "—1—"
    text = re.sub(r'[-—]\s*\d+\s*[-—]', '', text)
    
    # 3. 压缩连续的空行（包括带有空格的空行）
    text = re.sub(r'\n\s*\n+', '\n', text)
    
    return text.strip()
