# -*- coding: utf-8 -*-
"""
vision.py
多模态视觉识别服务（独立 LLM API，OpenAI 兼容格式）

用途：
  将概念投放中的图片/视频上传到独立的多模态 LLM（默认智谱 GLM-4.6V），
  返回一段"只对主体进行细致外观描述"的客观文本。
  该文本随后与车身颜色、尺寸等一起作为外观评价因素，注入 stimulus。

环境变量（.env）：
  VISION_API_KEY    视觉模型 API Key
  VISION_BASE_URL   视觉模型 Base URL（OpenAI 兼容，如 https://open.bigmodel.cn/api/paas/v4/）
  VISION_MODEL      视觉模型名（如 glm-4.6v）
"""
import base64
import logging
import os
from typing import Optional

from openai import OpenAI

logger = logging.getLogger("vision")

# ----------------------------------------------------------------------
# 客观外观描述 Prompt（用户指定，禁止任何主观评价）
# ----------------------------------------------------------------------
APPEARANCE_PROMPT = (
    "请对图片/视频中的主体进行细致的外观描述，"
    "客观描述，不带夸张的修饰词，不作任何主观定义，不作任何主观评价。"
    "一句话总结特征"
)

# 兜底：当未配置 VISION_API_KEY 时，前端仍可继续流程，仅外观描述为空
_FALLBACK_EMPTY = ""

_client: Optional[OpenAI] = None


def _get_client() -> Optional[OpenAI]:
    """惰性构建 OpenAI 客户端；未配置则返回 None。"""
    global _client
    if _client is not None:
        return _client
    api_key = os.getenv("VISION_API_KEY", "").strip()
    base_url = os.getenv("VISION_BASE_URL", "").strip()
    if not api_key:
        logger.warning("VISION_API_KEY 未配置，视觉识别将返回空描述")
        return None
    _client = OpenAI(
        api_key=api_key,
        base_url=base_url or None,
    )
    return _client


def _model() -> str:
    return os.getenv("VISION_MODEL", "glm-4.6v").strip() or "glm-4.6v"


def _b64_data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _clean_thinking(text: str) -> str:
    """清理 thinking 模型返回的 <think>...</think> 和 <answer>...</answer> 标签，
    只保留实际描述内容。"""
    import re
    if not text:
        return ""
    # 优先提取 <answer>...</answer> 中的内容
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # 否则移除所有 <think>...</think> 块
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


# ----------------------------------------------------------------------
# 公开接口
# ----------------------------------------------------------------------
def describe_image(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    """
    输入图片二进制 + mime，返回客观外观描述文本。
    失败时返回空串（不抛异常，避免阻断主流程）。
    """
    client = _get_client()
    if client is None:
        return _FALLBACK_EMPTY
    try:
        data_url = _b64_data_url(image_bytes, mime)
        resp = client.chat.completions.create(
            model=_model(),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": APPEARANCE_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=2000,
        )
        text = _clean_thinking((resp.choices[0].message.content or "").strip())
        logger.info("视觉识别(图片)成功，描述长度=%s", len(text))
        return text
    except Exception as e:
        logger.exception("视觉识别(图片)失败: %s", e)
        return _FALLBACK_EMPTY


def describe_video(video_bytes: bytes, mime: str = "video/mp4") -> str:
    """
    输入视频二进制 + mime，返回客观外观描述文本。
    采用 GLM-4V 系列的 video_url 扩展字段（base64 data URL）。
    失败时返回空串。
    """
    client = _get_client()
    if client is None:
        return _FALLBACK_EMPTY
    try:
        data_url = _b64_data_url(video_bytes, mime)
        resp = client.chat.completions.create(
            model=_model(),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": APPEARANCE_PROMPT},
                        {"type": "video_url", "video_url": {"url": data_url}},
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=2000,
        )
        text = _clean_thinking((resp.choices[0].message.content or "").strip())
        logger.info("视觉识别(视频)成功，描述长度=%s", len(text))
        return text
    except Exception as e:
        logger.exception("视觉识别(视频)失败: %s", e)
        return _FALLBACK_EMPTY


def is_configured() -> bool:
    """供 /api/health 暴露视觉服务可用性。"""
    return bool(os.getenv("VISION_API_KEY", "").strip())
