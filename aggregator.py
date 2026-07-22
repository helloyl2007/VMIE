# -*- coding: utf-8 -*-
"""
aggregator.py
把 simulation.evaluate_car() 返回的逐个虚拟用户结构化结果，
聚合为 ui.html 前端 genBaseline() 期望的统一数据结构。

输出字段与 ui.html 中的 baseline 完全一致：
  params, nps, pos, neu, neg,
  cognition, interest, intent, trial, prebuy, reach,
  praise[{text,n}], complain[{text,n}],
  wow24[], wow7[], wow30[], radar[], overall, agents[]
"""
import math
import random
from collections import Counter
from typing import List, Dict, Any


RADAR_DIMS = ["续航", "智能", "空间", "豪华", "性价比", "外观"]


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def _gen_wow_series(n: int, overall: float, jitter: float = 0.06) -> List[float]:
    """
    生成口碑传播时间序列（与 ui.html.genWowSeries 行为一致）。
    用 overall 作为基准，叠加小幅噪声并做平滑。
    """
    arr = []
    v = overall * 0.4
    for i in range(n):
        tgt = overall * random.uniform(0.5, 1.1)
        v += (tgt - v) * 0.3 + random.uniform(-jitter, jitter)
        arr.append(round(_clamp(v), 3))
    return arr


def aggregate(
    agent_results: List[Dict[str, Any]],
    total_users: int,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    聚合 N 个虚拟用户的评价结果。

    Args:
        agent_results: evaluate_car() 的输出，每项含
            name, rating(1-5), sentiment, praise[], complain[],
            funnel_stage, scores{6dims}, summary
        total_users: 前端显示的虚拟用户总数（视觉放大用，与实际 agent 数解耦）
        params: 当前生效的参数快照，原样回传前端
    """
    n = len(agent_results) or 1

    ratings = [int(r.get("rating", 3)) for r in agent_results]

    # ---------- NPS ----------
    # promoters: 4-5, detractors: 1-2
    promoters = sum(1 for r in ratings if r >= 4)
    detractors = sum(1 for r in ratings if r <= 2)
    nps_raw = round((promoters - detractors) / n * 100)  # -100..100
    # 映射到 ui.html 习惯的 -10..60 范围，避免 5 人样本时跳变过大
    nps = round(-10 + (nps_raw + 100) / 200 * 70)
    nps = max(-10, min(60, nps))

    # ---------- 情绪 ----------
    pos_cnt = sum(1 for r in agent_results if r.get("sentiment") == "positive")
    neg_cnt = sum(1 for r in agent_results if r.get("sentiment") == "negative")
    neu_cnt = n - pos_cnt - neg_cnt
    pos = round(pos_cnt / n * 100)
    neg = round(neg_cnt / n * 100)
    neu = 100 - pos - neg
    pos = max(5, min(95, pos))
    neg = max(5, min(95, neg))
    neu = max(0, 100 - pos - neg)

    # ---------- 漏斗（累积百分比，单调递减）----------
    cognition = 100
    interest = round(sum(1 for r in ratings if r >= 2) / n * 100)
    intent = round(sum(1 for r in ratings if r >= 3) / n * 100)
    trial = round(sum(1 for r in ratings if r >= 4) / n * 100)
    prebuy = round(sum(1 for r in ratings if r >= 5) / n * 100)
    # 保证单调
    interest = min(cognition, interest)
    intent = min(interest, intent)
    trial = min(intent, trial)
    prebuy = min(trial, prebuy)

    # ---------- 赞赏 / 吐槽 Top5 ----------
    praise_counter: Counter = Counter()
    complain_counter: Counter = Counter()
    for r in agent_results:
        for p in r.get("praise", []):
            p = str(p).strip()
            if p:
                praise_counter[p] += 1
        for c in r.get("complain", []):
            c = str(c).strip()
            if c:
                complain_counter[c] += 1

    # 把 agent 内部计数放大到与 ui.html 视觉量级匹配
    scale = max(1, total_users // max(1, n))
    praise = [
        {"text": t, "n": int(c * scale * random.uniform(0.8, 1.4))}
        for t, c in praise_counter.most_common(5)
    ]
    complain = [
        {"text": t, "n": int(c * scale * random.uniform(0.8, 1.4))}
        for t, c in complain_counter.most_common(5)
    ]
    # 兜底：若 LLM 没产出赞赏/吐槽，用默认占位
    if not praise:
        praise = [{"text": "整体观感尚可", "n": int(scale * 0.6)}]
    if not complain:
        complain = [{"text": "暂无明显槽点", "n": int(scale * 0.3)}]

    # ---------- 雷达 ----------
    radar = []
    for dim in RADAR_DIMS:
        vals = []
        for r in agent_results:
            sc = r.get("scores") or {}
            try:
                vals.append(float(sc.get(dim, 0.5)))
            except Exception:
                vals.append(0.5)
        avg = sum(vals) / len(vals) if vals else 0.5
        radar.append(round(_clamp(avg), 3))

    # ---------- overall ----------
    overall = sum(ratings) / (n * 5)  # 0..1
    overall = _clamp(overall, 0.05, 0.98)

    # ---------- 口碑触达（按实际样本数据反映）----------
    reach = round(total_users * overall * random.uniform(2.2, 3.5))

    # ---------- 口碑曲线 ----------
    wow24 = _gen_wow_series(24, overall)
    wow7 = _gen_wow_series(28, overall)
    wow30 = _gen_wow_series(30, overall)

    # ---------- agents 明细（前端日志/调试用，不强制渲染）----------
    agents = [
        {
            "name": r.get("name", f"用户{i+1}"),
            "rating": int(r.get("rating", 3)),
            "sentiment": r.get("sentiment", "neutral"),
            "funnel_stage": r.get("funnel_stage", "认知"),
            "summary": r.get("summary", ""),
        }
        for i, r in enumerate(agent_results)
    ]

    return {
        "params": params,
        "nps": nps,
        "pos": pos, "neu": neu, "neg": neg,
        "cognition": cognition, "interest": interest,
        "intent": intent, "trial": trial, "prebuy": prebuy,
        "reach": reach,
        "praise": praise, "complain": complain,
        "wow24": wow24, "wow7": wow7, "wow30": wow30,
        "radar": radar,
        "overall": round(overall, 3),
        "agents": agents,
    }
