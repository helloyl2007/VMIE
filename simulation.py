# -*- coding: utf-8 -*-
"""
simulation.py
基于 TinyTroupe 框架的虚拟用户仿真核心逻辑：
  1. 创建关注汽车的虚拟用户（TinyPersonFactory）
  2. 让虚拟用户对新车概念进行评价（listen_and_act）
  3. 提取结构化结果（ResultsExtractor）

本模块只关注基本功能：
  - 不包含"虚拟用户自我进化"
  - 不包含"蒸馏用户"
"""
import os
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

# 把本地 TinyTroupe 源码目录加入 sys.path，无需 pip install tinytroupe
_TINYTROUPE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TinyTroupe")
if _TINYTROUPE_PATH not in sys.path:
    sys.path.insert(0, _TINYTROUPE_PATH)

# 加载 .env（若存在）
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# TinyTroupe 的 config 读取会读 os.getenv("OPENAI_API_KEY") / OPENAI_BASE_URL
# 所以 .env 中只需配置这两个变量即可走 OpenAI 兼容网关

logger = logging.getLogger("simulation")


def _import_tinytroupe():
    """惰性导入 TinyTroupe，避免 Flask 启动时即触发 LLM 客户端初始化报错。"""
    import tinytroupe  # noqa: F401  触发 config_manager 初始化
    from tinytroupe.agent import TinyPerson
    from tinytroupe.factory import TinyPersonFactory
    from tinytroupe.extraction import ResultsExtractor
    return TinyPerson, TinyPersonFactory, ResultsExtractor


# =====================================================================
# 1. 上下文 / 人格池
# =====================================================================

# 与 ui.html 中 INDUSTRIES.auto.personaPool 对应的关注汽车的用户原型
PERSONA_POOL = [
    "都市白领，28-35岁，工作日通勤为主，关注颜值与智能配置，预算25-35万",
    "奶爸车主，32-40岁，已婚有两个孩子，看重空间、安全与续航，预算25-40万",
    "科技极客，25-35岁，对智能驾驶、车机系统、OTA 升级非常敏感，预算30-50万",
    "自驾达人，30-45岁，喜欢周末远行与露营，关注续航、快充与越野能力，预算30-50万",
    "理性工薪，30-45岁，家庭年收入20-30万，极度看重性价比、保值率与可靠性，预算15-25万",
]

FACTORY_CONTEXT = """
中国汽车市场的潜在消费者群体。
覆盖不同年龄段（25-55岁）、不同收入水平（年收入10万-80万）、不同家庭状况（单身、已婚有娃、多代同堂）。
他们都对汽车有一定关注，从日常代步需求到深度车迷，关注维度包括：续航、价格、智能化、空间、豪华感、品牌、安全性、保值率等。
请生成符合中国当下消费特征的真实人格，包含姓名、年龄、职业、收入、家庭、用车需求、价值观等。
"""

INDUSTRY_CONTEXTS = {
    "auto": FACTORY_CONTEXT,
    "elec": """
中国服装市场的潜在消费者群体。
覆盖不同年龄段、消费层级和生活方式的人群，关注款式、面料、版型、品牌调性、价格与穿着场景。
请生成符合中国当下服装消费特征的真实人格，包含姓名、年龄、职业、收入、穿衣偏好、购买场景、价值观等。
""",
    "fmcg": """
中国手机市场的潜在消费者群体。
覆盖学生、职场人士、游戏玩家、内容创作者和商务用户，关注性能、影像、续航、系统体验、价格与品牌。
请生成符合中国当下手机消费特征的真实人格，包含姓名、年龄、职业、收入、使用习惯、换机动机、价值观等。
""",
}


# =====================================================================
# 2. 创建虚拟用户
# =====================================================================

def create_virtual_users(
    num_users: int = 5,
    industry: str = "auto",
    persona_specs: Optional[List[Dict[str, str]]] = None,
) -> List[Any]:
    """
    使用 TinyPersonFactory 生成指定数量的虚拟用户。
    会按照 PERSONA_POOL 给每个用户注入差异化人格。
    """
    TinyPerson, TinyPersonFactory, _ = _import_tinytroupe()

    # 清理之前的 agent（重复创建时）
    TinyPerson.clear_agents()

    context = INDUSTRY_CONTEXTS.get(industry, FACTORY_CONTEXT)
    factory = TinyPersonFactory(context=context)

    prepared_specs = [spec for spec in (persona_specs or []) if spec.get("description")]
    persona_pool = prepared_specs or [
        {"name": f"原型{i+1}", "description": text}
        for i, text in enumerate(PERSONA_POOL)
    ]

    # 轮转使用 persona 池作为差异化指令
    users: List[TinyPerson] = []
    for i in range(num_users):
        spec = persona_pool[i % len(persona_pool)]
        particularities = spec["description"]
        person = factory.generate_person(agent_particularities=particularities)
        if person is not None:
            users.append(person)
            logger.info(
                f"[{i+1}/{num_users}] 生成虚拟用户({industry}/{spec['name']}): {person.minibio()[:80]}..."
            )
        else:
            logger.warning(f"[{i+1}/{num_users}] 生成失败，跳过")

    if not users:
        raise RuntimeError("未能生成任何虚拟用户，请检查 LLM API 配置")

    return users


def serialize_virtual_users(users: List[Any]) -> List[Dict[str, Any]]:
    """把已生成的 TinyPerson 持久化为可重载的规格。"""
    serialized: List[Dict[str, Any]] = []
    profiles = get_user_profiles(users, extended=True)
    for user, profile in zip(users, profiles):
        serialized.append({
            "name": profile["name"],
            "description": profile["minibio"],
            "spec": user.to_json(
                suppress=["episodic_memory", "semantic_memory", "_mental_state"],
                serialization_type_field_name="type",
            ),
        })
    return serialized


def generate_virtual_user_specs(
    num_users: int = 5,
    industry: str = "auto",
    persona_specs: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """调用 LLM 生成虚拟用户，并返回可保存的规格。"""
    users = create_virtual_users(
        num_users=num_users,
        industry=industry,
        persona_specs=persona_specs,
    )
    return serialize_virtual_users(users)


def load_virtual_users_from_specs(saved_specs: List[Dict[str, Any]]) -> List[Any]:
    """从已保存的规格恢复虚拟用户，避免重复调用 LLM。"""
    TinyPerson, _, _ = _import_tinytroupe()
    TinyPerson.clear_agents()

    users: List[Any] = []
    for idx, item in enumerate(saved_specs, start=1):
        spec = item.get("spec") if isinstance(item, dict) else None
        if not isinstance(spec, dict):
            continue
        try:
            user = TinyPerson.load_specification(
                spec,
                suppress_memory=True,
                suppress_mental_state=True,
            )
            users.append(user)
        except Exception as exc:
            logger.warning("加载已保存虚拟用户失败 [%s]: %s", idx, exc)

    if not users:
        raise RuntimeError("未能加载任何已保存的虚拟用户，请先在设置页重新生成")

    return users


# =====================================================================
# 3. 构建新车概念刺激文本
# =====================================================================

def build_car_stimulus(concept: Dict[str, Any]) -> str:
    """
    把前端传来的车型概念参数，拼成一段给虚拟用户"听"的产品介绍文本。
    concept 字段：
      brand, product_plan, color, size, power, range, price, audience, tags,
      charge, smart, space, lux, vision_description(多模态识别的客观外观描述)
    """
    tags = concept.get("tags") or []
    tags_str = "、".join(tags) if tags else "（未指定）"
    colors = concept.get("colors") or []
    color_str = "、".join(colors) if colors else concept.get("color", "未指定")
    video_line = concept.get("video_url") or "未提供"
    video_name = concept.get("video_name") or ""
    image_note = concept.get("image_note") or "未提供"
    brand = concept.get("brand", "") or ""
    vision_desc = (concept.get("vision_description") or "").strip()

    # 视频显示：优先文件名，否则用链接
    video_display = video_name or video_line or "未提供"

    # 视觉识别的客观外观描述（作为外观评价因素之一）
    vision_block = (
        f"【多模态识别 · 客观外观描述】\n{vision_desc}\n（以上为对图片/视频主体的客观外观描述，"
        "请将其与车身颜色、尺寸等一起作为你对外观维度评价的依据。）"
        if vision_desc else
        "【多模态识别 · 客观外观描述】未提供"
    )

    return f"""【新车概念发布】

我们正在研发一款面向「{concept.get('audience', '泛大众')}」的{concept.get('car_type', '')}（{concept.get('size', '')}·{concept.get('power', '')}）。
品牌：{brand or '（未指定）'}

【产品方案】
{concept.get('product_plan', '（暂无描述）')}

【核心配置】
- 品牌：{brand or '未指定'}
- 外观图来源：{image_note}
- 演示视频：{video_display}
- 车身颜色：{color_str}
- 车型：{concept.get('car_type', '未指定')}
- 尺寸级别：{concept.get('size', '未指定')}
- 动力类型：{concept.get('power', '未指定')}
- 续航里程：{concept.get('range', '未指定')}km
- 价格区间：{concept.get('price', '未指定')}万元
- 快充时间：{concept.get('charge', 15)}分钟
- 智能化水平：{concept.get('smart', 7)}/10
- 空间水平：{concept.get('space', 6)}/10
- 豪华感：{concept.get('lux', 5)}/10
- 核心卖点：{tags_str}

（注意：如果以上配置选项与【产品方案】描述存在冲突，以【产品方案】描述为准。）

{vision_block}

【你的任务】
请基于你真实的身份、用车场景、预算与偏好，仔细评估这款车型。
你需要像真实用户一样完成一轮自然反应：可以先思考，再表达观点；如果这符合你的人设，也可以进一步做出行动，例如去对比竞品、问朋友、收藏、转发、写短评、发朋友圈/微博、咨询销售、预约试驾等。
请不要为了完成任务而机械地凑动作；只有当这些行为符合你的真实偏好和当下情境时才去做。

在这一轮反应里，尽量自然地产生一小段动作序列，动作可包含 THINK / TALK / DO / WRITE；其中：
- THINK：你的内心判断与权衡
- TALK：你会对谁说什么
- DO：你会去做什么实际动作
- WRITE：你会公开或私下写下什么内容，例如朋友圈、微博、群消息、短评、收藏备注

完成这轮自然反应后，再给出结构化判断：
1) 购买意愿评分（1-5）：1=绝不会买，2=不太可能，3=观望，4=很可能，5=一定会买
2) 情感倾向：positive / neutral / negative
3) 你最赞赏的 1-3 个点
4) 你最不满意的 1-3 个点
5) 你当前所处的购买漏斗阶段：认知 / 兴趣 / 意向 / 试驾 / 预购
6) 对六个维度的评分（0.0-1.0）：续航 / 智能 / 空间 / 豪华 / 性价比 / 外观
（注意：外观维度的评分请综合参考"车身颜色、尺寸级别"以及上方【多模态识别 · 客观外观描述】中的细节。）
"""


# =====================================================================
# 4. 评价 & 提取
# =====================================================================

EXTRACT_FIELDS = [
    "name", "rating", "sentiment", "praise", "complain",
    "funnel_stage", "scores", "summary",
]

EXTRACT_FIELDS_HINTS = {
    "name": "用户姓名（字符串）",
    "rating": "整数 1-5。1=绝不会买，5=一定会买。必须是字符串形式的数字，如 '4'",
    "sentiment": "必须是 positive / neutral / negative 之一",
    "praise": "用户赞赏的具体点列表（list[str]），每项一个短语，最多3条",
    "complain": "用户吐槽的具体点列表（list[str]），每项一个短语，最多3条",
    "funnel_stage": "必须是 认知 / 兴趣 / 意向 / 试驾 / 预购 之一",
    "scores": "dict，包含键 续航/智能/空间/豪华/性价比/外观，每个值为 0.0-1.0 的浮点数",
    "summary": "用户整体评价的一句话总结",
}


def _make_extractor():
    _, _, ResultsExtractor = _import_tinytroupe()
    return ResultsExtractor(
        extraction_objective=(
            "提取该虚拟用户对新车概念的结构化评价：购买意愿评分、情感倾向、赞赏点、"
            "吐槽点、购买漏斗阶段、六维度评分、一句话总结。"
        ),
        situation=(
            "该虚拟用户刚刚了解了一款新车的概念信息（产品方案、配置、价格等），"
            "并已给出自己的反应与评价。请从其交互历史中提取结构化结论。"
        ),
        fields=EXTRACT_FIELDS,
        fields_hints=EXTRACT_FIELDS_HINTS,
        verbose=False,
    )


def _normalize_action_for_ui(action: Dict[str, Any]) -> Dict[str, str]:
    raw_type = str(action.get("type", "")).upper().strip()
    target = str(action.get("target", "")).strip()
    content = str(action.get("content", "")).strip()[:200]

    if raw_type == "TALK":
        ui_type = "SAY"
    elif raw_type == "THINK":
        ui_type = "THINK"
    elif raw_type.startswith("WRITE"):
        ui_type = "WRITE"
    else:
        ui_type = "DO"

    return {
        "type": ui_type,
        "raw_type": raw_type,
        "target": target,
        "content": content,
    }


def _evaluate_one(agent, stimulus, extractor):
    """单个虚拟用户：听完车型介绍 → 作出反应 → 提取结构化结果。"""
    actions = None
    try:
        actions = agent.listen_and_act(stimulus, return_actions=True)
    except Exception as e:
        logger.warning(f"agent {getattr(agent, 'name', '?')} listen_and_act 失败: {e}")

    # 提取有效动作（供前端气泡展示）
    user_actions = []
    for item in (actions or []):
        if isinstance(item, dict) and isinstance(item.get("action"), dict):
            act = item["action"]
            act_type = act.get("type")
            if act_type in (None, "DONE"):
                continue
            user_actions.append(_normalize_action_for_ui(act))

    has_substantive_action = len(user_actions) > 0

    if not has_substantive_action:
        logger.warning(
            "agent %s 未产生有效动作，跳过结构化提取",
            getattr(agent, "name", "?"),
        )
        result = None
    else:
        try:
            result = extractor.extract_results_from_agent(agent)
        except Exception as e:
            logger.warning(f"agent {getattr(agent, 'name', '?')} 提取结果失败: {e}")
            result = None

    if not isinstance(result, dict):
        result = {}

    # 兜底默认值，保证后续聚合不崩
    result.setdefault("name", getattr(agent, "name", "unknown"))
    result.setdefault("rating", 3)
    try:
        result["rating"] = int(result.get("rating", 3))
    except Exception:
        result["rating"] = 3
    if result["rating"] < 1:
        result["rating"] = 1
    if result["rating"] > 5:
        result["rating"] = 5

    sent = str(result.get("sentiment", "neutral")).lower().strip()
    if sent not in ("positive", "neutral", "negative"):
        sent = "neutral"
    result["sentiment"] = sent

    if not isinstance(result.get("praise"), list):
        result["praise"] = []
    if not isinstance(result.get("complain"), list):
        result["complain"] = []

    stage = result.get("funnel_stage", "")
    if stage not in ("认知", "兴趣", "意向", "试驾", "预购"):
        # 根据 rating 推断漏斗阶段
        r = result["rating"]
        stage = {1: "认知", 2: "兴趣", 3: "意向", 4: "试驾", 5: "预购"}.get(r, "认知")
    result["funnel_stage"] = stage

    if not isinstance(result.get("scores"), dict):
        result["scores"] = {}
    for dim in ("续航", "智能", "空间", "豪华", "性价比", "外观"):
        try:
            v = float(result["scores"].get(dim, 0.5))
        except Exception:
            v = 0.5
        result["scores"][dim] = max(0.0, min(1.0, v))

    result.setdefault("summary", "")
    result["_actions"] = user_actions
    return result


def evaluate_car(concept: Dict[str, Any], users: List[Any]) -> List[Dict[str, Any]]:
    """
    让所有虚拟用户对新车概念进行评价，并提取结构化结果。
    使用线程池并发以加速（受 MAX_CONCURRENT_MODEL_CALLS 限制）。
    """
    stimulus = build_car_stimulus(concept)
    extractor = _make_extractor()

    results: List[Dict[str, Any]] = []
    # 并发执行；TinyTroupe 内部有信号量限流
    with ThreadPoolExecutor(max_workers=min(5, len(users) or 1)) as pool:
        futures = [pool.submit(_evaluate_one, u, stimulus, extractor) for u in users]
        for idx, fut in enumerate(futures):
            try:
                results.append(fut.result())
                logger.info(f"完成评价 [{idx+1}/{len(users)}]")
            except Exception as e:
                logger.error(f"虚拟用户 {idx+1} 评价失败: {e}")
                # 兜底，保证数量一致
                results.append({
                    "name": f"虚拟用户{idx+1}",
                    "rating": 3, "sentiment": "neutral",
                    "praise": [], "complain": [],
                    "funnel_stage": "认知", "scores": {},
                    "summary": "（评价失败）",
                })
    return results


def evaluate_car_stream(concept: Dict[str, Any], users: List[Any]):
    """
    流式版本：每完成一个用户即 yield 该用户的动作与结果。
    yield 格式: {"type":"user_done","index":int,"name":str,"actions":[...],"result":dict}
    最后 yield: {"type":"all_done","results":[...]}
    """
    stimulus = build_car_stimulus(concept)
    extractor = _make_extractor()
    results: List[Dict[str, Any]] = [None] * len(users)

    with ThreadPoolExecutor(max_workers=min(5, len(users) or 1)) as pool:
        future_to_idx = {
            pool.submit(_evaluate_one, u, stimulus, extractor): idx
            for idx, u in enumerate(users)
        }
        done_count = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
            except Exception as e:
                logger.error(f"虚拟用户 {idx+1} 评价失败: {e}")
                result = {
                    "name": f"虚拟用户{idx+1}",
                    "rating": 3, "sentiment": "neutral",
                    "praise": [], "complain": [],
                    "funnel_stage": "认知", "scores": {},
                    "summary": "（评价失败）",
                    "_actions": [],
                }
            results[idx] = result
            done_count += 1
            logger.info(f"完成评价 [{done_count}/{len(users)}]")

            yield {
                "type": "user_done",
                "index": idx,
                "name": result.get("name", f"用户{idx+1}"),
                "actions": result.get("_actions", []),
                "result": {k: v for k, v in result.items() if k != "_actions"},
                "done_count": done_count,
                "total_count": len(users),
            }

    yield {"type": "all_done", "results": results}


# =====================================================================
# 5. 工具：获取用户画像摘要（供前端展示用）
# =====================================================================

def get_user_profiles(users: List[Any], extended: bool = False) -> List[Dict[str, str]]:
    """返回每个虚拟用户的简要画像信息；默认不触发扩展摘要生成。"""
    profiles = []
    for u in users:
        try:
            profiles.append({
                "name": u.name,
                "minibio": u.minibio(extended=extended),
            })
        except Exception:
            profiles.append({"name": getattr(u, "name", "?"), "minibio": ""})
    return profiles
