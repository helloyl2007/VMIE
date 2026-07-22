# -*- coding: utf-8 -*-
"""
app.py
AI 虚拟市场洞察平台 · Flask 后端

对外接口：
  GET  /api/health            健康检查
  POST /api/create_market     创建虚拟用户（TinyTroupe TinyPersonFactory）
  POST /api/analyze            让虚拟用户评价新车概念，返回聚合数据
  POST /api/tune               调参后重算，返回新聚合数据
  GET  /api/users              查看已创建的虚拟用户画像

启动：
  python app.py
默认监听 http://127.0.0.1:5000
前端 ui.html 通过 fetch 调用上述接口。
"""
import os
import json
import logging
from typing import Dict, Any, List

from flask import Flask, request, jsonify, send_file, send_from_directory, Response, stream_with_context
from flask_cors import CORS

import simulation
import aggregator
import vision

# ----------------------------------------------------------------------
# 日志
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app")

# ----------------------------------------------------------------------
# Flask 应用
# ----------------------------------------------------------------------
app = Flask(__name__)
CORS(app)  # 允许 ui.html 以 file:// 或其它端口访问

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSONAS_DIR = os.path.join(BASE_DIR, "personas")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
INDUSTRY_LABELS = {
    "auto": "汽车",
    "elec": "服装",
    "fmcg": "手机",
}


def _ensure_personas_dir() -> None:
    os.makedirs(PERSONAS_DIR, exist_ok=True)


def _industry_file(industry: str) -> str:
    return os.path.join(PERSONAS_DIR, f"{industry}.json")


def _normalize_personas(personas: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, str]] = []
    if not isinstance(personas, list):
        return normalized
    for idx, item in enumerate(personas, start=1):
        if isinstance(item, str):
            name = f"虚拟用户{idx}"
            description = item.strip()
            spec = None
        elif isinstance(item, dict):
            name = str(item.get("name", "")).strip() or f"虚拟用户{idx}"
            description = str(item.get("description", "")).strip()
            spec = item.get("spec") if isinstance(item.get("spec"), dict) else None
        else:
            continue
        if description:
            persona = {"name": name, "description": description}
            if spec:
                persona["spec"] = spec
            normalized.append(persona)
    return normalized


def _load_personas(industry: str) -> List[Dict[str, str]]:
    _ensure_personas_dir()
    path = _industry_file(industry)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return _normalize_personas(payload.get("personas", []))


def _save_personas(industry: str, personas: List[Dict[str, str]]) -> None:
    _ensure_personas_dir()
    payload = {
        "industry": industry,
        "industry_name": INDUSTRY_LABELS.get(industry, industry),
        "personas": _normalize_personas(personas),
    }
    with open(_industry_file(industry), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _clear_personas(industry: str) -> None:
    _ensure_personas_dir()
    path = _industry_file(industry)
    payload = {
        "industry": industry,
        "industry_name": INDUSTRY_LABELS.get(industry, industry),
        "personas": [],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _industry_meta(industry: str) -> Dict[str, Any]:
    personas = _load_personas(industry)
    return {
        "key": industry,
        "name": INDUSTRY_LABELS.get(industry, industry),
        "count": len(personas),
        "has_personas": bool(personas),
        "file": os.path.relpath(_industry_file(industry), BASE_DIR).replace("\\", "/"),
    }

# 全局状态：虚拟用户列表 + 最近一次概念 + 显示用用户总数
_state: Dict[str, Any] = {
    "users": [],              # List[TinyPerson]
    "last_concept": None,     # 最近一次 analyze 的完整 concept
    "total_users": 0,         # 前端显示的虚拟用户总数
    "real_user_count": 0,     # 实际生成的虚拟用户数
    "status": "idle",         # idle / creating / ready / analyzing
    "industry": None,
    "report": None,           # 最近一次分析报告文本
}


# ======================================================================
# 健康检查
# ======================================================================
@app.get("/api/health")
def health():
    api_key_set = bool(os.getenv("OPENAI_API_KEY"))
    base_url = os.getenv("OPENAI_BASE_URL", "")
    return jsonify({
        "status": "ok",
        "openai_api_key_set": api_key_set,
        "openai_base_url": base_url,
        "tinytroupe_loaded": True,  # 到这里说明 simulation 模块导入成功
        "vision_api_key_set": vision.is_configured(),
        "vision_model": os.getenv("VISION_MODEL", "glm-4.6v"),
        "state": _state["status"],
        "industry": _state["industry"],
        "real_user_count": _state["real_user_count"],
        "total_users": _state["total_users"],
    })


# ======================================================================
# 多模态视觉识别：上传图片/视频，返回客观外观描述
# ======================================================================
@app.post("/api/vision")
def vision_describe():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "缺少上传文件 (file)"}), 400
    f = request.files["file"]
    if not f or not f.filename:
        return jsonify({"status": "error", "message": "文件为空"}), 400

    raw = f.read()
    if not raw:
        return jsonify({"status": "error", "message": "文件内容为空"}), 400

    mime = (f.mimetype or "").lower()
    fname = (f.filename or "").lower()
    try:
        if mime.startswith("image/") or any(fname.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")):
            if not mime.startswith("image/"):
                mime = "image/jpeg"
            desc = vision.describe_image(raw, mime)
            kind = "image"
        elif mime.startswith("video/") or any(fname.endswith(ext) for ext in (".mp4", ".mov", ".avi", ".webm", ".mkv")):
            if not mime.startswith("video/"):
                mime = "video/mp4"
            desc = vision.describe_video(raw, mime)
            kind = "video"
        else:
            return jsonify({"status": "error", "message": f"不支持的文件类型: {mime or fname}"}), 400

        return jsonify({
            "status": "ok",
            "kind": kind,
            "filename": f.filename,
            "description": desc,
            "vision_configured": vision.is_configured(),
        })
    except Exception as e:
        logger.exception("视觉识别失败")
        return jsonify({"status": "error", "message": f"视觉识别失败: {e}"}), 500


@app.get("/api/industries")
def industries():
    return jsonify({
        "status": "ok",
        "industries": [_industry_meta(industry) for industry in INDUSTRY_LABELS],
    })


@app.get("/api/personas")
def get_personas():
    industry = str(request.args.get("industry", "auto")).strip()
    if industry not in INDUSTRY_LABELS:
        return jsonify({"status": "error", "message": "未知行业"}), 400
    personas = _load_personas(industry)
    return jsonify({
        "status": "ok",
        "industry": industry,
        "industry_name": INDUSTRY_LABELS[industry],
        "personas": personas,
        "has_personas": bool(personas),
        "file": os.path.relpath(_industry_file(industry), BASE_DIR).replace("\\", "/"),
    })


@app.post("/api/personas")
def save_personas():
    body = request.get_json(silent=True) or {}
    industry = str(body.get("industry", "auto")).strip()
    if industry not in INDUSTRY_LABELS:
        return jsonify({"status": "error", "message": "未知行业"}), 400

    personas = _normalize_personas(body.get("personas", []))
    _save_personas(industry, personas)
    return jsonify({
        "status": "ok",
        "industry": industry,
        "industry_name": INDUSTRY_LABELS[industry],
        "count": len(personas),
    })


@app.delete("/api/personas")
def delete_personas():
    industry = str(request.args.get("industry", "auto")).strip()
    if industry not in INDUSTRY_LABELS:
        return jsonify({"status": "error", "message": "未知行业"}), 400
    _clear_personas(industry)
    return jsonify({
        "status": "ok",
        "industry": industry,
        "industry_name": INDUSTRY_LABELS[industry],
        "count": 0,
    })


@app.post("/api/personas/generate")
def generate_personas():
    body = request.get_json(silent=True) or {}
    industry = str(body.get("industry", "auto")).strip()
    if industry not in INDUSTRY_LABELS:
        return jsonify({"status": "error", "message": "未知行业"}), 400

    num_users = int(body.get("num_users", 5))
    num_users = max(1, num_users)
    append_existing = bool(body.get("append_existing", False))

    try:
        logger.info("开始为行业 %s 生成 %s 个虚拟用户规格", industry, num_users)
        new_personas = simulation.generate_virtual_user_specs(
            num_users=num_users,
            industry=industry,
        )
        personas = _load_personas(industry) + new_personas if append_existing else new_personas
        _save_personas(industry, personas)
        logger.info("行业 %s 虚拟用户规格已保存，共 %s 个", industry, len(personas))
        return jsonify({
            "status": "ok",
            "industry": industry,
            "industry_name": INDUSTRY_LABELS[industry],
            "count": len(personas),
            "added_count": len(new_personas),
            "personas": personas,
        })
    except Exception as e:
        logger.exception("生成虚拟用户规格失败")
        return jsonify({"status": "error", "message": f"生成失败: {e}"}), 500


# ======================================================================
# 创建虚拟市场（加载已保存虚拟用户）
# ======================================================================
@app.post("/api/create_market")
def create_market():
    body = request.get_json(silent=True) or {}
    num_users = int(body.get("num_users", 5))
    num_users = max(1, min(num_users, 15))
    industry = str(body.get("industry", "auto")).strip()

    if industry not in INDUSTRY_LABELS:
        return jsonify({"status": "error", "message": "未知行业"}), 400

    if _state["status"] == "creating":
        return jsonify({"status": "error", "message": "正在创建中，请稍候"}), 409

    personas = _load_personas(industry)
    if not personas:
        return jsonify({
            "status": "error",
            "message": "当前行业还没有虚拟用户，请先到设置页创建",
            "code": "NO_PERSONAS",
        }), 400

    _state["status"] = "creating"
    try:
        logger.info(f"开始创建行业 {industry} 的虚拟市场，加载已保存的 {len(personas)} 个虚拟用户...")
        users = simulation.load_virtual_users_from_specs(personas)

        _state["users"] = users
        _state["industry"] = industry
        _state["real_user_count"] = len(users)
        _state["total_users"] = len(users)
        _state["status"] = "ready"

        profiles = simulation.get_user_profiles(users, extended=False)
        logger.info(f"虚拟用户创建完成，共 {len(users)} 个")
        return jsonify({
            "status": "ok",
            "industry": industry,
            "industry_name": INDUSTRY_LABELS[industry],
            "real_user_count": len(users),
            "total_users": len(users),
            "profiles": [
                {"name": p["name"], "minibio": p["minibio"][:200]}
                for p in profiles
            ],
        })
    except Exception as e:
        _state["status"] = "idle"
        logger.exception("创建虚拟用户失败")
        return jsonify({"status": "error", "message": f"创建失败: {e}"}), 500


# ======================================================================
# 评价新车概念
# ======================================================================
def _extract_concept(body: Dict[str, Any]) -> Dict[str, Any]:
    """从前端请求体中规范化概念参数。"""
    colors = body.get("colors", []) or []
    if not isinstance(colors, list):
        colors = []
    colors = [str(item).strip() for item in colors if str(item).strip()]

    concept = {
        "image_source_type": str(body.get("image_source_type", "")).strip(),
        "image_name": str(body.get("image_name", "")).strip(),
        "image_note": str(body.get("image_note", "")).strip(),
        "video_url": str(body.get("video_url", "")).strip(),
        "video_name": str(body.get("video_name", "")).strip(),
        "vision_description": str(body.get("vision_description", "")).strip(),
        "brand": str(body.get("brand", "")).strip(),
        "product_plan": str(body.get("product_plan", "")).strip(),
        "colors": colors,
        "color": str(body.get("color", "")).strip() or "、".join(colors),
        "size": str(body.get("size", "")).strip(),
        "power": str(body.get("power", "")).strip(),
        "range": int(body.get("range", 780)),
        "price": int(body.get("price", 28)),
        "audience": str(body.get("audience", "")).strip(),
        "tags": body.get("tags", []) or [],
        # 实验室调参维度（首次 analyze 时由前端传入默认值）
        "charge": int(body.get("charge", 15)),
        "smart": int(body.get("smart", 7)),
        "space": int(body.get("space", 6)),
        "lux": int(body.get("lux", 5)),
    }
    return concept


@app.post("/api/analyze")
def analyze():
    if not _state["users"]:
        return jsonify({"status": "error", "message": "请先创建虚拟市场"}), 400
    if _state["status"] == "analyzing":
        return jsonify({"status": "error", "message": "正在分析中，请稍候"}), 409

    body = request.get_json(silent=True) or {}
    concept = _extract_concept(body)
    _state["last_concept"] = concept
    _state["status"] = "analyzing"

    def generate():
        try:
            logger.info(f"开始流式评价新车概念，共 {len(_state['users'])} 个虚拟用户")
            all_results = []
            for event in simulation.evaluate_car_stream(concept, _state["users"]):
                if event["type"] == "user_done":
                    all_results.append(event["result"])
                yield json.dumps(event, ensure_ascii=False) + "\n"

            # 聚合最终结果
            data = aggregator.aggregate(
                agent_results=all_results,
                total_users=_state["total_users"],
                params=concept,
            )
            _state["status"] = "ready"
            _state["report"] = _generate_report(concept, data, all_results)
            logger.info(f"评价完成，NPS={data['nps']}, overall={data['overall']}")
            yield json.dumps({"type": "complete", "data": data}, ensure_ascii=False) + "\n"
        except Exception as e:
            _state["status"] = "ready"
            logger.exception("评价失败")
            yield json.dumps({"type": "error", "message": f"评价失败: {e}"}, ensure_ascii=False) + "\n"
        finally:
            # 客户端断开等场景下（GeneratorExit 不被 except Exception 捕获）兜底重置状态
            if _state["status"] == "analyzing":
                _state["status"] = "ready"
                logger.warning("流式评价被中断（客户端断开？），状态已重置为 ready")

    return Response(
        stream_with_context(generate()),
        mimetype="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/report")
def get_report():
    """下载最近一次分析报告（纯文本）。"""
    if not _state["report"]:
        return jsonify({"status": "error", "message": "暂无分析报告，请先执行分析"}), 400
    return Response(
        _state["report"],
        mimetype="text/plain; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=analysis_report.txt"},
    )


@app.get("/api/report/preview")
def preview_report():
    """预览最近一次分析报告（JSON 返回文本）。"""
    if not _state["report"]:
        return jsonify({"status": "error", "message": "暂无分析报告"}), 400
    return jsonify({"status": "ok", "report": _state["report"]})


def _generate_report(concept: Dict[str, Any], data: Dict[str, Any], agent_results: List[Dict[str, Any]]) -> str:
    """根据虚拟用户评分及评价整理分析报告（500-800字）。"""
    # 拼接产品名：品牌+动力+尺寸+目标人群（如"蔚来·纯电中型SUV（面向都市白领）"），无则降级
    brand = concept.get("brand", "") or ""
    parts = [p for p in [
        brand,
        concept.get("power", ""),
        concept.get("size", ""),
    ] if p]
    name = ("·".join(parts) + "（面向" + concept.get("audience", "泛大众") + "）") if parts else "新车概念"
    industry = INDUSTRY_LABELS.get(_state.get("industry", "auto"), "汽车")
    nps = data.get("nps", 0)
    pos = data.get("pos", 0)
    neu = data.get("neu", 0)
    neg = data.get("neg", 0)
    overall = data.get("overall", 0)
    praise = data.get("praise", [])
    complain = data.get("complain", [])
    funnel = [data.get("cognition", 0), data.get("interest", 0),
              data.get("intent", 0), data.get("trial", 0), data.get("prebuy", 0)]
    user_count = len(agent_results)
    avg_rating = sum(r.get("rating", 3) for r in agent_results) / max(user_count, 1)

    # radar 是 List[float]（顺序对应 RADAR_DIMS），转换为带标签的结构便于报告引用
    radar_raw = data.get("radar", [])
    radar_dims = ["续航", "智能", "空间", "豪华", "性价比", "外观"]
    if radar_raw and isinstance(radar_raw[0], (int, float)):
        radar = [{"label": dim, "v": float(v)} for dim, v in zip(radar_dims, radar_raw)]
    else:
        radar = radar_raw  # 兜底：已经是 dict 列表

    lines = []
    lines.append(f"《{name}》{industry}市场虚拟用户分析报告")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"样本规模：{user_count} 名虚拟用户")
    lines.append(f"平均评分：{avg_rating:.1f} / 5.0")
    lines.append(f"综合 NPS 指数：{nps}")
    lines.append("")
    lines.append("一、整体市场反馈")
    lines.append("-" * 40)
    if nps >= 50:
        lines.append(f"本次分析显示，虚拟用户对「{name}」的整体反馈偏向正面。"
                     f"正面情绪占比 {pos}%，中立 {neu}%，负面 {neg}%。"
                     f"产品在目标群体中获得了较高的认可度，具备一定的市场潜力。")
    elif nps >= 20:
        lines.append(f"虚拟用户对「{name}」的评价呈现分化态势。"
                     f"正面 {pos}%、中立 {neu}%、负面 {neg}%，NPS 为 {nps}。"
                     f"产品有亮点但也存在明显短板，需针对性优化。")
    else:
        lines.append(f"虚拟用户对「{name}」的整体反馈偏负面。"
                     f"正面仅 {pos}%，负面达 {neg}%，NPS 为 {nps}。"
                     f"产品在当前定位下市场接受度较低，建议重新审视产品策略。")

    lines.append("")
    lines.append("二、六维度评分分析")
    lines.append("-" * 40)
    if radar:
        for dim in radar:
            label = dim.get("label", "")
            val = dim.get("v", 0)
            bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
            lines.append(f"  {label}  {bar}  {val*100:.0f}")
        top_dim = max(radar, key=lambda d: d.get("v", 0))
        low_dim = min(radar, key=lambda d: d.get("v", 0))
        lines.append(f"\n最高维度：{top_dim.get('label','')}（{top_dim.get('v',0)*100:.0f}），"
                     f"是产品的核心竞争优势。")
        lines.append(f"最低维度：{low_dim.get('label','')}（{low_dim.get('v',0)*100:.0f}），"
                     f"建议优先改进。")

    lines.append("")
    lines.append("三、用户赞赏要点")
    lines.append("-" * 40)
    if praise:
        for i, p in enumerate(praise[:5], 1):
            lines.append(f"  {i}. {p.get('text','')}（{p.get('n',0)} 人提及）")
    else:
        lines.append("  暂无显著赞赏点。")

    lines.append("")
    lines.append("四、用户吐槽要点")
    lines.append("-" * 40)
    if complain:
        for i, c in enumerate(complain[:5], 1):
            lines.append(f"  {i}. {c.get('text','')}（{c.get('n',0)} 人提及）")
    else:
        lines.append("  暂无显著吐槽点。")

    lines.append("")
    lines.append("五、购买漏斗分析")
    lines.append("-" * 40)
    funnel_labels = ["认知", "兴趣", "意向", "试驾", "预购"]
    for i, (label, val) in enumerate(zip(funnel_labels, funnel)):
        lines.append(f"  {label}阶段：{val}%")
    drop_off = funnel[0] - funnel[-1] if funnel[0] > 0 else 0
    lines.append(f"\n从认知到预购的转化流失率为 {drop_off}%。"
                 f"{'转化效率良好。' if drop_off < 50 else '转化流失较大，需在兴趣和意向阶段加强引导。'}")

    lines.append("")
    lines.append("六、综合建议")
    lines.append("-" * 40)
    suggestions = []
    if radar:
        low_dim = min(radar, key=lambda d: d.get("v", 0))
        suggestions.append(f"重点提升「{low_dim.get('label','')}」维度表现")
    if complain:
        suggestions.append(f"针对「{complain[0].get('text','')}」等核心痛点进行优化")
    if nps < 30:
        suggestions.append("考虑调整定价策略或产品定位以提升市场接受度")
    if funnel[-1] < 20:
        suggestions.append("加强预购转化引导，优化试驾到预购的衔接环节")
    if not suggestions:
        suggestions.append("保持当前产品策略，持续监控市场反馈")

    for i, s in enumerate(suggestions, 1):
        lines.append(f"  {i}. {s}")

    lines.append("")
    lines.append("=" * 50)
    lines.append("本报告由 AI 虚拟市场洞察平台自动生成")
    lines.append(f"样本数：{user_count} | NPS：{nps} | 综合评分：{overall}")

    return "\n".join(lines)


# ======================================================================
# 调参重算
# ======================================================================
@app.post("/api/tune")
def tune():
    if not _state["users"] or not _state["last_concept"]:
        return jsonify({"status": "error", "message": "请先完成一次分析"}), 400
    if _state["status"] == "analyzing":
        return jsonify({"status": "error", "message": "正在分析中，请稍候"}), 409

    body = request.get_json(silent=True) or {}
    tune_params = {
        "range": int(body.get("range", _state["last_concept"]["range"])),
        "price": int(body.get("price", _state["last_concept"]["price"])),
        "charge": int(body.get("charge", _state["last_concept"]["charge"])),
        "smart": int(body.get("smart", _state["last_concept"]["smart"])),
        "space": int(body.get("space", _state["last_concept"]["space"])),
        "lux": int(body.get("lux", _state["last_concept"]["lux"])),
    }

    # 用调参后的值覆盖原概念，其它字段（product_plan/color/...）保持不变
    concept = dict(_state["last_concept"])
    concept.update(tune_params)
    _state["last_concept"] = concept
    _state["status"] = "analyzing"

    try:
        logger.info(f"调参重算: {tune_params}")
        agent_results = simulation.evaluate_car(concept, _state["users"])
        data = aggregator.aggregate(
            agent_results=agent_results,
            total_users=_state["total_users"],
            params=tune_params,
        )
        _state["status"] = "ready"
        _state["report"] = _generate_report(concept, data, agent_results)
        return jsonify({"status": "ok", "data": data, "_tuned": True})
    except Exception as e:
        _state["status"] = "ready"
        logger.exception("调参重算失败")
        return jsonify({"status": "error", "message": f"调参失败: {e}"}), 500


# ======================================================================
# 查看虚拟用户画像
# ======================================================================
@app.get("/api/users")
def users():
    if not _state["users"]:
        return jsonify({"status": "ok", "industry": _state["industry"], "users": []})
    profiles = simulation.get_user_profiles(_state["users"], extended=False)
    return jsonify({
        "status": "ok",
        "industry": _state["industry"],
        "real_user_count": _state["real_user_count"],
        "total_users": _state["total_users"],
        "users": [
            {"name": p["name"], "minibio": p["minibio"]}
            for p in profiles
        ],
    })


# ======================================================================
# 根路径：返回 ui.html
# ======================================================================
@app.get("/")
def index():
    ui_path = os.path.join(BASE_DIR, "ui.html")
    if os.path.exists(ui_path):
        return send_file(ui_path)
    return "ui.html not found", 404


@app.get("/settings")
def settings_page():
    page_path = os.path.join(BASE_DIR, "settings.html")
    if os.path.exists(page_path):
        return send_file(page_path)
    return "settings.html not found", 404


@app.get("/images/<path:filename>")
def serve_image(filename: str):
    if not os.path.isdir(IMAGES_DIR):
        return "images directory not found", 404
    return send_from_directory(IMAGES_DIR, filename)


# ======================================================================
# 启动
# ======================================================================
if __name__ == "__main__":
    # 启动时打印一次配置摘要，便于排查
    api_key_set = bool(os.getenv("OPENAI_API_KEY"))
    base_url = os.getenv("OPENAI_BASE_URL", "")
    logger.info("=" * 60)
    logger.info("AI 虚拟市场洞察平台 · Flask 后端启动")
    logger.info(f"  OPENAI_API_KEY 已配置: {api_key_set}")
    logger.info(f"  OPENAI_BASE_URL    : {base_url or '(默认 api.openai.com)'}")
    logger.info(f"  虚拟用户数          : 5 (可在请求 num_users 中调整)")
    logger.info("=" * 60)
    if not api_key_set:
        logger.warning("⚠ 未检测到 OPENAI_API_KEY，请先复制 .env.example 为 .env 并填写")

    app.run(host="127.0.0.1", port=5003, debug=False, threaded=True)
