"""BeforeAfter 比較 — 講師の成長を定量化

「指導前の授業」と「指導後の授業」を比較し、Claudeに成長分析レポートを書かせる。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass


COMPARE_PROMPT = """あなたは開拓塾の授業品質コーチです。
同じ講師の「指導前」と「指導後」の授業を比較し、成長ポイントを分析してください。

【講師名】{teacher_name}

## 指導前（{before_date}）
- 総合スコア: {before_score} ({before_grade})
- 良かった点: {before_good}
- 改善提案: {before_improvements}

## 指導後（{after_date}）
- 総合スコア: {after_score} ({after_grade})
- 良かった点: {after_good}
- 改善提案: {after_improvements}

## 12項目 比較
{checklist_diff}

以下のJSON形式で返答してください（JSON以外は一切書かないこと）:
{{
  "growth_score": 0-100の整数 (成長度の定量評価),
  "biggest_improvement": "最も伸びた項目名",
  "biggest_drop": "最も下がった項目名 or null",
  "summary": "2〜3文で成長概況を",
  "action_for_next": "次に伸ばすべき1項目と具体アクション",
  "celebrate": "本人に伝えたい褒めコメント（1〜2文・誠実に）"
}}
"""


@dataclass
class ComparisonResult:
    growth_score: int
    biggest_improvement: str
    biggest_drop: str | None
    summary: str
    action_for_next: str
    celebrate: str
    error: str | None = None


def compare(*, teacher_name: str,
            before_lesson: dict, after_lesson: dict,
            before_checklist: list[dict], after_checklist: list[dict],
            model: str | None = None) -> ComparisonResult:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return ComparisonResult(0, "—", None, "APIキー未設定", "—", "—",
                                error="no api key")
    try:
        from anthropic import Anthropic
    except ImportError:
        return ComparisonResult(0, "—", None, "", "—", "—",
                                error="anthropic not installed")

    # 12項目の差分を整形
    before_map = {c["item_id"]: c for c in before_checklist}
    after_map = {c["item_id"]: c for c in after_checklist}
    diff_lines = []
    for iid in sorted(set(before_map) | set(after_map)):
        b = before_map.get(iid)
        a = after_map.get(iid)
        title = (a or b or {}).get("item_title", f"項目{iid}")
        bs = b["score"] if b else None
        as_ = a["score"] if a else None
        delta = (as_ - bs) if (bs is not None and as_ is not None) else None
        delta_str = f"{delta:+d}" if delta is not None else "—"
        bs_disp = bs if bs is not None else "—"
        as_disp = as_ if as_ is not None else "—"
        diff_lines.append(f"  {iid}. {title}: {bs_disp} → {as_disp} ({delta_str})")

    prompt = COMPARE_PROMPT.format(
        teacher_name=teacher_name,
        before_date=before_lesson.get("lesson_date", "—"),
        before_score=before_lesson.get("overall_score", "—"),
        before_grade=before_lesson.get("grade_letter", "—"),
        before_good=before_lesson.get("good_points") or [],
        before_improvements=before_lesson.get("improvements") or [],
        after_date=after_lesson.get("lesson_date", "—"),
        after_score=after_lesson.get("overall_score", "—"),
        after_grade=after_lesson.get("grade_letter", "—"),
        after_good=after_lesson.get("good_points") or [],
        after_improvements=after_lesson.get("improvements") or [],
        checklist_diff="\n".join(diff_lines),
    )

    client = Anthropic(api_key=api_key)
    model_name = model or os.getenv("CLAUDE_REPORT_MODEL", "claude-sonnet-4-6")
    try:
        msg = client.messages.create(
            model=model_name,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        if not msg.content:
            return ComparisonResult(0, "—", None, "", "—", "—",
                                    error="AIからの応答が空でした")
        first = msg.content[0]
        text = (getattr(first, "text", None) or "").strip()
        if not text:
            return ComparisonResult(0, "—", None, "", "—", "—",
                                    error="AI応答のテキストが空でした")

        import re as _re
        fence_match = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, _re.DOTALL)
        if fence_match:
            text = fence_match.group(1)
        else:
            brace_match = _re.search(r"\{.*\}", text, _re.DOTALL)
            if brace_match:
                text = brace_match.group(0)

        data = json.loads(text)
        return ComparisonResult(
            growth_score=int(data.get("growth_score", 0)),
            biggest_improvement=str(data.get("biggest_improvement", "—")),
            biggest_drop=data.get("biggest_drop"),
            summary=str(data.get("summary", "")),
            action_for_next=str(data.get("action_for_next", "")),
            celebrate=str(data.get("celebrate", "")),
        )
    except Exception as exc:  # noqa: BLE001
        return ComparisonResult(0, "—", None, "", "—", "—",
                                error=f"{type(exc).__name__}: {exc}")
