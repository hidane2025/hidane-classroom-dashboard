"""AIコーチ対話層

講師の質問に対し、自分の授業履歴・12項目スコア・改善ポイントを踏まえて
Claudeが具体的に助言する。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime


COACH_SYSTEM_PROMPT = """あなたは開拓塾の授業品質コーチです。
対話相手は開拓塾の講師本人です。以下のルールを厳守してください。

# ルール
- 本人を萎縮させない。批判ではなく伴走。
- 曖昧な一般論で終わらせない。具体的な行動レベルで語る。
- 本人のスコア履歴・12項目の強弱・直近の改善点を根拠として示す。
- 1〜2分で読める長さ（日本語300〜500字）に収める。
- 最後に必ず「次の授業でやる1つの具体アクション」を提示する。
- 開拓塾の12項目チェックシート：
  1.挨拶のこだわり 2.テンポ調整 3.元気・楽しさ 4.熱量・パンチ
  5.指示・確認の統一感 6.不規則への注意 7.発声指導 8.生徒との向き合い
  9.見回りの熱量 10.指示の多さ 11.時間配分 12.遅刻対応

# 出力フォーマット（プレーンテキスト・見出しはMarkdownの##で）
## 観察
（データから読み取れること）

## 仮説
（なぜそうなっているかの推定）

## 次の授業でやる1つ
（具体アクション・1行）
"""


@dataclass
class CoachResponse:
    answer: str
    error: str | None = None


def _build_context(history: list[dict], checklist_avg: dict[int, dict]) -> str:
    lines = ["## 最近の授業（最新5件）"]
    for lesson in (history or [])[:5]:
        date_str = lesson.get("lesson_date", "—")
        score = lesson.get("overall_score", "—")
        grade = lesson.get("grade_letter", "—")
        subject = lesson.get("subject") or "—"
        lines.append(f"- {date_str} / {subject} / {score}点 ({grade})")

    lines.append("\n## 12項目の近況（最新5授業平均）")
    for iid in sorted(checklist_avg.keys()):
        info = checklist_avg[iid]
        lines.append(f"  {iid}. {info['title']} = {info['avg_score']:.2f} 点")
    return "\n".join(lines)


def ask_coach(*, teacher_name: str, question: str,
              history: list[dict], checklist_avg: dict[int, dict],
              model: str | None = None) -> CoachResponse:
    """講師の質問に AI コーチとして回答"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return CoachResponse(
            answer="APIキー未設定のためAIコーチを利用できません。"
                   "管理者に ANTHROPIC_API_KEY の設定を依頼してください。",
            error="no api key",
        )

    try:
        from anthropic import Anthropic
    except ImportError:
        return CoachResponse(answer="", error="anthropic package not installed")

    model_name = model or os.getenv("CLAUDE_REPORT_MODEL", "claude-sonnet-4-6")
    client = Anthropic(api_key=api_key)

    context = _build_context(history, checklist_avg)
    user_prompt = (
        f"【講師名】{teacher_name}\n\n"
        f"{context}\n\n"
        f"## 質問\n{question}\n\n"
        "上記データと質問に基づいて、コーチとして回答してください。"
    )

    try:
        message = client.messages.create(
            model=model_name,
            max_tokens=1200,
            system=COACH_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        if not message.content:
            return CoachResponse(answer="", error="AIからの応答が空でした")
        first = message.content[0]
        text = (getattr(first, "text", None) or "").strip()
        if not text:
            return CoachResponse(answer="", error="AI応答のテキストが空でした")
        return CoachResponse(answer=text)
    except Exception as exc:  # noqa: BLE001
        return CoachResponse(answer="", error=f"{type(exc).__name__}: {exc}")
