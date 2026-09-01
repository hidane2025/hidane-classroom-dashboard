"""使用するAIモデルの定義（唯一の正）

## なぜこのモジュールがあるか

モデル名が 10 箇所にベタ書きされていた（2026-08-06 実測）。
- `pipeline/` 5箇所 / `scripts/` 2箇所 / `run.py` 1箇所 / `model_pricing.py` 2箇所

そのため:
- モデルを更新するたび全箇所を探す必要があり、漏れると古いモデルが残る
- 実際に「`model.replace("opus","sonnet")` で報告用モデル名を派生させる」実装があり、
  Vision を opus-4-8 へ上げた瞬間に**存在しない `claude-sonnet-4-8`** を生んでいた
  （2026-07-02 に撤去）

**モデル名を書いてよいのはこのファイルと config.yaml だけ**にする。
コード側は既定値としてここを参照し、実際の値は config.yaml が上書きする。

## 更新手順

1. claude-api スキルで現行のモデルIDと料金を確認する（記憶で判断しない）
2. ここの定数と `model_pricing.MODEL_PRICING` を同時に更新する
3. `config.yaml` の `vision_judge.model` / `report.ai_summary_model` を更新する
4. `pytest tests/test_models.py` で整合を確認する
"""
from __future__ import annotations

# --- 用途別の既定モデル -----------------------------------------------------

#: 映像フレームの誤検知判定（Vision）。画像を読むので上位モデルを使う。
DEFAULT_VISION_MODEL = "claude-opus-4-8"

#: 文字起こしからの疑惑タグ抽出・AI講評・二段検証。テキストのみ。
DEFAULT_REPORT_MODEL = "claude-sonnet-4-6"

#: 教科分類・問いかけカウントなど軽量な分類処理。
DEFAULT_CLASSIFY_MODEL = "claude-sonnet-4-6"

#: ダッシュボードのAIコーチ・授業比較。
DEFAULT_COACH_MODEL = "claude-sonnet-4-6"


#: 設定ファイル／環境変数のキー名（文字列の散在を防ぐ）
CONFIG_KEY_VISION_MODEL = ("vision_judge", "model")
CONFIG_KEY_REPORT_MODEL = ("report", "ai_summary_model")
ENV_KEY_REPORT_MODEL = "CLAUDE_REPORT_MODEL"


def resolve_vision_model(config: dict | None = None) -> str:
    """Vision判定に使うモデル名を決める（config > 既定値）。"""
    section, key = CONFIG_KEY_VISION_MODEL
    return ((config or {}).get(section) or {}).get(key) or DEFAULT_VISION_MODEL


def resolve_report_model(config: dict | None = None) -> str:
    """レポート生成に使うモデル名を決める（config > 既定値）。

    ⚠️ **Vision のモデル名から文字列操作で派生させてはいけない**。
    以前 `config["vision_judge"]["model"].replace("opus", "sonnet")` という実装があり、
    Vision を opus-4-8 に更新した際に存在しない `claude-sonnet-4-8` を生成していた。
    用途ごとに独立した設定として持つこと。
    """
    section, key = CONFIG_KEY_REPORT_MODEL
    return ((config or {}).get(section) or {}).get(key) or DEFAULT_REPORT_MODEL


def all_default_models() -> dict[str, str]:
    """用途 → 既定モデル名。料金表との整合テストに使う。"""
    return {
        "vision": DEFAULT_VISION_MODEL,
        "report": DEFAULT_REPORT_MODEL,
        "classify": DEFAULT_CLASSIFY_MODEL,
        "coach": DEFAULT_COACH_MODEL,
    }
