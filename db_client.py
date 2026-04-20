"""ダッシュボード用 軽量Supabaseクライアント v2

v2 changelog:
- fetch_lesson_detail / fetch_events_for_lesson 追加（Phase 6 動画プレーヤー）

pipeline/db_writer.py は YOLO/MediaPipe 等の重量ライブラリに連鎖する import を持つため、
Streamlit Cloud 等の軽量デプロイ環境ではこちらを使う。
supabase-py のみに依存する。
"""
from __future__ import annotations

import os
from typing import Any


def _get_client() -> Any | None:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return None
    try:
        from supabase import create_client  # type: ignore
    except ImportError:
        return None
    return create_client(url, key)


def is_db_available() -> bool:
    return _get_client() is not None


def fetch_all_teachers() -> list[dict]:
    client = _get_client()
    if client is None:
        return []
    try:
        result = (
            client.table("teachers")
            .select("*, classrooms(name)")
            .eq("is_active", True)
            .order("name")
            .execute()
        )
        return result.data or []
    except Exception:  # noqa: BLE001
        return []


def fetch_all_classrooms() -> list[dict]:
    client = _get_client()
    if client is None:
        return []
    try:
        result = (
            client.table("classrooms")
            .select("*")
            .eq("is_active", True)
            .order("name")
            .execute()
        )
        return result.data or []
    except Exception:  # noqa: BLE001
        return []


def fetch_teacher_history(teacher_id: str, weeks: int = 12) -> list[dict]:
    client = _get_client()
    if client is None:
        return []
    try:
        result = (
            client.table("lessons")
            .select("*")
            .eq("teacher_id", teacher_id)
            .order("lesson_date", desc=True)
            .limit(weeks * 5)
            .execute()
        )
        return result.data or []
    except Exception:  # noqa: BLE001
        return []


def fetch_all_lessons(limit: int = 5000) -> list[dict]:
    client = _get_client()
    if client is None:
        return []
    try:
        result = (
            client.table("lessons")
            .select("*, teachers(name, classroom_id), classrooms(name)")
            .order("lesson_date", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception:  # noqa: BLE001
        return []


def fetch_checklist_scores(lesson_ids: list[str]) -> list[dict]:
    if not lesson_ids:
        return []
    client = _get_client()
    if client is None:
        return []
    try:
        result = (
            client.table("checklist_scores")
            .select("*")
            .in_("lesson_id", lesson_ids)
            .execute()
        )
        return result.data or []
    except Exception:  # noqa: BLE001
        return []


def fetch_lesson_detail(lesson_id: str) -> dict | None:
    """特定授業の詳細（動画URL込み）を取得"""
    client = _get_client()
    if client is None:
        return None
    try:
        result = (
            client.table("lessons")
            .select("*, teachers(name), classrooms(name)")
            .eq("id", lesson_id)
            .limit(1)
            .execute()
        )
        rows = result.data or []
        return rows[0] if rows else None
    except Exception:  # noqa: BLE001
        return None


def fetch_events_for_lesson(lesson_id: str) -> list[dict]:
    """授業の検知イベント一覧（タイムスタンプ・severity・Visionコメント）"""
    client = _get_client()
    if client is None:
        return []
    try:
        result = (
            client.table("events")
            .select("*")
            .eq("lesson_id", lesson_id)
            .order("start_sec")
            .execute()
        )
        return result.data or []
    except Exception:  # noqa: BLE001
        return []


def fetch_alerts(active_only: bool = True) -> list[dict]:
    client = _get_client()
    if client is None:
        return []
    try:
        query = client.table("alerts").select("*, teachers(name)").order("triggered_at", desc=True)
        if active_only:
            query = query.is_("acknowledged_at", "null")
        result = query.limit(200).execute()
        return result.data or []
    except Exception:  # noqa: BLE001
        return []
