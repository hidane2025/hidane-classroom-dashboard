"""ダッシュボード用 軽量Supabaseクライアント

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


def create_classroom(*, name: str, region: str | None = None) -> dict:
    """教室マスタ追加（UI用）"""
    client = _get_client()
    if client is None:
        return {"error": "DB未接続"}
    try:
        existing = client.table("classrooms").select("id").eq("name", name).execute()
        if existing.data:
            return {"error": f"既に存在: {name}"}
        result = client.table("classrooms").insert({
            "name": name, "region": region,
        }).execute()
        return {"id": result.data[0]["id"], "name": name} if result.data else {"error": "insert failed"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def create_teacher(*, name: str, classroom_id: str | None,
                   role: str, rank: str,
                   email: str | None = None,
                   line_user_id: str | None = None) -> dict:
    """講師マスタ追加（UI用）"""
    client = _get_client()
    if client is None:
        return {"error": "DB未接続"}
    try:
        existing = client.table("teachers").select("id").eq("name", name).execute()
        if existing.data:
            return {"error": f"既に存在: {name}"}
        result = client.table("teachers").insert({
            "name": name,
            "classroom_id": classroom_id,
            "role": role,
            "rank": rank,
            "email": email,
            "line_user_id": line_user_id,
        }).execute()
        return {"id": result.data[0]["id"], "name": name} if result.data else {"error": "insert failed"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def deactivate_classroom(classroom_id: str) -> bool:
    client = _get_client()
    if client is None:
        return False
    try:
        client.table("classrooms").update({"is_active": False}).eq("id", classroom_id).execute()
        return True
    except Exception:  # noqa: BLE001
        return False


def deactivate_teacher(teacher_id: str) -> bool:
    client = _get_client()
    if client is None:
        return False
    try:
        client.table("teachers").update({"is_active": False}).eq("id", teacher_id).execute()
        return True
    except Exception:  # noqa: BLE001
        return False


def upload_lesson_video(*, file_bytes: bytes, teacher_id: str, classroom_id: str,
                        lesson_date: str, subject: str | None,
                        grade: str | None, student_count: int | None,
                        notes: str | None, original_filename: str) -> dict:
    """Phase 6+: クライアントからの動画アップロード

    1. Supabase Storage inbox/{timestamp}.mp4 に保存
    2. lessons テーブルに status='pending' で挿入
    3. dict で結果を返す（lesson_id, storage_url, error）
    """
    client = _get_client()
    if client is None:
        return {"error": "DB未接続"}

    from datetime import datetime
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    storage_key = f"inbox/{ts}_{original_filename}"

    # 1. Storage にアップロード
    try:
        try:
            client.storage.from_("lesson-videos").remove([storage_key])
        except Exception:  # noqa: BLE001
            pass
        client.storage.from_("lesson-videos").upload(
            path=storage_key,
            file=file_bytes,
            file_options={"content-type": "video/mp4"},
        )
        storage_url = client.storage.from_("lesson-videos").get_public_url(storage_key)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Storage upload 失敗: {type(exc).__name__}: {exc}"}

    # 2. lessons にレコード挿入（analyzer が後で拾う）
    try:
        payload = {
            "teacher_id": teacher_id,
            "classroom_id": classroom_id,
            "lesson_date": lesson_date,
            "subject": subject,
            "grade": grade,
            "student_count": student_count,
            "video_filename": original_filename,
            "video_url": storage_url,
            "status": "pending",
            "notes": notes,
        }
        result = client.table("lessons").insert(payload).execute()
        if not result.data:
            return {"error": "lessons insert empty result", "storage_url": storage_url}
        return {
            "lesson_id": result.data[0]["id"],
            "storage_url": storage_url,
            "status": "pending",
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"DB insert 失敗: {type(exc).__name__}: {exc}",
                "storage_url": storage_url}


def fetch_pending_lessons() -> list[dict]:
    """ヒダネ側 watcher 用: status='pending' の授業一覧"""
    client = _get_client()
    if client is None:
        return []
    try:
        result = (
            client.table("lessons")
            .select("*, teachers(name), classrooms(name)")
            .eq("status", "pending")
            .order("created_at")
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
