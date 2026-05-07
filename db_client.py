"""ダッシュボード用 軽量Supabaseクライアント

pipeline/db_writer.py は YOLO/MediaPipe 等の重量ライブラリに連鎖する import を持つため、
Streamlit Cloud 等の軽量デプロイ環境ではこちらを使う。
supabase-py のみに依存する。

DEMO モード:
    SUPABASE_URL/KEY が未設定の場合、ハードコードされたダミーデータを返す。
    商談デモ・開発時の動作確認用。本番では Supabase 接続を有効化する。
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any


# ============================================================
# DEMO データ (Supabase 未設定時に返却される架空データ)
# ============================================================

DEMO_CLASSROOMS: list[dict] = [
    {"id": "demo-c-toyohashi", "name": "豊橋本校", "region": "愛知県東部",
     "is_active": True, "manager_teacher_id": "demo-t-tanaka",
     "created_at": "2026-04-01T00:00:00Z"},
    {"id": "demo-c-toyota-kita", "name": "豊田北校", "region": "愛知県中部",
     "is_active": True, "manager_teacher_id": "demo-t-suzuki",
     "created_at": "2026-04-01T00:00:00Z"},
    {"id": "demo-c-mihongi", "name": "三本木校", "region": "愛知県中部",
     "is_active": True, "manager_teacher_id": "demo-t-yamada",
     "created_at": "2026-04-01T00:00:00Z"},
    {"id": "demo-c-okazaki", "name": "岡崎校", "region": "愛知県西部",
     "is_active": True, "manager_teacher_id": None,
     "created_at": "2026-04-01T00:00:00Z"},
]

DEMO_TEACHERS: list[dict] = [
    {"id": "demo-t-tanaka", "name": "田中太郎", "classroom_id": "demo-c-toyohashi",
     "rank": "中堅", "role": "社員", "is_active": True,
     "email": "tanaka@kaitakujyuku.example", "hired_at": "2022-04-01"},
    {"id": "demo-t-sato", "name": "佐藤花子", "classroom_id": "demo-c-toyohashi",
     "rank": "ベテラン", "role": "社員", "is_active": True,
     "email": "sato@kaitakujyuku.example", "hired_at": "2018-04-01"},
    {"id": "demo-t-suzuki", "name": "鈴木次郎", "classroom_id": "demo-c-toyota-kita",
     "rank": "新人", "role": "社員", "is_active": True,
     "email": "suzuki@kaitakujyuku.example", "hired_at": "2026-04-01"},
    {"id": "demo-t-yamada", "name": "山田一郎", "classroom_id": "demo-c-mihongi",
     "rank": "教室長", "role": "教室長", "is_active": True,
     "email": "yamada@kaitakujyuku.example", "hired_at": "2015-04-01"},
    {"id": "demo-t-takahashi", "name": "高橋美咲", "classroom_id": "demo-c-toyota-kita",
     "rank": "中堅", "role": "社員", "is_active": True,
     "email": "takahashi@kaitakujyuku.example", "hired_at": "2023-04-01"},
]


def _demo_lessons() -> list[dict]:
    """4本の実E2E動画＋過去授業の架空データ。週次ヒートマップが見える程度に分散させる。"""
    today = date.today()
    rows: list[dict] = []

    # 実 E2E 動画 4本に対応する直近授業
    real_lessons = [
        {
            "id": "demo-l-toyohashi-good",
            "teacher_id": "demo-t-tanaka", "classroom_id": "demo-c-toyohashi",
            "lesson_date": (today - timedelta(days=1)).isoformat(),
            "subject": "英語", "grade": "中1", "student_count": 18,
            "video_filename": "豊橋本校中１良い例.mkv",
            "video_duration_sec": 365, "overall_score": 78,
            "grade_letter": "B+", "ai_commentary": "全体的に統一感のある授業進行。指示の揃え方が良好。",
            "good_points": ["挨拶のこだわりが高い", "テンポが安定"],
            "improvements": ["熱量・パンチをもう一段"],
        },
        {
            "id": "demo-l-toyota-normal",
            "teacher_id": "demo-t-takahashi", "classroom_id": "demo-c-toyota-kita",
            "lesson_date": (today - timedelta(days=2)).isoformat(),
            "subject": "数学A", "grade": "中3", "student_count": 22,
            "video_filename": "豊田北中3_.mkv",
            "video_duration_sec": 2700, "overall_score": 71,
            "grade_letter": "B", "ai_commentary": "標準的な進行。問いかけ計16回（open 2 / closed 14）で生徒参加は中程度。",
            "good_points": ["時間配分が適切"],
            "improvements": ["open question を増やすと深い対話に"],
        },
        {
            "id": "demo-l-toyota-noisy",
            "teacher_id": "demo-t-suzuki", "classroom_id": "demo-c-toyota-kita",
            "lesson_date": (today - timedelta(days=3)).isoformat(),
            "subject": "数学A", "grade": "中3", "student_count": 22,
            "video_filename": "豊田北中3_ざわつき.mkv",
            "video_duration_sec": 2700, "overall_score": 58,
            "grade_letter": "C", "ai_commentary": "ざわつき発生時の講師反応にラグ。新人講師として要1on1。",
            "good_points": ["指示の多さは適切"],
            "improvements": ["ざわつき検知時の即時対応", "見回り頻度を上げる"],
        },
        {
            "id": "demo-l-mihongi-late",
            "teacher_id": "demo-t-yamada", "classroom_id": "demo-c-mihongi",
            "lesson_date": (today - timedelta(days=4)).isoformat(),
            "subject": "英語", "grade": "中3", "student_count": 16,
            "video_filename": "三本木中3_遅刻.mkv",
            "video_duration_sec": 2700, "overall_score": 73,
            "grade_letter": "B", "ai_commentary": "遅刻者対応も含め全体安定。教室長らしい統率力。",
            "good_points": ["遅刻対応がスルーされていない", "授業全体のリズム良好"],
            "improvements": ["生徒との向き合いをもう一段（特に後列）"],
        },
    ]
    rows.extend(real_lessons)

    # 過去4週間 × 5講師 = 各講師4授業の架空履歴
    base_scores = {
        "demo-t-tanaka": [75, 78, 80, 78],
        "demo-t-sato": [85, 87, 88, 90],
        "demo-t-suzuki": [55, 60, 58, 58],
        "demo-t-yamada": [82, 80, 85, 73],
        "demo-t-takahashi": [70, 72, 71, 71],
    }
    grade_for = lambda s: "S" if s >= 90 else "A" if s >= 80 else "B+" if s >= 75 else "B" if s >= 65 else "C" if s >= 55 else "D"
    for tid, scores in base_scores.items():
        teacher = next((t for t in DEMO_TEACHERS if t["id"] == tid), None)
        if not teacher:
            continue
        for week_offset, sc in enumerate(scores):
            ld = today - timedelta(days=7 * (week_offset + 1) + 5)
            rows.append({
                "id": f"demo-l-{tid}-w{week_offset}",
                "teacher_id": tid, "classroom_id": teacher["classroom_id"],
                "lesson_date": ld.isoformat(),
                "subject": "英語" if week_offset % 2 == 0 else "数学A",
                "grade": "中1" if tid == "demo-t-tanaka" else "中3",
                "student_count": 20,
                "video_filename": f"archive_{tid}_w{week_offset}.mp4",
                "video_duration_sec": 2700, "overall_score": sc,
                "grade_letter": grade_for(sc),
                "ai_commentary": "（過去授業のサマリー）",
                "good_points": [], "improvements": [],
            })
    return rows


DEMO_LESSONS = _demo_lessons()


def is_demo_mode() -> bool:
    """SUPABASE_URL/KEY 未設定時は True。デモデータで動作する。"""
    return not (os.getenv("SUPABASE_URL") and (os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_KEY")))


def _attach_classroom_name(teacher: dict) -> dict:
    """teacher dict に classrooms.name を合成（DBクエリのJOIN代替）"""
    cid = teacher.get("classroom_id")
    name = next((c["name"] for c in DEMO_CLASSROOMS if c["id"] == cid), "—") if cid else "—"
    return {**teacher, "classrooms": {"name": name}}


def _attach_lesson_relations(lesson: dict) -> dict:
    """lesson dict に teachers.name と classrooms.name を合成"""
    tid = lesson.get("teacher_id")
    cid = lesson.get("classroom_id")
    teacher_name = next((t["name"] for t in DEMO_TEACHERS if t["id"] == tid), "—") if tid else "—"
    classroom_name = next((c["name"] for c in DEMO_CLASSROOMS if c["id"] == cid), "—") if cid else "—"
    return {
        **lesson,
        "teachers": {"name": teacher_name, "classroom_id": cid},
        "classrooms": {"name": classroom_name},
    }


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
    """DB接続またはデモモードのいずれかで利用可能なら True"""
    return _get_client() is not None or is_demo_mode()


def fetch_all_teachers() -> list[dict]:
    """講師一覧（is_activeフィルタ + 教室名合成）。JOIN回避で安定化。"""
    if is_demo_mode():
        return [_attach_classroom_name(t) for t in DEMO_TEACHERS if t.get("is_active")]
    client = _get_client()
    if client is None:
        return []
    try:
        result = (
            client.table("teachers")
            .select("*")
            .eq("is_active", True)
            .order("name")
            .execute()
        )
        rows = result.data or []
        # classroom 名を別クエリで取得し合成
        if rows:
            rooms_result = client.table("classrooms").select("id, name").execute()
            room_map = {r["id"]: r["name"] for r in (rooms_result.data or [])}
            for r in rows:
                cid = r.get("classroom_id")
                r["classrooms"] = {"name": room_map.get(cid, "—")} if cid else {"name": "—"}
        return rows
    except Exception as exc:  # noqa: BLE001
        # デバッグのため stderr に出す
        import sys
        print(f"[fetch_all_teachers ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return []


def fetch_all_classrooms() -> list[dict]:
    if is_demo_mode():
        return [c for c in DEMO_CLASSROOMS if c.get("is_active")]
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
    except Exception as exc:  # noqa: BLE001
        import sys
        print(f"[fetch_all_classrooms ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return []


def fetch_teacher_history(teacher_id: str, weeks: int = 12) -> list[dict]:
    if is_demo_mode():
        rows = [l for l in DEMO_LESSONS if l.get("teacher_id") == teacher_id]
        return sorted(rows, key=lambda l: l.get("lesson_date", ""), reverse=True)[: weeks * 5]
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
    if is_demo_mode():
        rows = [_attach_lesson_relations(l) for l in DEMO_LESSONS]
        return sorted(rows, key=lambda l: l.get("lesson_date", ""), reverse=True)[:limit]
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
    if is_demo_mode():
        match = next((l for l in DEMO_LESSONS if l.get("id") == lesson_id), None)
        return _attach_lesson_relations(match) if match else None
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


def fetch_check_item_events(lesson_id: str) -> list[dict]:
    """12項目×秒数の良例/悪例マーカー一覧。

    schema_v3_timeline.sql で定義した `check_item_events` テーブルから読み取る。
    テーブル未作成・権限不足の場合は空配列を返して UI 側で静かにスキップ。
    """
    client = _get_client()
    if client is None:
        return []
    try:
        result = (
            client.table("check_item_events")
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
    from uuid import uuid4
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    # Supabase Storage は非ASCII keyを拒否するため UUID baseに。
    # 元のファイル名は lessons.video_filename に保存される。
    ext = original_filename.rsplit(".", 1)[-1].lower() if "." in original_filename else "mp4"
    storage_key = f"inbox/{ts}_{uuid4().hex[:8]}.{ext}"

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
    # Phase A: video_storage_path も保存。private bucket 化後は signed URL 発行に必要。
    try:
        payload = {
            "teacher_id": teacher_id,
            "classroom_id": classroom_id,
            "lesson_date": lesson_date,
            "subject": subject,
            "grade": grade,
            "student_count": student_count,
            "video_filename": original_filename,
            "video_url": storage_url,                  # 旧互換（public URL）
            "video_storage_path": storage_key,         # Phase A: signed URL 発行用
            "status": "pending",
            "notes": notes,
        }
        try:
            result = client.table("lessons").insert(payload).execute()
        except Exception as exc:  # noqa: BLE001
            # video_storage_path カラムが未デプロイの環境では除外して再試行
            if "video_storage_path" in str(exc).lower() or "schema cache" in str(exc).lower():
                payload.pop("video_storage_path", None)
                result = client.table("lessons").insert(payload).execute()
            else:
                raise
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
