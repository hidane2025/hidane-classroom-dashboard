"""HIDANE Classroom Intelligence — 3層ダッシュボード

使い方:
    source ../.venv/bin/activate
    streamlit run dashboard/app.py

ビュー:
    - 社長ビュー（全17教室俯瞰）
    - 教室長ビュー（自教室の講師時系列）
    - 講師ビュー（自分の成長記録）
"""
from __future__ import annotations

import os

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

# ダッシュボード側は重量ライブラリ（YOLO/mediapipe等）を読み込まないように
# dashboard.db_client のみから DB ヘルパーを取得する（Streamlit Cloud 対応）
from db_client import (
    fetch_all_classrooms,
    fetch_all_teachers,
    fetch_teacher_history,
    fetch_lesson_detail,
    fetch_events_for_lesson,
    fetch_pending_lessons,
    upload_lesson_video,
)
from ai_coach import ask_coach  # anthropicのみ依存
from compare_lessons import compare  # anthropicのみ依存

load_dotenv(PROJECT_ROOT / ".env", override=True)

# Streamlit Cloud用: st.secrets の値を環境変数に橋渡し（ローカル開発は .env が優先）
try:
    import streamlit as _st_bridge
    for _key in ("ANTHROPIC_API_KEY", "SUPABASE_URL", "SUPABASE_KEY",
                 "SUPABASE_SERVICE_KEY", "SUPABASE_DB_URL",
                 "RESEND_API_KEY", "SENDER_EMAIL", "CEO_EMAIL"):
        if _key in _st_bridge.secrets and not os.getenv(_key):
            os.environ[_key] = str(_st_bridge.secrets[_key])
except Exception:  # noqa: BLE001
    pass


# ==========================================================
# ページ設定
# ==========================================================
st.set_page_config(
    page_title="HIDANE Classroom Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# カラーパレット（ヒダネブランド準拠）
BRAND_PRIMARY = "#C41E24"
BRAND_SECONDARY = "#F28C28"
BRAND_ACCENT = "#1a5276"
BRAND_DARK = "#0f1629"


# ==========================================================
# Supabase 接続チェック
# ==========================================================
def _get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return None
    try:
        from supabase import create_client
    except ImportError:
        return None
    return create_client(url, key)


def _db_available() -> bool:
    return _get_supabase_client() is not None


# ==========================================================
# データ取得（キャッシュ付き）
# ==========================================================
@st.cache_data(ttl=60)
def load_all_lessons() -> pd.DataFrame:
    client = _get_supabase_client()
    if client is None:
        return pd.DataFrame()
    try:
        result = (
            client.table("lessons")
            .select("*, teachers(name, classroom_id), classrooms(name)")
            .order("lesson_date", desc=True)
            .limit(5000)
            .execute()
        )
        rows = result.data or []
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["lesson_date"] = pd.to_datetime(df["lesson_date"])
        df["teacher_name"] = df["teachers"].apply(
            lambda x: x["name"] if isinstance(x, dict) else None
        )
        df["classroom_name"] = df["classrooms"].apply(
            lambda x: x["name"] if isinstance(x, dict) else None
        )
        return df
    except Exception as exc:  # noqa: BLE001
        st.error(f"DB取得エラー: {exc}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_checklist_scores(lesson_ids: list[str]) -> pd.DataFrame:
    if not lesson_ids:
        return pd.DataFrame()
    client = _get_supabase_client()
    if client is None:
        return pd.DataFrame()
    try:
        result = (
            client.table("checklist_scores")
            .select("*")
            .in_("lesson_id", lesson_ids)
            .execute()
        )
        return pd.DataFrame(result.data or [])
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_alerts(active_only: bool = True) -> pd.DataFrame:
    client = _get_supabase_client()
    if client is None:
        return pd.DataFrame()
    try:
        query = client.table("alerts").select("*, teachers(name)").order("triggered_at", desc=True)
        if active_only:
            query = query.is_("acknowledged_at", "null")
        result = query.limit(200).execute()
        rows = result.data or []
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["teacher_name"] = df["teachers"].apply(
            lambda x: x["name"] if isinstance(x, dict) else None
        )
        return df
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


# ==========================================================
# 可視化ヘルパー
# ==========================================================
def render_brand_header(subtitle: str) -> None:
    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, {BRAND_DARK} 0%, {BRAND_ACCENT} 100%);
                    padding: 24px 32px; border-radius: 12px; color: white; margin-bottom: 24px;'>
          <div style='font-size: 12px; letter-spacing: 0.2em; color: {BRAND_SECONDARY};'>
            HIDANE CLASSROOM INTELLIGENCE
          </div>
          <h1 style='margin: 4px 0 0 0; font-size: 28px; font-weight: 700;'>{subtitle}</h1>
          <div style='font-size: 13px; color: #cbd5e1; margin-top: 8px;'>
            株式会社開拓塾 × 株式会社ヒダネ
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, delta: str | None = None, color: str = BRAND_PRIMARY):
    delta_html = (
        f"<div style='font-size:12px; color:#64748b; margin-top:4px;'>{delta}</div>"
        if delta
        else ""
    )
    st.markdown(
        f"""
        <div style='background: white; border-radius: 12px; padding: 20px;
                    border-top: 3px solid {color};
                    box-shadow: 0 1px 3px rgba(0,0,0,0.05);'>
          <div style='font-size: 12px; color: #64748b;'>{label}</div>
          <div style='font-size: 32px; font-weight: 700; color: #111827; margin-top: 4px;'>{value}</div>
          {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_no_data_notice():
    st.info(
        "📭 まだデータがありません。`python run.py 動画.mkv --teacher-id ... --classroom-id ...` "
        "で授業を解析・登録してください。"
    )


def render_no_db_notice():
    st.warning(
        "⚠️ Supabase 未接続。`.env` に `SUPABASE_URL` と `SUPABASE_KEY` を設定してください。\n"
        "Phase 1の単発レポート（`output/*.html`）はDBなしでも動作します。"
    )


# ==========================================================
# ビュー: 社長
# ==========================================================
def view_ceo():
    render_brand_header("社長ビュー — 全社俯瞰")

    if not _db_available():
        render_no_db_notice()
        return

    df = load_all_lessons()
    if df.empty:
        render_no_data_notice()
        return

    # 週次集計
    df["week_start"] = df["lesson_date"].dt.to_period("W").dt.start_time
    this_week = df["week_start"].max()
    last_week = this_week - pd.Timedelta(days=7)

    this_df = df[df["week_start"] == this_week]
    last_df = df[df["week_start"] == last_week]

    this_avg = this_df["overall_score"].mean() if not this_df.empty else 0
    last_avg = last_df["overall_score"].mean() if not last_df.empty else 0
    delta = this_avg - last_avg

    cols = st.columns(4)
    with cols[0]:
        kpi_card("全社平均スコア", f"{this_avg:.1f}", f"先週比 {delta:+.1f}")
    with cols[1]:
        req_1on1 = this_df[this_df["overall_score"] < 70]["teacher_id"].nunique()
        kpi_card("要1on1講師数", f"{req_1on1}", "スコア70未満", BRAND_SECONDARY)
    with cols[2]:
        active_rooms = this_df["classroom_id"].nunique()
        kpi_card("稼働教室数", f"{active_rooms}", "今週授業あり", BRAND_ACCENT)
    with cols[3]:
        lesson_count = len(this_df)
        kpi_card("今週の授業本数", f"{lesson_count}", "", "#64748b")

    st.markdown("---")

    # 教室×週 ヒートマップ
    st.subheader("📊 教室×週次スコアヒートマップ")
    heat_df = (
        df.groupby(["classroom_name", "week_start"])["overall_score"]
        .mean()
        .reset_index()
    )
    if not heat_df.empty:
        pivot = heat_df.pivot(
            index="classroom_name", columns="week_start", values="overall_score"
        )
        fig = px.imshow(
            pivot,
            labels=dict(x="週", y="教室", color="平均スコア"),
            color_continuous_scale="RdYlGn",
            aspect="auto",
        )
        fig.update_layout(height=400, coloraxis_colorbar=dict(title="点"))
        st.plotly_chart(fig, use_container_width=True)

    # トップ/ワースト
    col_top, col_worst = st.columns(2)
    with col_top:
        st.subheader("🏆 今週のトップ5")
        top_df = (
            this_df.groupby("teacher_name")["overall_score"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
        )
        top_df.columns = ["講師", "平均スコア"]
        top_df["平均スコア"] = top_df["平均スコア"].round(1)
        st.dataframe(top_df, use_container_width=True, hide_index=True)

    with col_worst:
        st.subheader("⚠️ 今週の要介入5名")
        worst_df = (
            this_df.groupby("teacher_name")["overall_score"]
            .mean()
            .sort_values()
            .head(5)
            .reset_index()
        )
        worst_df.columns = ["講師", "平均スコア"]
        worst_df["平均スコア"] = worst_df["平均スコア"].round(1)
        st.dataframe(worst_df, use_container_width=True, hide_index=True)

    # アラート
    st.markdown("---")
    st.subheader("🚨 未対応アラート")
    alerts_df = load_alerts(active_only=True)
    if alerts_df.empty:
        st.success("現在、未対応アラートはありません。")
    else:
        display_cols = ["triggered_at", "severity", "teacher_name", "kind", "message"]
        display_cols = [c for c in display_cols if c in alerts_df.columns]
        st.dataframe(alerts_df[display_cols], use_container_width=True, hide_index=True)


# ==========================================================
# ビュー: 教室長
# ==========================================================
def view_manager():
    render_brand_header("教室長ビュー — 自教室の講師管理")

    if not _db_available():
        render_no_db_notice()
        return

    classrooms = fetch_all_classrooms()
    if not classrooms:
        render_no_data_notice()
        return

    classroom_map = {c["name"]: c["id"] for c in classrooms}
    selected_name = st.sidebar.selectbox("教室を選択", list(classroom_map.keys()))
    selected_id = classroom_map[selected_name]

    df = load_all_lessons()
    if df.empty:
        render_no_data_notice()
        return

    room_df = df[df["classroom_id"] == selected_id]
    if room_df.empty:
        st.info(f"「{selected_name}」の授業データがまだありません。")
        return

    # 自教室の講師一覧（直近4週）
    recent_start = pd.Timestamp(date.today() - timedelta(weeks=4))
    recent = room_df[room_df["lesson_date"] >= recent_start]

    summary = (
        recent.groupby("teacher_name")
        .agg(
            平均スコア=("overall_score", "mean"),
            授業本数=("id", "count"),
            最終授業日=("lesson_date", "max"),
        )
        .round(1)
        .sort_values("平均スコア", ascending=False)
        .reset_index()
    )
    summary["要1on1"] = summary["平均スコア"].apply(lambda s: "⚠️" if s < 70 else "")

    st.subheader(f"📋 {selected_name} 講師一覧（直近4週）")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # 講師選択→時系列
    teachers = summary["teacher_name"].tolist()
    if teachers:
        selected_teacher = st.selectbox("講師を選んで詳細を見る", teachers)
        teacher_df = room_df[room_df["teacher_name"] == selected_teacher].sort_values("lesson_date")
        if teacher_df.empty:
            return

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(f"📈 {selected_teacher} — スコア時系列")
            fig = px.line(
                teacher_df,
                x="lesson_date", y="overall_score",
                markers=True,
                labels={"lesson_date": "日付", "overall_score": "スコア"},
            )
            fig.update_traces(line_color=BRAND_PRIMARY)
            fig.update_layout(yaxis_range=[0, 100], height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🎯 直近の授業")
            latest = teacher_df.iloc[-1]
            score = latest.get("overall_score")
            grade = latest.get("grade_letter") or "—"
            st.metric("最新スコア", f"{score}点" if score is not None else "—",
                     f"グレード: {grade}")
            try:
                date_str = latest["lesson_date"].strftime("%Y-%m-%d")
            except Exception:  # noqa: BLE001
                date_str = str(latest.get("lesson_date", "—"))
            st.caption(f"{date_str} / {latest.get('subject') or '—'}")
            if latest.get("ai_commentary"):
                st.info(latest["ai_commentary"])

        # 12項目レーダー
        recent_lesson_ids = teacher_df["id"].tolist()[-5:]
        cs_df = load_checklist_scores(recent_lesson_ids)
        if not cs_df.empty:
            avg_by_item = (
                cs_df.groupby(["item_id", "item_title"])["score"]
                .mean()
                .reset_index()
                .sort_values("item_id")
            )
            st.subheader("🎯 12項目レーダー（直近5授業平均）")
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=avg_by_item["score"],
                theta=avg_by_item["item_title"],
                fill="toself",
                line_color=BRAND_PRIMARY,
                name=selected_teacher,
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

        # ビフォーアフター比較
        st.markdown("---")
        st.subheader("📊 ビフォーアフター比較（指導前後の成長を定量化）")
        if len(teacher_df) < 2:
            st.caption("授業が2本以上登録されると比較できます。")
        else:
            sorted_df = teacher_df.sort_values("lesson_date").reset_index(drop=True)
            # lesson_id を裏に持ち、表示はラベル（同日複数本でも一意に識別）
            option_map = {}
            for _, row in sorted_df.iterrows():
                date_str = row["lesson_date"].strftime("%Y-%m-%d")
                subj = row.get("subject") or ""
                label = f"{date_str} / {subj} / {row.get('overall_score', '—')}点 (id:{str(row['id'])[:8]})"
                option_map[label] = row["id"]
            option_labels = list(option_map.keys())
            col_bf, col_af = st.columns(2)
            with col_bf:
                before_label = st.selectbox("指導前の授業", option_labels, index=0)
            with col_af:
                after_label = st.selectbox("指導後の授業", option_labels, index=len(option_labels) - 1)
            if st.button("比較する", type="primary"):
                if before_label == after_label:
                    st.warning("異なる2授業を選んでください")
                else:
                    before_id = option_map[before_label]
                    after_id = option_map[after_label]
                    before_row = sorted_df[sorted_df["id"] == before_id].iloc[0]
                    after_row = sorted_df[sorted_df["id"] == after_id].iloc[0]
                    before_cs = load_checklist_scores([before_row["id"]])
                    after_cs = load_checklist_scores([after_row["id"]])
                    with st.spinner("AIが成長を分析中…"):
                        result = compare(
                            teacher_name=selected_teacher,
                            before_lesson=before_row.to_dict(),
                            after_lesson=after_row.to_dict(),
                            before_checklist=before_cs.to_dict("records") if not before_cs.empty else [],
                            after_checklist=after_cs.to_dict("records") if not after_cs.empty else [],
                        )
                    if result.error:
                        st.error(f"比較失敗: {result.error}")
                    else:
                        cols = st.columns(3)
                        with cols[0]:
                            st.metric("成長度スコア", f"{result.growth_score}/100")
                        with cols[1]:
                            st.metric("最も伸びた項目", result.biggest_improvement)
                        with cols[2]:
                            st.metric("下がった項目", result.biggest_drop or "なし")
                        st.success(f"🎉 {result.celebrate}")
                        st.markdown(f"**まとめ:** {result.summary}")
                        st.markdown(f"**次に伸ばすべき:** {result.action_for_next}")


# ==========================================================
# ビュー: 講師
# ==========================================================
def view_teacher():
    render_brand_header("講師ビュー — 自分の成長記録")

    if not _db_available():
        render_no_db_notice()
        return

    teachers = fetch_all_teachers()
    if not teachers:
        render_no_data_notice()
        return

    teacher_map = {t["name"]: t["id"] for t in teachers}
    selected_name = st.sidebar.selectbox("あなたの名前", list(teacher_map.keys()))
    selected_id = teacher_map[selected_name]

    history = fetch_teacher_history(selected_id, weeks=12)
    if not history:
        st.info(f"{selected_name} さんの授業データがまだありません。")
        return

    hist_df = pd.DataFrame(history)
    hist_df["lesson_date"] = pd.to_datetime(hist_df["lesson_date"])
    hist_df = hist_df.sort_values("lesson_date")

    latest = hist_df.iloc[-1]
    prev = hist_df.iloc[-2] if len(hist_df) > 1 else None

    cols = st.columns(3)
    with cols[0]:
        delta = (latest["overall_score"] - prev["overall_score"]) if prev is not None else 0
        kpi_card("最新スコア", f"{latest['overall_score']}", f"前回比 {delta:+}点", BRAND_PRIMARY)
    with cols[1]:
        kpi_card("グレード", latest.get("grade_letter") or "—", "", BRAND_SECONDARY)
    with cols[2]:
        kpi_card("今月の授業本数", f"{len(hist_df[hist_df['lesson_date'] >= pd.Timestamp.now() - pd.Timedelta(days=30)])}", "", BRAND_ACCENT)

    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📈 あなたのスコア推移")
        fig = px.line(
            hist_df,
            x="lesson_date", y="overall_score",
            markers=True,
            labels={"lesson_date": "日付", "overall_score": "スコア"},
        )
        fig.update_traces(line_color=BRAND_PRIMARY)
        fig.update_layout(yaxis_range=[0, 100], height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("✅ 直近の良かった点")
        good = latest.get("good_points") or []
        for pt in (good if isinstance(good, list) else []):
            st.markdown(f"- {pt}")

        st.subheader("📈 改善ポイント")
        imp = latest.get("improvements") or []
        for pt in (imp if isinstance(imp, list) else []):
            st.markdown(f"- {pt}")

    # 12項目レーダー
    cs_df = load_checklist_scores(hist_df["id"].tolist()[-5:])
    if not cs_df.empty:
        st.markdown("---")
        st.subheader("🎯 12項目（直近5授業平均）")
        avg = (
            cs_df.groupby(["item_id", "item_title"])["score"]
            .mean()
            .reset_index()
            .sort_values("item_id")
        )
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=avg["score"],
            theta=avg["item_title"],
            fill="toself",
            line_color=BRAND_PRIMARY,
            name=selected_name,
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    # AIコーチ対話
    st.markdown("---")
    st.subheader("💬 AIコーチに相談する")
    q = st.text_area(
        "聞きたいこと",
        placeholder="例: 今回3点だった「生徒との向き合い」を4点に上げるには？",
        key="coach_question",
    )
    if st.button("相談する", type="primary"):
        if not q.strip():
            st.warning("質問を入力してください")
        else:
            with st.spinner("AIコーチが考えています…"):
                checklist_avg_map: dict[int, dict] = {}
                if not cs_df.empty:
                    agg = (
                        cs_df.groupby(["item_id", "item_title"])["score"]
                        .mean()
                        .reset_index()
                    )
                    for _, row in agg.iterrows():
                        checklist_avg_map[int(row["item_id"])] = {
                            "title": row["item_title"],
                            "avg_score": float(row["score"]),
                        }
                resp = ask_coach(
                    teacher_name=selected_name,
                    question=q,
                    history=history,
                    checklist_avg=checklist_avg_map,
                )
                if resp.error:
                    st.error(f"AIコーチが応答できませんでした: {resp.error}")
                else:
                    st.markdown(resp.answer)


# ==========================================================
# ビュー: 管理（マスタ登録）
# ==========================================================
def view_admin():
    render_brand_header("管理 — 講師・教室マスタ")

    if not _db_available():
        render_no_db_notice()
        return

    tab1, tab2, tab3 = st.tabs(["教室マスタ", "講師マスタ", "授業履歴"])

    with tab1:
        st.subheader("🏫 教室一覧")
        rooms = fetch_all_classrooms()
        if rooms:
            st.dataframe(pd.DataFrame(rooms), use_container_width=True, hide_index=True)
        else:
            st.info("教室が未登録です。`admin/register_classroom.py` から登録してください。")

    with tab2:
        st.subheader("👤 講師一覧")
        teachers = fetch_all_teachers()
        if teachers:
            st.dataframe(pd.DataFrame(teachers), use_container_width=True, hide_index=True)
        else:
            st.info("講師が未登録です。`admin/register_teacher.py` から登録してください。")

    with tab3:
        st.subheader("📚 直近の授業履歴")
        df = load_all_lessons()
        if df.empty:
            render_no_data_notice()
        else:
            cols = ["lesson_date", "teacher_name", "classroom_name", "subject", "overall_score", "grade_letter"]
            cols = [c for c in cols if c in df.columns]
            st.dataframe(df[cols].head(50), use_container_width=True, hide_index=True)


# ==========================================================
# ビュー: 授業詳細（動画プレーヤー＋イベント連動）
# ==========================================================
KIND_ICONS_V = {
    "phone_use": "📱", "sleeping": "😴", "head_down": "🙇",
    "excessive_motion": "💥", "loud_noise": "🔊", "long_silence": "🤫",
    "eating": "🍫", "other_item": "❓",
}
KIND_LABELS_V = {
    "phone_use": "スマホ使用", "sleeping": "居眠り", "head_down": "うつむき継続",
    "excessive_motion": "ふざけ合い/立ち歩き", "loud_noise": "私語・騒ぎ",
    "long_silence": "長い沈黙", "eating": "飲食", "other_item": "その他",
}
SEV_COLORS_V = {"low": "#64748b", "medium": "#F28C28", "high": "#C41E24"}


def view_lesson_detail():
    render_brand_header("授業詳細 — 動画プレーヤー＋AI判定")

    if not _db_available():
        render_no_db_notice()
        return

    df = load_all_lessons()
    if df.empty:
        render_no_data_notice()
        return

    # 授業選択肢
    df = df.sort_values("lesson_date", ascending=False)
    labels = {}
    for _, row in df.iterrows():
        date_str = row["lesson_date"].strftime("%Y-%m-%d")
        label = (
            f"{date_str} / {row.get('classroom_name', '—')} / "
            f"{row.get('teacher_name', '—')} / "
            f"{row.get('subject') or '科目未設定'} / "
            f"{row.get('overall_score', '—')}点"
        )
        labels[label] = row["id"]

    selected = st.sidebar.selectbox("授業を選択", list(labels.keys()))
    if not selected:
        return
    lesson_id = labels[selected]

    lesson = fetch_lesson_detail(lesson_id)
    if not lesson:
        st.error("授業データが取得できませんでした")
        return

    # session state でシーク制御
    if "seek_sec" not in st.session_state:
        st.session_state.seek_sec = 0

    # ヘッダーメタ
    cols = st.columns(4)
    with cols[0]:
        kpi_card("総合スコア",
                 f"{lesson.get('overall_score', '—')}点" if lesson.get('overall_score') is not None else "—",
                 f"グレード {lesson.get('grade_letter') or '—'}",
                 BRAND_PRIMARY)
    with cols[1]:
        kpi_card("講師",
                 (lesson.get("teachers") or {}).get("name", "—"),
                 lesson.get("subject") or "—",
                 BRAND_ACCENT)
    with cols[2]:
        kpi_card("教室",
                 (lesson.get("classrooms") or {}).get("name", "—"),
                 f"{lesson.get('grade') or '—'} / {lesson.get('student_count') or '—'}名",
                 BRAND_SECONDARY)
    with cols[3]:
        dur = lesson.get("video_duration_sec") or 0
        kpi_card("動画尺", f"{dur//60}分{dur%60}秒", lesson.get("lesson_date", "")[:10], "#64748b")

    st.markdown("---")

    video_url = lesson.get("video_url")
    events = fetch_events_for_lesson(lesson_id)

    col_v, col_e = st.columns([3, 2])

    with col_v:
        st.subheader("🎥 授業動画")
        if video_url:
            # start_time 指定で該当イベント位置から再生
            st.video(video_url, start_time=int(st.session_state.seek_sec))
            st.caption(
                f"現在の再生開始位置: {int(st.session_state.seek_sec)//60}:"
                f"{int(st.session_state.seek_sec)%60:02d}"
            )
        else:
            st.info(
                "📹 動画は未アップロードです。"
                "`python scripts/upload_lesson_media.py` で Supabase Storage にアップロード後、"
                "ここに表示されます。"
            )

    with col_e:
        st.subheader("🚨 AI検知イベント")
        if not events:
            st.success("✅ 問題行動の検知は0件（Vision判定で全て除外）")
        else:
            for ev in events:
                mmss = f"{int(ev['start_sec']) // 60:02d}:{int(ev['start_sec']) % 60:02d}"
                icon = KIND_ICONS_V.get(ev.get("kind"), "•")
                label = KIND_LABELS_V.get(ev.get("kind"), ev.get("kind", "—"))
                color = SEV_COLORS_V.get(ev.get("severity"), "#64748b")
                with st.container():
                    c1, c2, c3 = st.columns([1, 3, 1])
                    with c1:
                        st.markdown(f"**{mmss}**")
                    with c2:
                        st.markdown(f"{icon} **{label}**")
                        if ev.get("vision_explanation"):
                            st.caption(ev["vision_explanation"][:100])
                        elif ev.get("description"):
                            st.caption(ev["description"][:100])
                    with c3:
                        if st.button("▶ 再生", key=f"seek_{ev['id']}"):
                            st.session_state.seek_sec = int(ev["start_sec"])
                            st.rerun()
                    st.markdown(
                        f"<div style='height:2px;background:{color};margin-bottom:12px;'></div>",
                        unsafe_allow_html=True,
                    )

    # AI講評
    st.markdown("---")
    st.subheader("🤖 AI講評")
    if lesson.get("ai_commentary"):
        st.info(lesson["ai_commentary"])

    col_good, col_imp = st.columns(2)
    with col_good:
        st.markdown("### ✅ 良かった点")
        for p in lesson.get("good_points") or []:
            st.markdown(f"- {p}")
    with col_imp:
        st.markdown("### 📈 改善ポイント")
        for p in lesson.get("improvements") or []:
            st.markdown(f"- {p}")

    # 12項目チェックシート
    cs_df = load_checklist_scores([lesson_id])
    if not cs_df.empty:
        st.markdown("---")
        st.subheader("🎯 12項目チェックシート")
        cs_df = cs_df.sort_values("item_id")
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=cs_df["score"], theta=cs_df["item_title"],
            fill="toself", line_color=BRAND_PRIMARY,
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("各項目の詳細コメント"):
            for _, r in cs_df.iterrows():
                st.markdown(
                    f"**{r['item_id']}. {r['item_title']}**: "
                    f"{'★' * int(r['score'])}{'☆' * (5 - int(r['score']))} — {r.get('ai_comment', '')}"
                )
                if r.get("evidence"):
                    st.caption(f"根拠: {r['evidence']}")


# ==========================================================
# ビュー: 動画投入（クライアントセルフサービス）
# ==========================================================
def view_upload():
    render_brand_header("📤 動画投入 — 授業録画のアップロード")

    if not _db_available():
        render_no_db_notice()
        return

    st.info(
        "授業録画（mkv / mp4 ・ 500MB以下）をアップロードすると、"
        "ヒダネ側の解析ワーカーが自動で拾って AI 採点します（通常5〜15分）。"
    )

    classrooms = fetch_all_classrooms()
    teachers = fetch_all_teachers()
    if not classrooms or not teachers:
        st.warning("教室または講師マスタが未登録です。先に「⚙️ 管理」タブで確認してください。")
        return

    room_map = {c["name"]: c["id"] for c in classrooms}
    teacher_map = {t["name"]: t["id"] for t in teachers}

    with st.form("upload_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            classroom_name = st.selectbox("教室", list(room_map.keys()))
            teacher_name = st.selectbox("講師", list(teacher_map.keys()))
            lesson_date = st.date_input("授業日", value=date.today())
        with col2:
            subject = st.text_input("科目", placeholder="例: 数学A / 英語")
            grade = st.text_input("学年", placeholder="例: 中1 / 中3")
            student_count = st.number_input("出席生徒数", min_value=0, max_value=50, value=20)
        notes = st.text_area("メモ（教室長のコメント等）", placeholder="任意")

        uploaded_file = st.file_uploader(
            "授業動画（mkv / mp4 ・最大500MB）",
            type=["mkv", "mp4", "mov", "avi"],
            accept_multiple_files=False,
        )

        submitted = st.form_submit_button("🚀 解析キューに投入", type="primary")

    if submitted:
        if not uploaded_file:
            st.warning("動画ファイルを選択してください")
            return
        size_mb = uploaded_file.size / (1024 * 1024)
        if size_mb > 500:
            st.error(f"ファイルが大きすぎます（{size_mb:.1f}MB > 500MB）。圧縮してから再試行してください。")
            return

        with st.spinner(f"Supabase Storage にアップロード中（{size_mb:.1f}MB）…"):
            result = upload_lesson_video(
                file_bytes=uploaded_file.getvalue(),
                teacher_id=teacher_map[teacher_name],
                classroom_id=room_map[classroom_name],
                lesson_date=lesson_date.isoformat(),
                subject=subject or None,
                grade=grade or None,
                student_count=int(student_count) if student_count else None,
                notes=notes or None,
                original_filename=uploaded_file.name,
            )

        if result.get("error"):
            st.error(f"アップロード失敗: {result['error']}")
        else:
            st.success(
                f"✅ アップロード完了！lesson_id: `{result['lesson_id']}`\n\n"
                f"ステータス: **解析待ち (pending)**\n\n"
                "ヒダネ解析ワーカーが数分以内に拾って採点を開始します。"
                "完了後「🎥 授業詳細」ビューでスコアとレポートが表示されます。"
            )
            st.video(result["storage_url"], start_time=0)

    # 解析待ち一覧
    st.markdown("---")
    st.subheader("⏳ 解析待ち授業一覧")
    pending = fetch_pending_lessons()
    if not pending:
        st.caption("解析待ちの授業はありません。")
    else:
        rows = []
        for p in pending:
            rows.append({
                "投入日時": p.get("created_at", "—")[:19].replace("T", " "),
                "講師": (p.get("teachers") or {}).get("name", "—"),
                "教室": (p.get("classrooms") or {}).get("name", "—"),
                "授業日": p.get("lesson_date", "—"),
                "ファイル": p.get("video_filename", "—"),
                "ステータス": p.get("status", "—"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ==========================================================
# メイン
# ==========================================================
VIEWS = {
    "🏢 社長ビュー": view_ceo,
    "👥 教室長ビュー": view_manager,
    "👤 講師ビュー": view_teacher,
    "🎥 授業詳細": view_lesson_detail,
    "📤 動画投入": view_upload,
    "⚙️ 管理": view_admin,
}


def main():
    with st.sidebar:
        st.markdown(
            f"<div style='color: {BRAND_SECONDARY}; font-size: 11px; letter-spacing: 0.2em;'>HIDANE CI</div>",
            unsafe_allow_html=True,
        )
        st.markdown("## 🎓 授業品質管理")
        view_name = st.radio("ビューを選択", list(VIEWS.keys()))
        st.markdown("---")
        st.caption(f"更新時刻 {datetime.now().strftime('%H:%M:%S')}")
        if not _db_available():
            st.error("DB未接続")
        else:
            st.success("DB接続中")

    VIEWS[view_name]()


if __name__ == "__main__":
    main()
