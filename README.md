# HIDANE Classroom Intelligence — ダッシュボード公開版

株式会社開拓塾様向け「授業品質管理システム」のダッシュボード部分。
Streamlit Community Cloud 用の軽量公開リポジトリ。

## 本リポジトリの位置付け

フル機能のソース（動画解析パイプライン含む）は別リポジトリ（プライベート）で管理されています。
本リポジトリにはダッシュボード UI に必要な最小限のファイルのみ含まれます：

```
.
├── app.py              # Streamlit エントリポイント
├── db_client.py        # Supabase クライアント（軽量）
├── ai_coach.py         # AIコーチ対話
├── compare_lessons.py  # ビフォーアフター比較
├── requirements.txt    # 依存（streamlit/pandas/plotly/supabase/anthropic）
├── .streamlit/
│   └── config.toml     # ブランドカラー設定
└── README.md
```

## 機能

- **社長ビュー**: 全教室俯瞰 / ヒートマップ / トップ5・ワースト5 / アラート
- **教室長ビュー**: 自教室の講師一覧 / 時系列 / 12項目レーダー / BeforeAfter比較
- **講師ビュー**: 個人の成長記録 / 12項目レーダー / AIコーチ対話
- **管理**: マスタ一覧・授業履歴

## ローカル起動

```bash
pip install -r requirements.txt

cat > .env <<EOF
ANTHROPIC_API_KEY=...
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=sb_publishable_...
EOF

streamlit run app.py
```

## Streamlit Cloud デプロイ

1. https://share.streamlit.io/ → New app
2. Repository: `hidane2025/hidane-classroom-dashboard`
3. Branch: `main` / Main file path: `app.py`
4. Advanced settings → Secrets に以下を設定：
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   SUPABASE_URL = "https://xxxxx.supabase.co"
   SUPABASE_KEY = "sb_publishable_..."
   ```
5. Deploy

## 制作

株式会社ヒダネ × 株式会社開拓塾
