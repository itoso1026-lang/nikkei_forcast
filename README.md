# 日経平均 当日終値予測（決定木ベース）

毎営業日 **JST 8:00 時点で確定済みの情報だけ**を使って、当日の日経平均（^N225）終値を予測する。
課題（学習・検証）用。**実際の売買には使わない。** 仕様は [開発仕様.md](開発仕様.md)。

初心者向けの手順の解説（図・表つき）：[docs/walkthrough.html](docs/walkthrough.html)（ダウンロードしてブラウザで開く）

## セットアップ

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
$env:PYTHONIOENCODING = "utf-8"   # Windows のコンソールで日本語を出すため
```

## 実行手順

すべてプロジェクトのルートで `.venv\Scripts\python -m ...` として実行する。

| 順 | コマンド | 内容 | 出力 |
|---|---|---|---|
| 1 | `src.check_bar_boundaries` | yfinance の日足の区切り時刻を 1時間足と突き合わせる | `reports/bar_boundaries.csv` |
| 2 | `src.fetch_data [--full]` | 全ティッカーを取得（2回目以降は差分） | `data/raw/*.parquet` |
| 3 | `src.data_quality` | 欠損・異常値・限月切り替えのレポート | `reports/data_quality*.csv`, `roll_analysis.csv` |
| 4 | `src.features` | 特徴量テーブル | `data/processed/features.parquet` |
| 5 | `src.tune` | LightGBM の Optuna チューニング（50 trials、四半期ごとに再学習、直近1年は除外） | `reports/lgb_best_params.json` |
| 6 | `src.train` | walk-forward（全設定、毎月再学習） | `data/processed/oof.parquet` |
| 7 | `src.evaluate` | 評価表・図・SHAP | `reports/eval_*.csv`, `fig_*.png` |
| 8 | `src.train --final` | 本番用モデルを全データで学習 | `models/production.*` |
| 9 | `src.predict_today` | 当日の予測（取得 → 予測 → ログ追記） | `predictions/log.csv`, `latest.json` |
| 10 | `src.evaluate_live [--fetch]` | ライブ予測と実績の照合 | `reports/live_eval.csv` |

テスト：`.venv\Scripts\python -m pytest -q`

毎朝の自動実行：`scripts/run_daily.ps1` をタスクスケジューラに登録する（JST 7:30 を想定。登録例はファイル内）。

## リーク防止の仕組み

- 取得した全ての日足に **確定時刻 `available_at_utc`** を付ける（ティッカーごとの引け時刻＋30分の安全マージン。`config.yaml` の `availability`）。
- 予測対象日 t に使えるのは `available_at_utc < JST t日 8:00` の行だけ。`align.asof_align` が t ごとにその条件を満たす最新の行を選び、`align.assert_no_leak` が全ての列を検査する。
- この仕組みから、次のルールが自動的に成り立つ。
  - 米国株・CME：暦日で t より前の直近の取引日（日本の祝日明けには、東証の前営業日より新しい米国の日付になる）
  - JPY=X：t-2 以前（日足が翌日 London 0:00 に確定するため）
  - CME の bar 日付 t（東証の t 日の取引時間を含む足）は使わない
- テスト `tests/test_leak.py::test_features_unchanged_when_future_data_removed` で、t の cutoff 以降に確定したデータを全て消しても、t の特徴量が変わらないことを実データで確かめている。

## bar 境界の確認結果（check_bar_boundaries.py、2026-10-09 実行）

1時間足（直近約730日。CME は約600日）を候補の区切り時刻で日足に集計し、日足の終値との相対誤差の中央値が最小になる区切りを探した。**730日より前の期間は同じ区切りだと仮定している。**

| 対象 | 日足に入る最後の 1時間足の終了時刻 | 設定した確定時刻（＋30分） | 誤差中央値 |
|---|---|---|---|
| ^N225・東京の個別株 | 16:00 JST（15:00〜16:00 の足） | 16:00 JST（2024-11-04 以前は 15:30） | 3〜9bp |
| 米国株指数・ADR | 16:30 ET | 17:15 ET | 1〜4bp |
| ^VIX | 17:00 ET | 17:15 ET | 5bp |
| ^TNX | 15:20 ET | 17:15 ET | 0bp |
| CME（NIY, NKD, 6J, CL, GC） | 13:00〜15:00 CT | 16:00 CT | 1〜29bp |
| 欧州（DAX, STOXX50E） | 18:00 CET | 22:30 CET | 5bp |
| JPY=X | **01:00 London（当日）** | 翌日 00:00 London | 2bp |

- **CME の bar 日付 D は取引日 D**（前日 17:00 CT 〜 D 16:00 CT）を表す。D の足は東証の D 日の取引時間を含む。
- CME の日足の終値は清算値のことがあり、1時間足の最後の値とずれる（6J=F 29bp、CL=F 17bp）。区切りの判定には影響しない。
- JPY=X の日足は London の日付が変わってすぐの値で区切られていた。仕様どおり t-2 以前だけを使うよう、確定時刻を翌日 00:00 London とした。
- 全ティッカーで、設定した確定時刻は日足に入る最後の 1時間足の終了時刻以降（`configured_ok = True`）。

## データ品質（data_quality.py、2010-01〜2026-10）

- **^N225**：欠損なし。東京の個別株（7203.T など）には、2017〜2018年の祝日に 21日分の行が混ざっていた → 東証カレンダーは ^N225 の取引日を正とし、個別株は ^N225 の営業日に絞って使う。
- **NIY=F**：欠損 12日（2011〜2018）。**約6割の日（2,590日）で OHLC が平らで出来高 0**。そのうち一部の日（約100日）は、終値が前日の東証終値に張り付いた値で、米国引けの値になっていない（例：2021-06-22 は NIY 27,985、前日の ^N225 28,011、NKD 28,725）。
  - 高値・安値のレンジ特徴量は NKD=F で作る。
  - `niy_flat`（平らな日）と `niy_nkd_diff`（NIY と NKD の差）を特徴量に入れている。
  - B1・T2 の基準は仕様どおり NIY だが、`config.yaml` の `futures.base` で `niy_clean`（平らで NKD と 50bp 以上離れた日は NKD に置き換え）や `nkd` に切り替えられる。
- **NKD=F**：欠損 5日、平らな日 0日。
- 異常値（|リターン| > 5σ または 8%）として検出された日は、全て実際の急変（2011-03、2013-04、2016-06、2020-03、2024-08、2025-04 など）だったので、値は変えずにフラグだけ立てる（`outlier_action: keep`）。
- **限月切り替え**：メジャーSQ 前後 ±5営業日のベーシスに、はっきりした段差は見えなかった（`reports/roll_analysis.csv`）。`roll_flag` はメジャーSQ の翌営業日とした。

## バックテスト結果（2026-10-09 実行）

全ての予測が揃う共通期間 **2016-02-01〜2026-10-08（2,611日）**、毎月再学習の walk-forward。主指標は bp-MAE（低いほど良い）。詳細は `reports/eval_*.csv`。

| 予測 | bp-MAE | 円MAE | B1との差 | B1比 改善率 | 方向的中率（B1比） |
|---|---|---|---|---|---|
| スタッキング（T2, l1, expanding, B） | **69.55** | 215 | −2.61bp | 3.6% | 56.8% |
| XGBoost（T2, l1, expanding, B） | 69.59 | 216 | −2.56bp | 3.5% | 57.2% |
| 単純平均（T2, l1, expanding, B） | 69.85 | 217 | −2.30bp | 3.2% | 57.1% |
| LightGBM（T2, huber, expanding, B） | 69.89 | 217 | −2.26bp | 3.1% | 55.8% |
| LightGBM（T2, l1, expanding, B）＝本番設定 | 69.96 | 217 | −2.19bp | 3.0% | 55.9% |
| B1_nkd（参考：NKD 前日終値） | 70.56 | 219 | −1.60bp | 2.2% | 53.3% |
| B2（Ridge） | 71.39 | 221 | −0.76bp | 1.1% | 53.3% |
| **B1（NIY 前日終値）** | 72.15 | 223 | 0 | — | — |
| B1'（20日平均ベーシス補正） | 74.48 | 231 | +2.33bp | −3.2% | 48.2% |
| B0（前日終値） | 94.28 | 293 | +22.13bp | −30.7% | 54.0% |

**holdout（2025-11-01〜、226日。チューニングに使っていない）**：スタッキング 99.24bp、B1 101.89bp（−2.66bp、2.6%）、B2 101.13bp、B1_nkd 102.23bp。

設定ごとの比較（各グループの全予測の平均 bp-MAE）：
- ウィンドウ：expanding 70.28 ＜ rolling 70.94
- 目的変数：T2 70.33 ＜ T1 70.90
- 目的関数：huber 70.48 ≒ l1 70.58
- early stopping：方式B（ラウンド数の中央値で固定）70.44 ＜ 方式A 70.79
- モデル：stack 70.11 ＜ avg 70.23 ＜ cat 70.62 ≒ xgb 70.71 ＜ lgb 70.99 ＜ ridge 71.44

### 期間別（スタッキング − B1、bp）

| 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|
| −1.8 | −1.8 | −0.6 | −0.0 | −1.8 | **−11.7** | −4.5 | −1.8 | −0.5 | −0.8 | −3.3 |

- 高ボラ期間：2020-03（48日）は LightGBM が B1 より −3.2bp。**2024-08（26日）は全モデルが B1 より悪い**（LightGBM +3.6bp、Ridge +7.5bp）。

### 解釈（正直な結論）

- GBDT は B1・B2 に毎年ほぼ勝つが、差は **2〜3%（約2bp、約8円）** と小さい。リーク警告の閾値（15%）からは遠い。
- 改善のかなりの部分は **2021年**（−11.7bp）に集中している。この年は NIY=F の終値が前日の東証終値に張り付く日が多く、モデルは `niy_nkd_diff` などでそれを補正している。**データの不具合を補正している面が大きく、市場の情報から当日の動きを予測できているわけではない。**
- NKD の終値をそのまま使うだけ（B1_nkd）で B1 比 2.2% 改善する。B1_nkd に対するスタッキングの改善は約1.4%（−1.0bp）。holdout では B1_nkd は B1 より悪い。
- SHAP 重要度（holdout、mean |SHAP|）の上位は `niy_gap` 6.9bp、`niy_nkd_diff` 3.8bp、`n225_ma20dev` 1.5bp、ADR プレミアム 1.2〜1.3bp。米国株のリターンは、すでに先物の終値に織り込まれているので上位に来ない。
- yfinance は過去の日足を後から書き換えることがある（2026-10-09 に、10/8 の NIY の終値が 68,495 → 68,420 に変わった）。朝の時点で取得した値と、バックテストに使う最終値は同じとは限らない。

## 図（reports/）

- `fig_by_year.png`：年別 bp-MAE（本番設定のモデル・B1・B2）
- `fig_timeseries_holdout.png`：holdout 期間の予測と実績
- `fig_residuals.png`：残差の分布
- `fig_shap_importance.png`、`fig_waterfall_20240805.png`、`fig_waterfall_20240806.png`：SHAP

## 予測 API

```powershell
.\scripts\run_api.ps1
# または
.venv\Scripts\python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Swagger UI：http://127.0.0.1:8000/docs

> **外部に公開しないこと。** データ源の yfinance は非公式の API で、利用規約上も再配布に向かない。初期設定ではローカル（127.0.0.1）でだけ待ち受ける。

| メソッド | パス | 内容 |
|---|---|---|
| GET | `/health` | モデルのバージョン・学習終了日、各データの最新日付と確定時刻、サーバー時刻（JST） |
| GET | `/predict/today` | latest.json。今日の分でなければ `stale: true`。休場日は 404 |
| GET | `/predict/{date}` | log.csv の該当日（live の 8:00 前の最後 → late → backfill の順）。なければ 404 |
| POST | `/predict/run` | 取得してから予測し、ログを更新。body `{"target_date": "YYYY-MM-DD" \| null}` |
| GET | `/predictions?from=&to=` | 期間内の予測に、実績終値と bp誤差を付けて返す |
| POST | `/model/reload` | `models/` のモデルを読み直す |

- `POST /predict/run`
  - 実行中なら 409（CLI の実行中も含む。`predictions/.run.lock` で判定）
  - 同じ日付の再実行は、`api.min_rerun_interval_min`（10分）以内なら前回の結果を返す（`cached: true`）
  - 過去日付は `mode=backfill`、当日分を 8:00 以降に実行したら `mode=late` として記録する。どちらも使うデータは JST 8:00 より前に確定したものだけで、ライブ評価からは除外する
  - 学習期間に含まれる日付の予測には、`warnings` に in-sample と入る
  - エラー：データ取得失敗 503、休場日 404、実行中 409、未来の日付 422
- `api.api_key_required: true` にすると、`X-API-Key` ヘッダーを環境変数 `NIKKEI_API_KEY` と照合する（`/health` は除外）

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/predict/today
curl http://127.0.0.1:8000/predict/2026-10-09
curl -X POST http://127.0.0.1:8000/predict/run -H "Content-Type: application/json" -d '{"target_date": null}'
curl -X POST http://127.0.0.1:8000/predict/run -H "Content-Type: application/json" -d '{"target_date": "2026-10-01"}'
curl "http://127.0.0.1:8000/predictions?from=2026-10-01&to=2026-10-31"
curl -X POST http://127.0.0.1:8000/model/reload
```

毎朝の予測はタスクスケジューラ（`predict_today.py`）で行い、API はその結果を返す。API の中にスケジューラは持たない。

コンテナ（任意）：`Containerfile` の先頭のコメントを参照。`data/`・`models/`・`predictions/` はボリュームでマウントする。
