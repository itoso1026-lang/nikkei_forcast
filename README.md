# 日経平均 当日終値予測（決定木ベース）

毎営業日 **JST 8:00 時点で確定済みの情報だけ**を使って、当日の日経平均（^N225）終値を予測する。
課題（学習・検証）用。**実際の売買には使わない。** 仕様は [開発仕様.md](開発仕様.md)。

初心者向けの手順の解説（図・表つき）：[docs/walkthrough.html](docs/walkthrough.html)（ダウンロードしてブラウザで開く）

## はじめての使い方（Windows）

クローンから最初の予測までの手順です。コマンドはすべて **PowerShell** に入力します（スタートメニューで「PowerShell」と検索して開く）。`#` から後ろはコメントなので、入力しなくてかまいません。

### 0. 必要なもの

| もの | 確認するコマンド | 入っていなければ |
|---|---|---|
| Git | `git --version` | `winget install Git.Git` を実行し、PowerShell を開き直す |
| Python 3.12 | `py -3.12 --version` | `winget install Python.Python.3.12` を実行し、PowerShell を開き直す |
| インターネット接続 | ― | データを yfinance（Yahoo Finance）から取得する |

- ディスクは、ライブラリとデータで約 2GB 使います。
- バージョン表示（例：`Python 3.12.10`）が出れば OK です。Python 3.13 などでは動作を確認していません。

### 1. リポジトリをクローンする

置きたいフォルダに移動してからクローンします。リポジトリ名の綴りは `nikkei_forcast` です。

```powershell
cd $HOME\Documents                     # 置き場所はどこでもよい
git clone https://github.com/itoso1026-lang/nikkei_forcast.git
cd nikkei_forcast
```

以降のコマンドは、すべてこの `nikkei_forcast` フォルダの中で実行します。

### 2. Python の仮想環境を作り、ライブラリを入れる

仮想環境（`.venv`）は、このプロジェクト専用の Python の置き場所です。パソコン全体の Python を汚さずにすみます。

```powershell
py -3.12 -m venv .venv                                  # 仮想環境を作る（数秒）
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt   # 5〜10分
```

### 3. 仮想環境を有効にする（PowerShell を開くたびに必要）

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass   # この画面だけスクリプトの実行を許可する
.\.venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING = "utf-8"                              # 日本語の出力が文字化けしないようにする
```

行の先頭に `(.venv)` と表示されれば有効になっています。以降の `python` は、この仮想環境の Python を指します。

> 有効にしない場合は、`python` の代わりに `.\.venv\Scripts\python` と入力しても同じです。

### 4. 動作確認（テスト）

```powershell
python -m pytest -q
```

最後に `passed` と出れば OK です。データをまだ取得していないので、実データを使うテスト3件は `skipped` になります（この段階では正常）。

### 5. 初期設定：モデルを作る（最初に1回だけ・約5分）

学習済みのモデルはリポジトリに入っていません。最初に1回、次のコマンドで自分のパソコンにモデルを作ります。

```powershell
python -m src.retrain
```

このコマンドは次の3つを続けて行います。

1. 2010年から今日までのデータを yfinance から取得する（初回は 2〜5分。2回目以降は新しい分だけ）
2. 特徴量の表を作る（数秒）
3. 本番モデルを全データで学習し、`models\` に保存する（約10秒）

成功すると、このような表示が出ます（2026-10-09 に実行した例。日付とハッシュは実行した日によって変わります）。

```text
1/3 データを取得しています…
2/3 特徴量を作っています…
3/3 本番モデルを学習しています…
  lgb: 187 rounds（config）
  xgb: 156 rounds（config）
  cat: 226 rounds（config）
  stack: lgb -0.063, xgb 0.175, cat 1.135, ridge 0.117, 切片 3.17bp（config）
保存しました: ...\models\production.pkl  version=stack-T2-20261008-<コミットのハッシュ>  学習期間 〜2026-10-08

新しいモデル: stack-T2-20261008-<コミットのハッシュ>（学習期間 〜2026-10-08）
```

本番モデルの初期設定は `config.yaml` の `production:` にあります。4つのモデルの予測を、決まった係数で足し合わせる「スタッキング」です。

| 設定 | 初期値 | 意味 |
|---|---|---|
| `model` | `stack` | スタッキングを使う（`lgb` にすると LightGBM 1つだけ） |
| `members` | `lgb, xgb, cat, ridge` | 組み合わせるモデル：LightGBM・XGBoost・CatBoost・Ridge |
| `rounds` | lgb 187、xgb 156、cat 226 | 各モデルの学習回数 |
| `stack_weights` | lgb −0.063、xgb 0.175、cat 1.135、ridge 0.117、切片 3.166 | 予測 ＝ Σ 係数 × 各モデルの予測 ＋ 切片（単位は bp） |

- `rounds` と `stack_weights` は、2016〜2026年の過去データでの検証（walk-forward）で求めた値です。表示の `（config）` は、この初期設定を使ったという意味です。
- このモデルの過去の成績（2016-02〜2026-10、2,611日）：誤差の平均 69.55bp（約215円）。先物の値（B1）だけで予測するより約3.6% 小さい誤差です（後半の「バックテスト結果」を参照）。

### 6. 予測する

予測したいときに、このコマンドを入力します。

```powershell
python -m src.predict_today
```

新しいデータを取得してから予測し、結果を `predictions\log.csv` に追記します。成功すると、このような表示が出ます。

```text
予測対象日      : 2026-10-09（mode=late）
予測終値        : 68,511.17 円（前日比 -1.15%）
前日終値（B0）  : 69,306.33 円
先物（B1）      : 68,420.00 円  予測との差 +91.17 円（+13.3bp）
先物データ確定  : 2026-10-09T06:30:00+09:00
モデル          : stack-T2-20261008-<コミットのハッシュ>
```

- **平日の 7:50〜8:00 に実行するのがおすすめです。** 各データの確定時刻（引け＋30分の安全マージン）は、夏時間なら 6:00〜6:45、冬時間なら 7:00〜7:45（米国株 7:45、CME 先物 7:30）です。それより前に実行すると、まだ確定していないデータは使われません。夏時間なら 7:00 でもかまいません。
- `mode` は、8:00 より前に実行すると `live`、8:00 以降だと `late` になります。`late` でも予測は正しく出ますが（8:00 までに確定したデータだけを使う）、毎朝の成績の集計（手順 8）からは外れます。
- `[warn]` の行が出たら、その内容を読んでください。CME のデータが未確定のときは、モデルを使わずに先物の値（B1）を出し、`モデル` の欄が `fallback-...` になります。
- 休場日（土日・祝日・年末年始）に実行すると「休場日です。何もせずに終了します。」と出て終わります。
- 過去の日付も予測できます：`python -m src.predict_today --date 2026-10-01`（`mode=backfill` として記録）

### 7. モデルを更新したい場合

データがたまって、新しいデータも含めて学習し直したくなったら、初期設定と同じコマンドを入力します。

```powershell
python -m src.retrain
```

- 新しい日の分のデータを取得し、学習期間を最新の日まで延ばして、モデルを作り直します（1分ほど）。
- 学習回数と係数は `config.yaml` の初期設定のままです。変わるのは「学習に使うデータの期間」だけです。
- 表示の最後に、前のモデルと新しいモデルのバージョン（学習期間の最終日を含む）が出ます。
- 前のモデルは上書きされます。残しておきたいときは、先に `models\` フォルダをコピーしてください。
- API を起動している場合は、そのあと `POST /model/reload` を呼ぶと新しいモデルに切り替わります（手順 9）。

モデルの構成を変えたい場合は、`config.yaml` の `production:` を書き換えてから `python -m src.retrain` を実行します。

- LightGBM 1つだけにする：`model: lgb`（学習回数は `rounds:` の `lgb` を使う）
- `members` を変えると、初期設定の係数が使えなくなり、単純平均（係数がすべて同じ）になります。

### 8. 予測の成績を確かめる

数日分の予測がたまったら、実績と照合します。

```powershell
python -m src.evaluate_live --fetch    # 最新の実績を取得してから照合 → reports\live_eval.csv
```

`mode=live`（8:00 より前に実行した予測）だけを集計し、モデル・B1・B0 の誤差を比べます。

### 9. API を使う（任意）

別の PowerShell の画面で API を起動します（手順 3 で仮想環境を有効にしてから）。

```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

`Uvicorn running on http://127.0.0.1:8000` と出たら、ブラウザで http://127.0.0.1:8000/docs を開くと、画面から各 API を試せます。止めるときは `Ctrl + C` を押します。コマンドから呼ぶ例は、下の「予測 API」の節にあります。

### 10. リポジトリが更新されたとき

```powershell
git pull
python -m pip install -r requirements.txt
python -m src.retrain        # 設定やコードが変わっている場合に備えて、モデルを作り直す
```

### つまずいたとき

| 症状 | 原因と対処 |
|---|---|
| `py` が見つからない | Python が入っていないか、PowerShell を開き直していない。手順 0 をやり直す |
| `Activate.ps1 を読み込むことができません`（スクリプトの実行が無効） | 手順 3 の `Set-ExecutionPolicy -Scope Process ...` を先に実行する |
| 日本語が文字化けする | `$env:PYTHONIOENCODING = "utf-8"` を実行してから、もう一度実行する |
| `FileNotFoundError: ... data\raw\... がありません` | データを取得していない。`python -m src.retrain` を実行する |
| `FileNotFoundError: ... models\production.pkl` | モデルを作っていない。`python -m src.retrain` を実行する |
| `... の取得に失敗しました`、`429`、`Too Many Requests` | Yahoo Finance のアクセス制限。数分〜数十分待ってから、もう一度実行する |
| `別の予測処理が実行中です` | 別の画面で予測・再学習・API の予測が動いている。終わってから実行する |
| `curl` で `Invoke-WebRequest` のエラーが出る | Windows PowerShell 5.1 では `curl` が別のコマンドの別名。`curl.exe` と入力する |
| 予測が `fallback-B1-provisional` になる | CME のデータがまだ確定していない時刻に実行した。7:50 以降に実行する |

<details>
<summary>macOS・Linux の場合</summary>

```bash
git clone https://github.com/itoso1026-lang/nikkei_forcast.git
cd nikkei_forcast
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.retrain          # 初期設定（最初に1回）
python -m src.predict_today    # 予測する
```

LightGBM が `libomp` を必要とするので、macOS では先に `brew install libomp` を実行してください。

</details>

## コマンド一覧

すべてリポジトリのフォルダで、仮想環境を有効にしてから実行する。

| コマンド | 内容 | 出力 |
|---|---|---|
| `python -m src.retrain [--no-fetch]` | データ取得 → 特徴量 → 本番モデルの学習（初期設定・モデルの更新） | `data/`, `models/production.*` |
| `python -m src.predict_today [--date YYYY-MM-DD] [--no-fetch]` | 予測（取得 → 予測 → ログ追記） | `predictions/log.csv`, `latest.json` |
| `python -m src.evaluate_live [--fetch]` | 予測と実績の照合 | `reports/live_eval.csv` |
| `python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 1` | 予測 API | ― |
| `python -m pytest -q` | テスト | ― |

`src/` には、ここで紹介していない検証用のスクリプト（`check_bar_boundaries`・`data_quality`・`tune`・`train`・`evaluate`）もあります。どんな検証をしたかは [docs/walkthrough.html](docs/walkthrough.html) と [開発仕様.md](開発仕様.md) にまとめています。

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
| スタッキング（T2, l1, expanding, B）＝本番設定 | **69.55** | 215 | −2.61bp | 3.6% | 56.8% |
| XGBoost（T2, l1, expanding, B） | 69.59 | 216 | −2.56bp | 3.5% | 57.2% |
| 単純平均（T2, l1, expanding, B） | 69.85 | 217 | −2.30bp | 3.2% | 57.1% |
| LightGBM（T2, huber, expanding, B） | 69.89 | 217 | −2.26bp | 3.1% | 55.8% |
| LightGBM（T2, l1, expanding, B） | 69.96 | 217 | −2.19bp | 3.0% | 55.9% |
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

PowerShell から呼ぶ例（`Invoke-RestMethod` は結果を表の形で表示する）：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/predict/today
Invoke-RestMethod http://127.0.0.1:8000/predict/2026-10-09
Invoke-RestMethod "http://127.0.0.1:8000/predictions?from=2026-10-01&to=2026-10-31"
# 今日の予測を実行する
Invoke-RestMethod -Method Post http://127.0.0.1:8000/predict/run -ContentType "application/json" -Body '{"target_date": null}'
# 過去の日付を予測する（mode=backfill として記録）
Invoke-RestMethod -Method Post http://127.0.0.1:8000/predict/run -ContentType "application/json" -Body '{"target_date": "2026-10-01"}'
Invoke-RestMethod -Method Post http://127.0.0.1:8000/model/reload
```

curl を使う例（Windows の PowerShell 5.1 では `curl` ではなく `curl.exe` と入力する。macOS・Linux では `curl`）：

```bash
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/predict/today
curl.exe -X POST http://127.0.0.1:8000/predict/run -H "Content-Type: application/json" -d "{}"
curl.exe -X POST http://127.0.0.1:8000/model/reload
```

（`-d "{}"` は target_date を省略した形で、今日の予測になる。PowerShell 5.1 から curl.exe に JSON の `"` を渡すと崩れるので、日付を指定するときは上の `Invoke-RestMethod` を使う）

予測は、必要なときに `python -m src.predict_today` か `POST /predict/run` で実行する。API の中にスケジューラは持たない。

コンテナ（任意）：`Containerfile` の先頭のコメントを参照。`data/`・`models/`・`predictions/` はボリュームでマウントする。
