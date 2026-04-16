# 設計書：美的評価分布推定のためのMCD-based UDA

---

## 1. タスク概要

- **問題設定**：教師なしドメイン適応（UDA）
- **ドメイン**：art / fashion / scenery の3種類
- **タスク**：各画像に対する美的評価スコアの**分布推定**（例：評価者の評点分布をヒストグラムとして予測）
- **ドメイン適応の方向**：任意の1ドメインをソース，別の1ドメインをターゲットとする（計6方向）

---

## 2. ネットワーク構成

```
入力画像 x
    ↓
G : 特徴抽出器（例：ResNet-50など事前学習済みCNN）
    ↓
z = G(x)  ← ここでMCDを適用
    ↓
F1, F2 : 分布推定器（2つの独立したヘッド）
    ↓
p1(x), p2(x)  ← K-binのヒストグラム（softmax出力）
```

- `G`：ソース・ターゲット共有
- `F1`, `F2`：独立して初期化，出力はK次元のsoftmax（評価スコアの離散化された確率分布）
- `K`：評価スコアのbin数（例：10段階）

---

## 3. 損失関数

### 3.1 ソース分類損失（Step A・B共通）

ソースデータに対してEMDで学習する：

$$\mathcal{L}_s = \text{EMD}(F_i(G(x^s)),\ y^s)$$

- $y^s$：ソースの真の評価分布（ヒストグラム）
- EMDは離散分布間のWasserstein-1距離（累積和のL1距離で効率的に計算可能）：

$$\text{EMD}(p, q) = \sum_{k=1}^{K} \left| \sum_{j=1}^{k} p_j - \sum_{j=1}^{k} q_j \right|$$

### 3.2 Discrepancy損失

2つの推定分布間のEMDをDiscrepancyとして使用：

$$\mathcal{L}_{\text{adv}} = \mathbb{E}_{x^t}\left[\text{EMD}(F_1(G(x^t)),\ F_2(G(x^t)))\right]$$

> **備考**：美的評価スコアは順序構造（1点と2点は近い，1点と10点は遠い）を持つため，L1 discrepancyよりもEMDのほうがこの構造を自然に捉えられる．

---

## 4. 学習ステップ

元論文のStep A/B/Cに対応：

### Step A：ソース学習
$$\min_{G, F_1, F_2} \mathcal{L}_s$$
両分類器とジェネレータをソースデータで学習する．

### Step B：Discrepancy最大化（G固定）
$$\min_{F_1, F_2}\ \mathcal{L}_s - \lambda \cdot \mathcal{L}_{\text{adv}}$$
ターゲットに対して2つの推定器の予測を食い違わせる．

### Step C：Discrepancy最小化（F1, F2固定，n回繰り返し）
$$\min_{G}\ \mathcal{L}_{\text{adv}}$$
ターゲット特徴をソースのサポート内に引き込む．

---

## 5. 実装上の注意点・検討事項

- **EMDの微分可能性**：累積和のL1距離として実装すれば自動微分可能
- **ハイパーパラメータ**：
  - $\lambda$：ソース損失とDiscrepancy損失のバランス
  - $n$：Step Cのジェネレータ更新回数（論文では2〜4を検証）
  - $K$：スコアのbin数

---

## 6. 実装前の意思決定事項

本プロジェクト固有の構造（NIMAモデル・GIAA/PIAAデータパイプライン・既存DAメソッドのI/F）との整合を踏まえ，実装着手前に確定が必要な項目を以下に列挙する．

---

### 6.1 モデル構造：G / F1 / F2 の分割境界

**問題**：現在の`NIMA`は `backbone → feat_proj → fc_aesthetic` という単一ヘッド構造であり，MCD用の2ヘッド構造をどこで分割するかを決める必要がある．

| 案 | G の範囲 | F1 / F2 の範囲 | 備考 |
|---|---|---|---|
| **案A** | `backbone + feat_proj` | `fc_aesthetic` × 2（独立線形層） | 変更が最小限。F1/F2が浅いため表現力が低い |
| **案B** | `backbone` | `feat_proj + fc_aesthetic` × 2（独立MLP） | feat_projを含む分だけF1/F2が深い。Discrepancyが大きくなりやすい |

> **決定済み（2026-04-16）**：**案A を採用**。DANN・DJDOTがいずれも `feat_proj` 出力（`domain_feat`，256次元）をアライメント対象として扱っており，設計思想と一致する。既存の `return_feat=True` がそのまま使え，`NIMA` クラスへの変更も最小限に抑えられる。

---

### 6.2 モデルクラスの実装方針

**問題**：NIMAに2ヘッドを追加する際，既存クラスを改造するか新クラスを作るか．

| 案 | 内容 | 長所・短所 |
|---|---|---|
| **案A** | `NIMA`に`fc_aesthetic2`を追加し，`forward(head=1/2)`で切り替え | 既存コードへの変更が最小限。ただし単一モデルへの追加でinterface汚染の懸念 |
| **案B** | `NIMA_MCD`などMCD専用ラッパークラスを作り，内部で`G`・`F1`・`F2`を保持 | 疎結合で他メソッドへの影響ゼロ。DANN・DJDOTと同様の外部コンポーネント方式 |
| **案C** | `setup()`内で`fc_aesthetic2`を外部コンポーネントとして追加し，`model.fc_aesthetic`（F1）と`components['f2']`（F2）として分離 | 既存`NIMA`を無改造で使える。`trainer()`内での参照が煩雑になりうる |

> **決定済み（2026-04-16）**：**案B を採用**。`NIMA_MCD`（仮称）専用クラスを `src/methods/mcd.py` に定義し，`setup()`・`trainer()` と同一ファイルに置く。内部で `G`（`NIMA` の `backbone + feat_proj` 相当）・`F1`・`F2`（独立した `fc_aesthetic` 相当の線形層）を保持し，既存の `NIMA` クラスおよび DANN・DJDOT 実装に一切影響を与えない。

---

### 6.3 オプティマイザの分離構成

**問題**：Step A/B/CではGとF1,F2を別々に最適化する必要があり，現在のDANNは`optimizer`（モデル全体）と`optimizer_disc`（discriminator）の2本立てだが，MCDでは異なる分離が必要になる．

| コンポーネント | Step A | Step B | Step C |
|---|---|---|---|
| `optimizer_G`（G用） | 更新 | 固定 | 更新 |
| `optimizer_F`（F1+F2用） | 更新 | 更新 | 固定 |

- `optimizer_F`はF1とF2をまとめて1つにするか，別々にするか．
- AMPの`GradScaler`は1インスタンスを共有してよいか（`scaler.step(opt1); scaler.step(opt2)`の順序依存に注意）．

> **決定済み（2026-04-16）**：**G・F1+F2 の2本立てを採用**。MCD の Step 設計上 F1 と F2 は常に同じ Step で同じ方向に更新される（B では両方更新，C では両方固定）ため，独立したオプティマイザは不要。`optimizer_F = AdamW(list(f1.parameters()) + list(f2.parameters()), ...)` として1本にまとめる。`GradScaler` は1インスタンスを共有し，`scaler.step(optimizer_G)` → `scaler.step(optimizer_F)` の順で呼び出す。

---

### 6.4 データパイプライン

**問題**：現在，GIAAターゲットデータは`Image_GIAA_HistogramDataset`として存在するが，このDatasetは`Aesthetic`（ラベル）込みで作られている．MCDのStep B/Cではターゲットのラベルは使わない．

- ターゲット用DataLoaderとして，既存の`Image_GIAA_HistogramDataset`（ラベルを捨てて使う）をそのまま用いるか，画像のみを返す軽量Datasetを新設するか．
- ソースとターゲットのバッチサイズを同一にする必要があるか（`drop_last=True`で揃えるか）．
- 現状の`collate_fn`がターゲット側にも適用可能か（`Aesthetic`キーが必須かどうか）を確認する．

> **決定済み（2026-04-16）**：**DANN・DJDOTと同じ構造を採用**。`Image_GIAA_HistogramDataset` をそのまま `tgt_loader` として使い，`trainer()` 内で `sample_tgt['image']` のみ参照してラベルは無視する。`collate_fn` も共通のものをそのまま使用。`train_GIAA.py` 側は `_DA_METHOD_MODULES` に `'MCD': '.methods.mcd'` を追加するだけでよく，`_build_target_loaders_*()` の変更は不要。

---

### 6.5 GIAAタスクへの適用範囲

**問題**：MCD適用範囲をGIAAのみとするかPIAAにも拡張するかを決める必要がある．

- `train_PIAA.py` には `train_dann_piaa_pretrain()` / `train_dann_piaa_finetune()` および DJDOT 相当の関数が実装されており，**DANN・DJDOTはどちらもGIAA・PIAA両フェーズに適用されている**。
- MCDをGIAAのみに適用する場合，`src/methods/mcd.py` の実装だけで完結するが，PIAAフェーズでは恩恵が限定的になる可能性がある．
- PIAAにも適用する場合，`train_PIAA.py` に `train_mcd_piaa_pretrain()` / `train_mcd_piaa_finetune()` を追加する必要があり，実装コストが増大する．

> **決定済み（2026-04-16）**：**まずGIAAのみに実装し，PIAAは後回し**。最初のスコープは `src/methods/mcd.py` の `setup()` / `trainer()` のみとし，`train_PIAA.py` への追加は別タスクとして切り出す。
>
> **PIAAを後回しにする背景**：GIAA側のDA手法は `setup(model, ...) / trainer(src_dataloaders, tgt_loader, model, ...)` という統一I/Fで，モデル・DataLoaderは外部から渡される。一方，PIAA側のDA実装（`trainer_dann_piaa_pretrain()` など）はモデル構築・DataLoader構築・DA済みNIMAウェイトのロードをすべて内部で行っており，finetune はユーザーごとのループまで内包している。GIAA用I/Fとは構造が根本的に異なるため，PIAAを `methods/` に移すには専用インターフェースの設計から始める別リファクタリング作業が必要になる。

---

### 6.6 EMD実装の統一（L1 vs L2）

**問題**：設計書3.2節ではDiscrepancy損失にL1-EMD（累積和のL1距離）を採用することが明記されているが，現在の`EarthMoverDistance`実装（`train_common.py:34`）はL2ノルム（`torch.norm(..., p=2)`）を使っている．

- ソース損失（`L_s`）と同じEMD実装をDiscrepancyにも使う（L2）のか，設計書どおりL1に統一するのか．
- DiscrepancyのみL1，ソース損失はL2という非対称な設計を許容するか．

> **決定済み（2026-04-16）**：**Discrepancy計算はL1-EMD，ソース損失は既存のL2-EMDをそのまま使用**。原論文（Section 3.2）ではL1採用の根拠として，①Ben-Davidらの定理（H∆H距離）に基づく理論的根拠，②L2距離は実験的にうまくいかなかったという実験的根拠の2点が明記されている。原論文は分類タスクだがL1優位の知見を尊重し，Discrepancy計算のみL1を適用する。実装上は既存の`EarthMoverDistance`（L2）を改修せず，MCD専用のL1-EMD関数（累積和のL1距離）を `src/methods/mcd.py` に追加する。

---

### 6.7 ハイパーパラメータの`argflags.py`への追加

MCDに必要なハイパーパラメータを`parse_arguments()`に追加する．確定が必要な項目：

| パラメータ | 候補フラグ名 | デフォルト候補 | 備考 |
|---|---|---|---|
| λ（Step B/Cのバランス） | `--mcd_lambda` | `1.0` | 固定値 |
| Step C繰り返し回数 n | `--mcd_n_steps` | `4` | 論文では2〜4を検証 |

> **決定済み（2026-04-16）**：**λは固定値（デフォルト`1.0`），`--mcd_n_steps` のデフォルトは`4`**。λのスケジューリングは行わないため `--mcd_use_schedule` フラグは追加しない。

---

### 6.8 バリデーション評価方針

**問題**：ターゲットのバリデーションセットにはラベル（ヒストグラム）があるため，EMDで評価できる．しかしこれはUDAの想定に反する（ラベルを使っていることになる）．

- **モニタリング目的**（早期停止には使わず，参考値として記録するのみ）：ターゲットValのEMDを記録してよい．
- **早期停止の基準**：DANNと同様，ソースValのEMDを基準とするか，それとも別の無監督指標（Discrepancyそのもの）を使うか．

> **決定済み（2026-04-16）**：**早期停止の基準はソースVal EMD**。以下をDANNと同様にwandbへ記録する：
> - 早期停止基準：ソースVal EMD（小さいほど良い）
> - モニタリング（参考値）：ターゲットVal EMD，各Step のDiscrepancy損失（`L_adv`）