# DeepJDOT 実装設計書
## タスク：美的評価分布のUnsupervised Domain Adaptation

---

## 1. タスク概要

### 1.1 問題設定

- **タスク：** 画像の美的評価分布予測におけるUDA（Unsupervised Domain Adaptation）
- **ドメイン：** art / fashion / scenery の3ドメイン間で適応
- **入力：** 画像
- **出力：** 美的評価スコアの分布（例：評価者の評点分布）
- **教師なし：** ターゲットドメインのラベルは学習時に使用しない

### 1.2 通常の分類タスクとの違い

```
通常の分類タスク：
  出力 → クラスラベル（例：犬 or 猫）

本タスク：
  出力 → 評価スコアの分布（例：[0.1, 0.2, 0.4, 0.2, 0.1]）
  → 複数の評価者がそれぞれ異なるスコアをつける
  → その分布を予測したい
```

---

## 2. モデル構成

### 2.1 全体アーキテクチャ

```
入力画像 x
    ↓
g：特徴抽出器（Feature Extractor）
    ↓
潜在表現 z = g(x)   ← ここでOTを計算
    ↓
f：分布予測器（Distribution Predictor）
    ↓
予測分布 f(g(x))
```


---

## 3. 損失関数

### 3.1 ラベルコストとしてのEMD

通常のDeepJDOTでは交差エントロピーを使うが、
本タスクでは**EMD（Earth Mover's Distance）** を採用する。

### 3.2 DeepJDOTの目的関数（EMD版）

$$\min_{\gamma, f, g} \frac{1}{n_s}\sum_i L_s(y^s_i, f(g(x^s_i))) + \sum_{i,j} \gamma_{ij} \left[ \alpha\|g(x^s_i) - g(x^t_j)\|^2 + \lambda_t \cdot \text{EMD}(y^s_i, f(g(x^t_j))) \right]$$

| 項 | 内容 |
|---|---|
| ① ソース損失 Ls | ソースドメインでEMDを最小化（破滅的忘却を防ぐ） |
| ② 特徴整合 | γで対応づけられたペアの特徴を近づける |
| ③ ラベル整合 | γで対応づけられたペアのEMDを最小化 |

---

## 4. 交互最適化

### 4.1 γの更新

```
f, gを固定してコスト行列Cを計算：

Cᵢⱼ = α・||g(xˢᵢ) - g(xᵗⱼ)||²
     + λt・EMD(yˢᵢ, f(g(xᵗⱼ)))

→ OTソルバーでγを求める
```

### 4.2 f, gの更新

```
γを固定して勾配法で更新：

Loss = ①ソース損失 + ②特徴整合損失 + ③ラベル整合損失

→ 誤差逆伝播で f, g のパラメータを更新
```

### 4.3 アルゴリズム

```
for each batch (ソースバッチ, ターゲットバッチ):
    # ステップ1：γの更新
    z_s = g(x_s)
    z_t = g(x_t)
    C = compute_cost_matrix(z_s, z_t, y_s, f)  # EMDを使用
    γ = ot_solver(C)

    # ステップ2：f, gの更新
    loss = source_loss(y_s, f(z_s))
           + feature_align_loss(γ, z_s, z_t)
           + label_align_loss(γ, y_s, f(z_t))
    loss.backward()
    optimizer.step()
```

---

## 5. マルチドメイン適応の設計

3ドメイン（art / fashion / scenery）をどう扱うか：

### ペアワイズ適応

```
art → fashion
art → scenery
fashion → scenery
...
各ペアに対して独立にDeepJDOTを適用
```

## 6. ハイパーパラメータ（要検討）

| パラメータ | 意味 | 初期値候補 |
|---|---|---|
| α | 特徴整合の重み | 要調整 |
| λt | ラベル整合の重み | 要調整 |
| m | ミニバッチサイズ | 32 |
| lr | 学習率 | 1e-5 |

---

## 7. 実装上の注意点

### 7.1 OTソルバー

```
推奨ライブラリ：POT（Python Optimal Transport）
  import ot
  γ = ot.emd(a, b, C)  # network simplex flow
```

### 7.2 γのdetach

```
γの更新はOTソルバーで解くため
勾配グラフから切り離す必要がある

γ = ot.emd(a, b, C).detach()
```

---

## 8. 未決定事項・要検討リスト

### 8.1 OT計算に使う特徴量の選択

現在NIMAの`forward(return_feat=True)`は3種類の特徴量を返す：

| 変数 | 意味 | 次元 |
|---|---|---|
| `raw_feat` | backboneの生出力 | 512（clip_vit_b16）|
| `domain_feat` | feat_proj後の射影 | 256（feat_dim）|

**検討事項：** OTのコスト行列（特徴距離項）にどちらを使うか？
- `domain_feat`（256dim）：DANNと統一、計算軽量
- `raw_feat`（512dim）：より豊富な表現だが、feat_projの学習がOTに直接影響しない

**指示：**
`domain_feat`（256dim）を使って

---

### 8.2 OTソルバーの選択

| 選択肢 | 特徴 | 注意点 |
|---|---|---|
| `ot.emd` | 正確なLP解（network simplex） | 計算コスト O(n³)、勾配不可 |
| `ot.sinkhorn` | エントロピー正則化、微分可能 | 近似解、正則化強度ε要調整 |

**現状：** batch_size=16 のためemdでも現実的だが、コスト行列内のEMD計算（n_s × n_t = 256回）と合わせたボトルネックを測定する必要あり。

**指示:**
正確なLP解（network simplex）を使って

---

### 8.3 コスト行列内のEMD計算の計算量

コスト行列の各要素 C_ij に含まれるEMD計算：

```
n_s × n_t = batch_size² = 16² = 256 回 / バッチ
```

各EMDは `num_bins=7` の1D分布に対するCDF距離（既存の `earth_mover_distance` で計算可能）。

**検討事項：** ベクトル化できるか？ブロードキャストで `(n_s, n_t, num_bins)` テンソルとして一括計算する実装を検討。

**指示：** ブロードキャストで `(n_s, n_t, num_bins)` テンソルとして一括計算する実装を検討して

---

### 8.4 バッチサイズとOT精度のトレードオフ

ミニバッチOTはバッチが小さいほど輸送計画の精度が低下する。

**検討事項：**
- batch_size=16（現在のDANNと同じ）で十分か
- DeepJDOT論文の推奨バッチサイズとの比較
- ソース・ターゲットのバッチサイズは揃える必要があるか（DANNでは独立）

**指示：**　batch_size=32にして．バッチ内のソース・ターゲット比率が一緒なら問題ないです

---

### 8.5 Early Stoppingの基準

DANNでは **ソースval EMD** を基準にしているが、DeepJDOTではターゲットへの適応が目的。

**検討事項：**
- ソースval EMDで early stopping → ターゲット適応を見落とすリスク
- ターゲットval EMD（ラベルなし評価）をモニタリングしつつ、ソースEMDで保存するか
- 現在の `trainer` / `trainer_dann_giaa` のどちらをベースに `trainer_djdot_giaa` を作るか

**指示：**　DeepJDOTでもソースval EMDで early stopping して

---

### 8.6 γのマージナル制約

OT計算の周辺分布 a, b をどう設定するか：

| 設定 | 内容 |
|---|---|
| uniform | `a = 1/n_s`, `b = 1/n_t` （デフォルト、最も一般的）|
| 重み付き | データセットサイズ比を考慮 |

**検討事項：** art/fashion/scenery 間でデータ数の偏りがある場合の影響。

**指示：**　データ数はほとんど一緒なのでuniformで

---

### 8.7 ハイパーパラメータ α, λt の探索戦略

設計書6章の初期値が未定。

**参考：** DeepJDOT論文ではα=0.001, λt=0.1 が使われる例があるが、EMDをラベルコストに使う場合のスケールが異なる可能性あり（交差エントロピーと異なりEMDのスケールは[0, √num_bins]）。

**検討事項：** ラベルコストにEMDを使う場合、特徴距離（L2二乗）とのスケール合わせが必要か。

**指示：**　初期値は，α  = 0.001，λt = 0.0001で．あとで，手動で調整します


---

### 8.8 マルチドメイン適応の実行順序

設計書5章でペアワイズ適応を採用する方針だが未確定。

**検討事項：**
- ペアワイズ（art→fashion, art→scenery, fashion→scenery）を独立して実行するか
- DANNの `domain_tag = f'{genre}2{dann_target_genre}'` の命名規則をDeepJDOTでも踏襲するか（例: `DJDOT-fashion`）

**指示：**　ペアワイズだけ想定して．命名規則をDeepJDOTでも踏襲して
