# DANN 振動問題の記録

## 問題

`python -m src.train_GIAA --genre art --dann_target DANN-fashion` において、

- **10 epoch 以前**: val EMD が少しずつ安定して減少
- **10 epoch 以降**: val EMD が 0.35〜0.65 の間で急上昇・急降下を繰り返す
- lambda を 0.3〜1.0 で変えても振動が解消しない
- ソースオンリー（通常の教師あり学習）と比べてソース側の EMD が変わらない

## 原因

### Adversarial Oscillation（敵対的振動）

DANN の訓練において、Discriminator と Feature Extractor が同一 Optimizer・同一 LR で更新されていたため、以下のサイクルが発生していた。

```
① acc 高（Discriminator が正確に識別できている）
   → GRL 逆勾配が小さい（∂L_d/∂z ≈ 0）
   → Feature Extractor へのドメイン混乱シグナルが弱く、L_y が支配的
   → Feature Extractor は主にタスク方向へ更新されるが、
     小さな逆勾配が蓄積しわずかにドメイン混乱方向へ微シフト
   → Discriminator の acc がわずかに低下し始める

② acc がわずかに低下 → GRL 逆勾配が大きくなる（正のフィードバック）
   → Feature Extractor がより強くドメイン混乱方向へ押される
   → 特徴がさらにドメイン混乱 → Discriminator の acc がさらに低下
   → 勾配がさらに大きくなる → Feature Extractor がさらに強く押される...
   → acc が急落（0 付近まで）

③ acc → 0（Feature Extractor が Discriminator を完全に混乱させた状態）
   → GRL 逆勾配が最大（∂L_d/∂z ≈ -1）
   → Feature Extractor はドメイン混乱をさらに強化しようとし続けるが、
   → Discriminator も同じく大きな勾配で急速に学習・回復
   → Discriminator が再び識別できるようになる → acc が急上昇

④ ① へ戻る → 振動
```

BCE の勾配の性質 `∂L_d/∂z = σ(z) - y` より、**Discriminator が正確なほど勾配は小さく（acc→1 で ≈0）、混乱するほど大きくなる（acc→0 で ≈-1）**。高 acc の均衡は不安定で、acc がわずかに下がると雪だるま式に crash する正のフィードバック構造になっている。したがって lambda のスケール調整は振動の振幅を変えるだけで、サイクル自体を止められない。

### なぜ lambda 調整が効かなかったか

lambda は GRL 内で逆勾配に定数倍するだけ。振動の根本原因は「Discriminator と Feature Extractor の更新速度のミスマッチ」であり、両者に同一 LR を使っていることで一方が少し勝つと他方が過剰反応するサイクルが止まらない。

### なぜソース側 EMD が変わらなかったか

`model.freeze_backbone()` によりバックボーンが凍結されており、GRL が動かせる層は Head のみ。バックボーンがドメイン情報を保持し続けるため、Head をいくら動かしても特徴表現全体のドメイン混乱が起きず、ソース EMD に影響が出なかった。

## 解決

### Discriminator を別 Optimizer に分離し LR を高く設定

```python
# Before
optimizer = optim.AdamW(
    list(model.parameters()) + list(discriminator.parameters()), lr=args.lr)

# After
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
optimizer_disc = optim.AdamW(
    discriminator.parameters(), lr=args.lr * 10)
```

更新も分離：

```python
optimizer.zero_grad()
optimizer_disc.zero_grad()
loss.backward()
scaler.step(optimizer)
scaler.step(optimizer_disc)
scaler.update()
```

Discriminator を「常にやや優勢」に保つことで GRL 経由の逆勾配が安定し、振動が解消。あわせて lambda の上限キャップ（誤った対処療法）も削除し、Ganin et al. の元のスケジュール（0→1）に戻した。

### 結果

- ターゲット側 val EMD が安定
- ターゲット EMD がソースオンリー比で大幅改善
- Disc Acc (tgt) の急激な振動が消えた

## 今後は

### 検討事項

| テーマ | 内容 |
|--------|------|
| Backbone 部分解凍 | 最終数層だけ学習可能にすることで、真のドメイン混乱が起きやすくなる可能性。ソース EMD とのトレードオフ要確認 |
| Gradient Clipping | GRL 経由の逆勾配爆発に対する追加の安定化手段として有効 |
| Label Smoothing | Discriminator の過信を抑え、勾配の急変を和らげる補助策 |
| optimizer_disc の LR チューニング | `lr * 10` は暫定値。Disc Acc が 0.5〜0.8 程度に安定するよう調整する余地あり |

### 設計上の教訓

- DANN の振動デバッグには `L_d` 単体でなく **Disc Acc（特にターゲット側）** を wandb で記録して見ることが有効
- lambda の調整は「振動を止める手段」ではなく「適応の強さを調整する手段」として使う
- Discriminator と Generator（Feature Extractor）は GAN と同様に**別 Optimizer で管理する**のが基本
