# 最小二乗とオンライン推定の使い方

`sample/least_squares` では、回帰モデルを学習して予測する方法が示されています。
ここでは、`LeastSquares` と `RecursiveLeastSquares` の使い方に絞って説明します。

## 1. LeastSquares の使い方

`sample/least_squares/least_squares_method.py` では、2つの特徴量 `x1`, `x2` から目的変数 `y` を予測するモデルを作っています。

```python
from python_control.least_squares import LeastSquares

ls = LeastSquares(X)
ls.fit(X, y)
predictions = ls.predict(X)
print(ls.get_weights())
```

このクラスは、最初に `X` の列数から「特徴量の数」を認識し、内部でバイアス項を含めた重み行列を用意します。
その後、`fit(X, y)` で最小二乗法により重みを算出し、`predict(X)` で新しいデータに対する予測ができます。

### 使いどころ

- 事前に十分な学習データが揃っている
- バッチ処理で一度にモデルを作りたい
- 回帰係数を可視化したい

## 2. RecursiveLeastSquares の使い方

`sample/least_squares/recursive_squares_method.py` では、データを1件ずつ受け取りながら重みを更新しています。

```python
from python_control.least_squares import RecursiveLeastSquares

rls = RecursiveLeastSquares(feature_size=X.shape[1], lambda_factor=0.9)

for i in range(n_samples):
    x = X[i].reshape(-1, 1)
    y_true = y[i].reshape(-1, 1)
    rls.update(x, y_true)

predictions = rls.predict(X)
```

`update(x, y_true)` は、一回のデータごとに重みを少しずつ変えるため、オンライン推定や逐次同定に向いています。

## 3. どちらを使うべきか

| アプローチ | クラス | 特徴 |
|---|---|---|
| 一括でモデルを作る | `LeastSquares` | 学習データが揃ってから最適化 |
| 流れ続けるデータに対応 | `RecursiveLeastSquares` | 逐次更新で重みを調整 |

## 4. 学習の流れ

```mermaid
flowchart LR
    A[入力データ X] --> B[fit(X, y) / update(x, y_true)]
    B --> C[重みの算出]
    C --> D[predict(X)]
    D --> E[予測値の出力]
```

## 5. 使うときのポイント

- `LeastSquares` は 1 回の学習で完結するので、固定モデルの作成に向いています
- `RecursiveLeastSquares` は、環境が変化しやすい状況で重みが追従しやすいです
- `get_weights()` を呼ぶと、現在の係数を簡単に確認できます
- どちらも入力の形状に注意が必要です。サンプルでは `X` が `(n_samples, n_features)` の2次元配列で、各サンプルがベクトルとして扱われています

このリポジトリでは、回帰や推定の導入として最小二乗を使い、さらに制御に進む前の「モデルの学習」を理解しやすくしています。

ここまでで、代表的なクラスの利用方法を押さえることができます。必要に応じて、各サンプルスクリプトを実行して、可視化と数値を照らし合わせながら学習を進めてください。
