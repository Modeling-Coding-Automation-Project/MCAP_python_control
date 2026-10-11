# Kalman Filter の使い方

`sample/kalman_filter` には、線形系・非線形系の両方で使える Kalman Filter の使い方が示されています。
主に次の3クラスが登場します。

- `LinearKalmanFilter`: 線形システム向け
- `ExtendedKalmanFilter`: 非線形システム向け
- `UnscentedKalmanFilter`: 非線形システムでより柔軟な推定向け

## 1. LinearKalmanFilter の使い方

`sample/kalman_filter/linear_kalman_filter.py` では、状態遷移行列 `A`、入力行列 `B`、観測行列 `C` を定義し、フィルタを生成しています。

```python
from python_control.kalman_filter import LinearKalmanFilter

lkf = LinearKalmanFilter(A, B, C, Q, R, Number_of_Delay)
lkf.x_hat = np.array([[0], [0], [0], [0]])

for k in range(1, num_steps):
    u = u_data[:, k - 1].reshape(-1, 1)
    lkf.predict_and_update(u, y_measured)
    x_estimate = lkf.get_x_hat_without_delay()
```

この流れは、状態推定の基本形として非常にわかりやすいです。

- `predict_and_update(u, y)`: 予測と観測更新を 1 回にまとめて実行
- `get_x_hat_without_delay()`: 遅延がある場合でも、推定後の状態を通常の時刻で確認

## 2. ExtendedKalmanFilter の使い方

`sample/kalman_filter/extended_kalman_filter.py` では、線形ではなく非線形の状態式と観測式を定義し、その Jacobian も渡します。

```python
ekf = ExtendedKalmanFilter(
    state_equation=fxu_script_function,
    measurement_equation=hx_script_function,
    state_equation_jacobian=fxu_jacobian_script_function,
    measurement_equation_jacobian=hx_jacobian_script_function,
    Q=Q_ekf,
    R=R_ekf,
    Parameters=parameters_ekf,
    Number_of_Delay=Number_of_Delay,
)

ekf.predict(u)
ekf.update(y_measured)
x_estimated = ekf.get_x_hat_without_delay()
```

ここで重要なのは、設計者が「状態更新式」と「観測式」の両方を関数として用意することです。
線形モデルでは行列で済みますが、非線形モデルでは状態遷移と観測が関数になるため、変数を関数に渡す形で利用します。

## 3. UnscentedKalmanFilter の使い方

`sample/kalman_filter/unscented_kalman_filter.py` では、状態遷移と観測式をシンボリックに定義し、それを `ExpressionDeploy` で Python 関数化しています。
その後、`UnscentedKalmanFilter` にその関数を渡します。

```python
ukf = UnscentedKalmanFilter(
    fxu_script_function,
    hx_script_function,
    Q_ukf,
    R_ukf,
    Parameters_ukf,
    Number_of_Delay,
)

ukf.predict(u)
ukf.update(y_store[delay_index])
x_estimated = ukf.get_x_hat_without_delay()
```

このクラスは、EKF よりも非線形性を直接扱いやすい設計です。
「シグマ点」を使って分布を近似するため、状態が複雑な場合でも安定して推定しやすくなります。

## 4. 使い分けの目安

| 使いどころ | 推奨クラス | 典型的な入力 |
|---|---|---|
| 線形モデルで簡単に状態を推定したい | `LinearKalmanFilter` | `A`, `B`, `C`, `Q`, `R` |
| 非線形モデルだが Jacobian を使える | `ExtendedKalmanFilter` | 関数 + Jacobian |
| 非線形モデルで分布近似が必要 | `UnscentedKalmanFilter` |  関数 + Q, R |

## 5. 推定ループの考え方

```mermaid
flowchart LR
    A[現在の状態 x_hat] --> B[predict(u)]
    B --> C[次状態を予測]
    C --> D[観測 y を受ける]
    D --> E[update(y)]
    E --> F[状態補正]
    F --> G[次のループへ]
```

重要なのは、`predict` と `update` を分離して考えることです。
`predict` は「現状で次の状態を想定する」処理で、`update` は「観測値を使って補正する」処理です。

特に `Number_of_Delay` を利用するサンプルでは、入力の遅延を考慮した状態補正が行われていることがあり、実運用に近い挙動を再現しやすくなっています。

次へ: [最小二乗とオンライン推定の使い方](./04-estimation.md)
