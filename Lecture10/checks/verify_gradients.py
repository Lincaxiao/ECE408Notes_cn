"""Numerically verify the lecture's sigmoid-sigmoid-softmax loss gradients.
Run: python checks/verify_gradients.py
Requires NumPy; writes numeric_results.json beside this script.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def evaluate(params: dict[str, np.ndarray], x: np.ndarray, target: int):
    w1, b1, w2, b2 = (params[n] for n in ('W1', 'b1', 'W2', 'b2'))
    fc1 = w1 @ x + b1
    y = sigmoid(fc1)
    fc2 = w2 @ y + b2
    z = sigmoid(fc2)
    expz = np.exp(z - np.max(z))
    k = expz / expz.sum()
    q = (np.arange(k.size) - target) ** 2
    loss = float(k @ q)
    gz = k * (q - loss)
    d2 = gz * z * (1.0 - z)
    gy = w2.T @ d2
    d1 = gy * y * (1.0 - y)
    grads = {'W1': np.outer(d1, x), 'b1': d1,
             'W2': np.outer(d2, y), 'b2': d2}
    cache = dict(fc1=fc1, y=y, fc2=fc2, z=z, k=k, q=q,
                 loss=loss, gz=gz, delta2=d2, gy=gy, delta1=d1)
    return loss, grads, cache


def main() -> None:
    params = {'W1': np.array([[.1, -.2], [.3, .25]]),
              'b1': np.array([.05, -.1]),
              'W2': np.array([[.4, -.3], [.1, .2], [-.2, .5]]),
              'b2': np.array([0., .1, -.05])}
    x = np.array([1., 2.]); target = 1; learning_rate = .1
    loss, grads, cache = evaluate(params, x, target)
    h = 1e-6
    errors = {}
    for name, values in params.items():
        numeric = np.zeros_like(values)
        for index in np.ndindex(values.shape):
            plus = {n: a.copy() for n, a in params.items()}
            minus = {n: a.copy() for n, a in params.items()}
            plus[name][index] += h
            minus[name][index] -= h
            numeric[index] = (evaluate(plus, x, target)[0]
                              - evaluate(minus, x, target)[0]) / (2*h)
        errors[name] = float(np.max(np.abs(numeric - grads[name])))
    updated = {n: a - learning_rate * grads[n] for n, a in params.items()}
    updated_loss = evaluate(updated, x, target)[0]
    assert max(errors.values()) < 1e-7, errors
    assert updated_loss < loss
    assert abs(cache['gz'].sum()) < 1e-14
    for x0, x1 in ((0, 0), (0, 1), (1, 0), (1, 1)):
        h_or = np.sign(x0 + x1 - .5)
        h_and = np.sign(x0 + x1 - 1.5)
        output = np.sign(2*h_or - h_and - 2)
        assert output == (1 if x0 != x1 else -1)
    assert 784*10 + 10 + 10*10 + 10 == 7960
    result = {'parameters': params, 'x': x, 'target': target,
              'learning_rate': learning_rate, 'forward_backward': cache,
              'gradients': grads, 'updated_loss': updated_loss,
              'central_difference_step': h, 'max_absolute_errors': errors,
              'parameter_count_checked': sum(a.size for a in params.values())}
    text = json.dumps(result, ensure_ascii=False, indent=2,
                      default=lambda a: a.tolist() if isinstance(a, np.ndarray) else float(a))
    Path(__file__).with_name('numeric_results.json').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
