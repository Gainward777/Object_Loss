# Depth Consistency Loss for Object Inpainting / Insertion (Flux Kontext LoRA)

Этот README описывает **loss-функцию согласованности глубины** (depth-consistency loss) для обучения модели, которая **вставляет (inpaint/insert)** объект в заданную область изображения.

## Когда это нужно

Даже если RGB-лоссы “в целом работают”, размер/геометрия вставленного объекта может **плавать**.

Если у вас есть:

- реалистичные рендеры **context** и **target**,
- **точные GT depth** карты (из рендера),
- и вы умеете получать **pred depth** из предсказанного изображения (например, через MiDaS),

то можно добавить регуляризатор, который проверяет:

1) **Внутри маски вставки** глубина должна совпадать с глубиной таргета.  
2) **Снаружи маски** глубина тоже должна совпадать с таргетом (в вашей постановке `target == context` вне маски).  
3) **На границе маски** должны совпадать **градиенты глубины** (чтобы “скачок” глубины и контур объекта получались правильными).

> MiDaS вычисляет **relative inverse depth** (относительную обратную глубину) по одному изображению. citeturn0search5  
> В этом README мы считаем, что все карты глубины уже **подготовлены** и приведены к одному виду (это вы вынесли за скобки).

---

## Что принимает лосс

Лосс **не** выполняет преобразования глубины — он принимает готовые карты:

- `d_pred`: depth/disparity для **предсказанного** изображения (из вашего пайплайна глубины)
- `d_tgt`: GT depth/disparity для **target** рендера (подготовленная так же, как `d_pred`)
- `mask`: маска вставки (1 внутри области вставки)
- `conf_pred` *(опционально)*: карта уверенности `d_pred` в диапазоне `[0..1]`

Требования по формам:
- `d_pred`: `[B, 1, H, W]` (или `[B, H, W]` / `[H, W]` — будет приведено)
- `d_tgt`: то же разрешение, что `d_pred`
- `mask`: то же разрешение, что `d_pred`
- `conf_pred` (если есть): то же разрешение, что `d_pred`

---

## Идея лосса (простыми словами)

1) **Depth match inside(mask)**: если вы вставляете объект, то внутри маски глубина должна стать такой, как у таргета.  
2) **Depth match outside(mask)**: вне маски сцена не должна “переезжать” по глубине.  
3) **Depth edge match on boundary**: края/контуры по глубине должны совпасть (это сильно помогает масштабу и “посадке” объекта).

Градиентные термы для улучшения резкости глубины — распространённая практика: в работах по depth-estimation часто добавляют gradient loss для sharper edges. citeturn0search10

---

## Почему SmoothL1 (Huber-like), а не MSE

Мы используем `smooth_l1_loss` как робастную метрику: она менее чувствительна к выбросам и локальным ошибкам depth-оценщика.

Параметр `beta` задаёт порог, где лосс переходит от квадратичного режима к L1-режиму. citeturn0search4turn0search0

---

## Формулы

Обозначения:
- \(d_{pred}\) — depth для предсказания
- \(d_{tgt}\) — depth для таргета
- \(M\in[0,1]\) — маска вставки
- \(C\in[0,1]\) — confidence mask (если есть)
- \(V\) — валидность пикселя (не NaN/Inf)
- \(B\) — полоса границы (boundary band)
- \(\rho(\cdot)\) — SmoothL1 (Huber-like) ошибка

### 1) Внутри маски
\[
L_{in} = \mathrm{mean}_w\;\rho(d_{pred} - d_{tgt}),\quad w = M \cdot V \cdot C
\]

### 2) Снаружи маски (sure outside)
Вне маски мы используем область “sure outside”, исключая узкую окрестность границы (через dilate), чтобы не штрафовать пиксели, где depth наиболее шумный:

\[
L_{out} = \mathrm{mean}_w\;\rho(d_{pred} - d_{tgt}),\quad w = (1-\mathrm{dilate}(M)) \cdot V \cdot C
\]

### 3) Градиенты на границе
\[
L_{edge} = \mathrm{mean}_w\;\rho(\nabla d_{pred} - \nabla d_{tgt}),\quad w = B \cdot V \cdot C
\]

### Итог
\[
L = \lambda_{in}L_{in} + \lambda_{out}L_{out} + \lambda_{edge}L_{edge}
\]

---

## Реализация (PyTorch)

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _as_4d(x: torch.Tensor, like: torch.Tensor, *, name: str) -> torch.Tensor:
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(1)
    elif x.dim() == 4:
        pass
    else:
        raise ValueError(f"{name} must have 2..4 dims, got {x.shape}")

    if x.shape[0] != like.shape[0]:
        if x.shape[0] == 1:
            x = x.expand(like.shape[0], -1, -1, -1)
        else:
            raise ValueError(f"{name} batch {x.shape[0]} != like batch {like.shape[0]}")

    return x.to(device=like.device, dtype=torch.float32)


def dilate(mask01: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return mask01
    k = 2 * r + 1
    return F.max_pool2d(mask01, kernel_size=k, stride=1, padding=r)


def erode(mask01: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return mask01
    k = 2 * r + 1
    return 1.0 - F.max_pool2d(1.0 - mask01, kernel_size=k, stride=1, padding=r)


def boundary_band(mask_bin: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return torch.zeros_like(mask_bin)
    d = dilate(mask_bin, r)
    e = erode(mask_bin, r)
    return (d - e).clamp_(0.0, 1.0)


def depth_gradients(depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dx = depth[..., :, 1:] - depth[..., :, :-1]
    dy = depth[..., 1:, :] - depth[..., :-1, :]

    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return dx, dy


def weighted_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    beta: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    per = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    wsum = weight.sum().clamp_min(eps)
    return (per * weight).sum() / wsum


@dataclass
class DepthConsistencyTerms:
    total: torch.Tensor
    inside: torch.Tensor
    outside: torch.Tensor
    edge: torch.Tensor


def depth_consistency_loss(
    d_pred: torch.Tensor,
    d_tgt: torch.Tensor,
    mask: torch.Tensor,
    *,
    conf_pred: Optional[torch.Tensor] = None,
    outside_dilate_r: int = 1,
    edge_band_r: int = 1,
    beta: float = 0.1,
    w_inside: float = 1.0,
    w_outside: float = 0.5,
    w_edge: float = 0.5,
    eps: float = 1e-8,
) -> DepthConsistencyTerms:
    like = d_pred if d_pred.dim() == 4 else d_pred.unsqueeze(1)
    d_pred = _as_4d(d_pred, like, name="d_pred")
    d_tgt  = _as_4d(d_tgt,  like, name="d_tgt")
    m      = _as_4d(mask,   like, name="mask").clamp(0.0, 1.0)
    m_bin  = (m > 0.5).to(m.dtype)

    valid = (torch.isfinite(d_pred) & torch.isfinite(d_tgt)).to(d_pred.dtype)
    if conf_pred is not None:
        c = _as_4d(conf_pred, like, name="conf_pred").clamp(0.0, 1.0)
        valid = valid * c

    outside_region = 1.0 - dilate(m_bin, outside_dilate_r)
    inside_region  = m
    band_region    = boundary_band(m_bin, edge_band_r)

    w_in   = inside_region * valid
    w_out  = outside_region * valid
    w_band = band_region * valid

    L_in  = weighted_smooth_l1(d_pred, d_tgt, w_in,  beta=beta, eps=eps)
    L_out = weighted_smooth_l1(d_pred, d_tgt, w_out, beta=beta, eps=eps)

    dx_p, dy_p = depth_gradients(d_pred)
    dx_t, dy_t = depth_gradients(d_tgt)

    L_edge_x = weighted_smooth_l1(dx_p, dx_t, w_band, beta=beta, eps=eps)
    L_edge_y = weighted_smooth_l1(dy_p, dy_t, w_band, beta=beta, eps=eps)
    L_edge = 0.5 * (L_edge_x + L_edge_y)

    total = w_inside * L_in + w_outside * L_out + w_edge * L_edge
    return DepthConsistencyTerms(total=total, inside=L_in, outside=L_out, edge=L_edge)
```

---

## Примеры использования

### Пример 1 — базовый (без confidence)

```python
terms = depth_consistency_loss(
    d_pred=d_pred_prepared,  # [B,1,H,W]
    d_tgt=d_tgt_prepared,    # [B,1,H,W]
    mask=mask01,             # [B,1,H,W]
    outside_dilate_r=1,
    edge_band_r=1,
    beta=0.1,
    w_inside=1.0,
    w_outside=0.5,
    w_edge=0.5,
)

L_total = L_other + 0.2 * terms.total   # 0.1..0.3 — хороший стартовый вес depth-терма
L_total.backward()
```

### Пример 2 — с confidence

```python
terms = depth_consistency_loss(
    d_pred=d_pred_prepared,
    d_tgt=d_tgt_prepared,
    mask=mask01,
    conf_pred=conf_pred01,   # [B,1,H,W] in [0,1]
)

L_total = L_other + 0.2 * terms.total
L_total.backward()
```

### Пример 3 — логирование компонент

```python
terms = depth_consistency_loss(...)
log({
    "loss/depth_total": terms.total.item(),
    "loss/depth_in": terms.inside.item(),
    "loss/depth_out": terms.outside.item(),
    "loss/depth_edge": terms.edge.item(),
})
```

---

## Как тюнить (коротко)

- **Объект “уменьшается” / не заполняет маску**  
  ↑ `w_inside`, ↑ `w_edge`, иногда ↑ общий вес depth-терма

- **Граница по глубине размыта**  
  ↑ `w_edge` или `edge_band_r` (обычно до 2)

- **Вне маски сцена портится**  
  ↑ `w_outside` и/или общий вес depth-терма,  
  также проверьте `outside_dilate_r` (чтобы “outside” был достаточно “далеко” от границы)

- **depth-предсказание местами шумное**  
  Подайте `conf_pred` — вклад сомнительных пикселей уменьшится.

---

## Notes

- Если `d_pred` и `d_tgt` не выровнены по масштабу/конвенции, лосс будет некорректен.
- Маска должна совпадать по разрешению с depth-картами.
- NaN/Inf игнорируются через `isfinite`, но большое количество невалидных пикселей — признак проблем в depth пайплайне.

---

## References

- MiDaS: “computes relative inverse depth from a single image.” citeturn0search5  
- PyTorch `smooth_l1_loss` / `SmoothL1Loss` (параметр `beta` и поведение). citeturn0search4turn0search0  
- Пример добавления gradient loss для улучшения sharp edges глубины (depth estimation). citeturn0search10
