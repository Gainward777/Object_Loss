# Depth Consistency Loss for Object Inpainting / Insertion (Flux Kontext LoRA)

Этот README описывает **loss-функцию согласованности глубины** (depth-consistency loss) для обучения модели, которая **вставляет (inpaint/insert)** объект в заданную область изображения.

## Проверяется:

1) **Внутри маски вставки** глубина должна совпадать с глубиной таргета.  
2) **Снаружи маски** глубина тоже должна совпадать с таргетом (в вашей постановке `target == context` вне маски).  
3) **На границе маски** должны совпадать **градиенты глубины** (чтобы “скачок” глубины и контур объекта получались правильными).

### **Важно!**

 > __*MiDaS вычисляет **relative inverse depth** (относительную обратную глубину) по одному изображению. citeturn0search5*__
 > 
 > __*В этом README мы считаем, что все карты глубины уже **подготовлены** и приведены к одному виду (это вы вынесли за скобки).*__ 

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
