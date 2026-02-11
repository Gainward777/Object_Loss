## Краткое описание функций

Ниже — кратко, что делает каждая функция и зачем она нужна (все работают **на уже посчитанных эмбеддингах**).

### Вспомогательные (агрегация/редукция)
- **`_reduce_over_nonbatch(x, agg)`** — *сжать патчи/пространство*: сводит `[B,P,...] → [B]`, чтобы лосс был “per‑sample”.
- **`_apply_reduction(per_sample, reduction, weights)`** — *суммировать/усреднить батч*: делает `mean/sum/none`, опционально с весами.

### Дистанции между эмбеддингами
- **`l2_distance(a, b, squared, agg)`** — *Евклидово расстояние*: база для метрических лоссов.
- **`cosine_distance(a, b, normalize, agg)`** — *косинус‑дистанция*: семантическая близость (часто стабильнее для CLIP‑подобных фич).
- **`pair_distance(a, b, dist, agg)`** — *выбор метрики*: единая точка переключения `l2/cos`.

### Общие лоссы (строительные блоки)
- **`feature_match_loss(pred, target, kind, ...)`** — *приблизить к GT*: “перцептуальный” матчинг в feature‑пространстве (полезно для формы/геометрии).
- **`contrastive_pair_loss(z1, z2, y, margin, ...)`** — *пары тянуть/толкать*: классический contrastive loss (позитивы ближе, негативи дальше маржи). ([lecun.com](https://lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf))
- **`triplet_margin_loss(anchor, positive, negative, margin, ...)`** — *ранжировать тройкой*: заставляет `d(a,p) + m < d(a,n)` (triplet loss). ([arxiv.org](https://arxiv.org/abs/1503.03832))
- **`margin_ranking_on_distances(d_pos, d_neg, margin, ...)`** — *hinge на дистанциях*: то же ранжирование, но когда расстояния уже посчитаны. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.margin_ranking_loss.html))
- **`multi_negative_ranking_loss(a, p, negs, margin, ...)`** — *ранжировать против многих*: триплет‑идея, но с несколькими негативами и их агрегацией (mean/max/sum).
- **`info_nce_loss(anchor, positive, negatives, temperature, ...)`** — *выбрать позитив среди негативов*: InfoNCE как cross‑entropy по similarity‑логитам (в т.ч. in‑batch negatives). ([lilianweng.github.io](https://lilianweng.github.io/posts/2021-05-31-contrastive))

### Три “прикладных” пункта под вашу задачу
- **`loss_no_edit_identity(z_pred, z_gt, z_src, ...)`** — *наказать “ничего не поменял”*: делает `pred` ближе к `GT`, чем к `source` (борется с пунктом 1).
- **`loss_geometry_match(z_pred, z_gt, ...)`** — *держать форму объекта*: притягивает предсказание к GT в feature‑пространстве (пункт 2).
- **`loss_overlay_avoidance(z_pred, z_gt, z_overlay, ...)`** — *наказать “наложил поверх”*: делает `pred` ближе к GT, чем к “overlay”‑негативу (пункт 3).

---

## Принимаемые аргументы — что есть что

Нотация:  
- **`pred` / `z_pred`** — эмбеддинг *текущего результата модели* (то, что “с ошибкой”, потому что ещё не GT).  
- **`target` / `z_gt`** — эмбеддинг *целевого изображения after / ground truth*.  
- **`src` / `z_src`** — эмбеддинг *исходного before* (негатив “ничего не поменяли”).  
- **`overlay` / `z_overlay`** — эмбеддинг *плохого примера “наложили поверх”* (негатив).  
- Форматы: `[B,D]` или spatial/patch `[B,P,D]` (функции умеют сводить патчи в per-sample).

### Утилиты редукции
- **`_reduce_over_nonbatch(x, agg)`**  
  - `x`: любой тензор `[B,...]`  
  - `agg`: как схлопывать не-batch измерения (`mean/sum/max`)  
  - *Зачем*: перевести patch/spatial расстояния в **одну скалярную ошибку на элемент батча**.

- **`_apply_reduction(per_sample, reduction, weights)`**  
  - `per_sample`: `[B]` — уже посчитанная ошибка на каждый сэмпл  
  - `reduction`: `none/mean/sum`  
  - `weights`: `[B]` (опционально) — веса сэмплов  
  - *Зачем*: получить итоговый scalar-loss или вернуть per-sample.

### Дистанции
- **`l2_distance(a, b, squared, agg)`**  
  - `a`, `b`: эмбеддинги одной формы (обычно `pred` и `target` или `anchor` и `neg`)  
  - `squared`: L2² или L2  
  - `agg`: как сводить патчи/пространство  
  - *Выход*: `[B]` расстояния.

- **`cosine_distance(a, b, normalize, agg)`**  
  - `a`, `b`: эмбеддинги  
  - `normalize=True`: L2-нормализация перед косинусом  
  - *Выход*: `[B]` (это `1 - cos_sim`).

- **`pair_distance(a, b, dist, agg)`**  
  - `dist`: `"cos"` или `"l2"`  
  - *Зачем*: единый переключатель метрики.

### Базовые “строительные блоки” лоссов
- **`feature_match_loss(pred, target, kind, ...)`**  
  - `pred`: эмбеддинг результата (**ошибка**)  
  - `target`: эмбеддинг GT (**цель**)  
  - `kind`: `"l2"|"l1"|"cos"`  
  - *Зачем*: **притягивать** предсказание к GT в feature-space (перцептуальный/структурный матчинг).

- **`contrastive_pair_loss(z1, z2, y, margin, ...)`**  
  - `z1`, `z2`: пара эмбеддингов  
  - `y`: метка пары (`1` = позитив → сблизить; `0` = негатив → раздвинуть минимум на `margin`)  
  - *Зачем*: классический **pairwise contrastive**. ([lecun.com](https://lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf))

- **`triplet_margin_loss(anchor, positive, negative, margin, ...)`**  
  - `anchor`: “опорный” (в вашей задаче чаще всего **`pred`**)  
  - `positive`: **`gt`** (должен быть ближе к anchor)  
  - `negative`: **`src`** или **`overlay`** (должен быть дальше)  
  - *Зачем*: enforce `d(a,p)+m < d(a,n)` (triplet). ([cv-foundation.org](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf))

- **`margin_ranking_on_distances(d_pos, d_neg, margin, ...)`**  
  - `d_pos`: расстояние “правильной” пары (должно быть **меньше**)  
  - `d_neg`: расстояние “неправильной” пары (должно быть **больше**)  
  - *Зачем*: то же, что triplet-hinge, но когда вы заранее посчитали расстояния. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.margin_ranking_loss.html))

- **`multi_negative_ranking_loss(z_anchor, z_positive, z_negs, ...)`**  
  - `z_anchor`: обычно **`pred`**  
  - `z_positive`: **`gt`**  
  - `z_negs`: негативы (тензор `[B,K,D]` или список тензоров) — напр. `[src, overlay, ...]`  
  - `neg_agg`: как агрегировать по K (`mean/max/sum`)  
  - *Зачем*: триплет-логика, но **с несколькими негативами**.

- **`info_nce_loss(anchor, positive, negatives=None, temperature, ...)`**  
  - `anchor`: обычно **`pred`**  
  - `positive`: **`gt`**  
  - `negatives`: либо явные `[B,K,D]`, либо `None` → in-batch negatives  
  - `temperature`: “острота” softmax  
  - *Зачем*: классифицировать позитив среди негативов (InfoNCE). ([lilianweng.github.io](https://lilianweng.github.io/posts/2021-05-31-contrastive))

### Три прикладных лосса под ваши симптомы
- **`loss_no_edit_identity(z_pred, z_gt, z_src, ...)`**  
  - `z_pred`: эмбеддинг результата (ошибка)  
  - `z_gt`: эмбеддинг GT after (цель)  
  - `z_src`: эмбеддинг before (негатив “ничего не изменили”)  
  - *Зачем*: бьёт по проблеме **(1) “не заменяет”**.

- **`loss_geometry_match(z_pred, z_gt, ...)`**  
  - `z_pred`: результат (ошибка)  
  - `z_gt`: GT (цель)  
  - *Зачем*: бьёт по **(2) “ломает геометрию”** — при условии, что эмбеддер чувствителен к форме (patch-фичи).

- **`loss_overlay_avoidance(z_pred, z_gt, z_overlay, ...)`**  
  - `z_pred`: результат (ошибка)  
  - `z_gt`: GT (цель)  
  - `z_overlay`: негатив “наложили поверх”  
  - *Зачем*: бьёт по **(3) “оверлей”**.

---

## Примеры сборки

### Рецепт A — “заставить редактировать” (минимальный)
Если главная боль — **иногда не происходит замена**:
```python
L = w_id * loss_no_edit_identity(z_pred, z_gt, z_src, dist="cos", margin=0.2)
```
Хорошо с **CLIP/global** эмбеддингом (семантика).

### Рецепт B — “не оверлей и не identity”
Для борьбы сразу с (1) и (3):
```python
L = (w_id * loss_no_edit_identity(z_pred, z_gt, z_src, margin=0.2) +
     w_ov * loss_overlay_avoidance(z_pred, z_gt, z_overlay, margin=0.2))
```

### Рецепт C — “качество формы/геометрии”
Добавить структурный матчинг (особенно если табуретка “плывёт”):
```python
L = (w_id   * loss_no_edit_identity(z_pred_sem, z_gt_sem, z_src_sem, dist="cos", margin=0.2) +
     w_geom * loss_geometry_match(z_pred_geom, z_gt_geom, kind="l2", agg="mean"))
```
- `*_sem`: CLIP/global (замена/класс объекта)  
- `*_geom`: DINO/U-Net patch features (форма)

### Рецепт D — “несколько негативов, жёсткое ранжирование”
Когда есть много типов плохих исходов:
```python
L = w_rank * multi_negative_ranking_loss(
        z_anchor=z_pred, z_positive=z_gt,
        z_negs=[z_src, z_overlay, z_other_bad],
        dist="cos", margin=0.2, neg_agg="max"
    )
```
`neg_agg="max"` полезен, когда хочешь **наказывать самый “опасный” негатив**.

### Рецепт E — “InfoNCE-версия (softmax по негативам)”
Если у тебя нормальный batch и хочешь более “гладкий” сигнал:
```python
L = (w_nce  * info_nce_loss(z_pred, z_gt, negatives=[z_src, z_overlay], temperature=0.07) +
     w_geom * loss_geometry_match(z_pred_geom, z_gt_geom, kind="l2"))
```
InfoNCE обычно хорошо работает, когда негативов много (в т.ч. in-batch). ([lilianweng.github.io](https://lilianweng.github.io/posts/2021-05-31-contrastive))
