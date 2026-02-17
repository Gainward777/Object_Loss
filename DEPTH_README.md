# Depth-aware scale loss (документация)

Этот документ описывает набор функций **depth-aware scale loss**, которые помогают стабилизировать **масштаб (размер в пикселях/латенте)** врисованного объекта, привязывая его к **глубине сцены** и **фокусному расстоянию** камеры.

Подходит для вашего кейса:
- **одна LoRA = один объект**;
- вставка происходит **в заданную маску**;
- есть пары **(context → target)**;
- основное обучение в **латентном пространстве**.

> Важно: depth-aware — это **регуляризатор** поверх “жёстких” масочных лоссов (Dice/Boundary/Area).  
> Его цель — уменьшить редкие “срывы” масштаба, которые могут оставаться даже при корректной маске.

---

## 1) Интуиция и модель

В pinhole-модели масштаб проекции на изображении примерно пропорционален:

- **в пикселях**: `scale ∝ f_px / Z`  
  где `f_px` — фокусное расстояние в пикселях, `Z` — глубина (distance) точки/объекта.

Многие monocular depth модели (например, MiDaS) выдают **relative inverse depth** (величину ~ `1/Z`, но с неизвестным масштабом/сдвигом). Поэтому удобно писать:

- если `d ≈ 1/Z` (inverse depth): `scale ∝ f_px * d`
- если `Z` метрическая (metric depth): `scale ∝ f_px / Z`

Чтобы это стало применимо на практике, вводим **калибровочную константу объекта** `k_obj`, которая связывает:
- выбранную вами **меру размера** на маске (например `sqrt(area)`),
- со сценовой геометрией (`f_px` и depth).

---

## 2) Что именно считается “scale” (размер)

В реализациях ниже “предсказанный размер” берётся как **прокси** из маски:

### 2.1 `sqrt(area)` (по умолчанию)
- `A = sum(mask)`
- `s = sqrt(A)`

Почему так:
- масштаб растёт примерно линейно с линейным размером объекта;
- площадь растёт квадратично → `sqrt(area)` ближе к линейной шкале.

Это работает хорошо, если маска объекта примерно “однотипной формы” между примерами (а у вас один объект — обычно так).

> Альтернатива (если захотите расширить): bbox height/width или периметр.  
> Если вам нужно — могу дописать функции для bbox-варианта и обновить docs.

---

## 3) Входные данные и требования

### 3.1 Маска
- `pred_mask01`: soft mask в `[0,1]`, чаще всего это ваша `M_hat` (soft change mask из латентов).
- Разрешение — **то же**, что у depth map.

### 3.2 Depth map
Вы можете использовать любой замороженный depth estimator:
- **MiDaS**: отдаёт *relative inverse depth*.
- **Depth Anything V2**: доступны варианты, включая модели, фокусированные на metric depth (в зависимости от версии/файна).

Важно:
- depth должен быть приведён к **тому же разрешению**, что и `pred_mask01` (обычно латентное).
- мы по умолчанию используем робастную статистику `median` внутри области объекта.

### 3.3 Focal length `f_px`
Нужно получить фокус в пикселях:
- либо напрямую из intrinsics (`fx`),
- либо из EXIF `f_mm` + ширины сенсора.

Базовая формула:
- `f_px = (f_mm / sensor_width_mm) * image_width_px`

---

## 4) Основные функции

## 4.1 `focal_mm_to_px(focal_mm, image_width_px, sensor_width_mm)`

**Зачем:** преобразовать EXIF-фокус (мм) в фокус в пикселях.

**Сигнатура:**
```python
f_px = focal_mm_to_px(focal_mm, image_width_px, sensor_width_mm)
```

**Аргументы:**
- `focal_mm`: `Tensor[B]` или scalar
- `image_width_px`: `Tensor[B]` или scalar
- `sensor_width_mm`: `Tensor[B]` или scalar

**Выход:**
- `f_px`: `Tensor[B]`

---

## 4.2 `depth_aware_scale_loss(...)`

**Зачем:** штрафовать несоответствие масштаба объекта ожидаемому по глубине и фокусу.

### Формула

Пусть:
- `s_pred` — масштаб из `pred_mask01` (по умолчанию `sqrt(area)`).
- `d_roi` — статистика depth внутри маски (median или mean).
- `f_px` — фокус в пикселях.
- `k_obj` — константа для данного объекта.

Тогда ожидаемый масштаб:

- если `depth_mode="inv_depth"` (MiDaS-подобное `d≈1/Z`):
  - `s_exp = k_obj * f_px * d_roi`
- если `depth_mode="depth"` (метрическое `Z`):
  - `s_exp = k_obj * f_px / (Z_roi + eps)`

Лосс — квадрат лог-отношения (стабилен и scale-invariant):
- `L = mean( log((s_pred+eps)/(s_exp+eps))^2 )`

### Сигнатура

```python
loss, info = depth_aware_scale_loss(
    pred_mask01=M_hat,
    depth_map=depth_lat,
    f_px=f_px_batch,
    k_obj=k_obj,
    depth_mode="inv_depth",   # или "depth"
    depth_stat="median",      # или "mean"
    detach_depth=True,
)
```

### Аргументы

- `pred_mask01`: `[B,1,H,W]` soft mask (0..1)
- `depth_map`: `[B,1,H,W]` depth или inverse depth
- `f_px`: `[B]` или scalar
- `k_obj`: scalar или `[B]` (обычно scalar на объект)
- `depth_mode`: `"inv_depth"` или `"depth"`
- `depth_stat`: `"median"` (рекомендуется) или `"mean"`
- `scale_measure`: сейчас поддержан `"sqrt_area"`
- `detach_depth`: по умолчанию True, чтобы не гонять градиенты в depth estimator (он frozen)

### Возврат

- `loss`: scalar `Tensor`
- `info`: `DepthScaleLossInfo` c полями:
  - `s_pred`: `[B,1,1,1]`
  - `s_exp`: `[B,1,1,1]`
  - `depth_stat`: `[B,1,1,1]`

---

## 4.3 `estimate_k_obj_from_dataset(...)`

**Зачем:** калибровать `k_obj` по обучающему датасету, чтобы depth-aware формула стала “в размерности вашей метрики масштаба”.

### Идея калибровки

Используем GT-маску (или маску таргета, которая у вас совпадает с маской врисовки):

- `s_gt = sqrt(area_gt)`
- `d_roi` / `Z_roi` как статистика depth в области объекта

Для `inv_depth`:
- `s_gt ≈ k_obj * f_px * d_roi`
- `k_i = s_gt / (f_px * d_roi)`

Для `depth`:
- `s_gt ≈ k_obj * f_px / Z_roi`
- `k_i = s_gt * Z_roi / f_px`

Дальше берём **медиану** `k_i` по датасету (робастно).

### Сигнатура

```python
k_obj = estimate_k_obj_from_dataset(
    gt_mask01=M_lat,
    depth_map=depth_lat,
    f_px=f_px_batch,
    depth_mode="inv_depth",
    depth_stat="median",
)
```

### Аргументы

- `gt_mask01`: `[B,1,H,W]` бинарная/soft GT маска в нужном разрешении
- `depth_map`: `[B,1,H,W]`
- `f_px`: `[B]` или scalar
- `depth_mode`: `"inv_depth"` или `"depth"`
- `depth_stat`: `"median"` или `"mean"`

### Возврат
- `k_obj`: scalar `Tensor` (медиана по датасету)

---

## 5) Примеры использования

## 5.1 MiDaS (relative inverse depth) + латентное разрешение

```python
import torch
import torch.nn.functional as F

# z_ctx, z_tgt, z_pred: [B,C,H_lat,W_lat]
# M_hat: soft change mask [B,1,H_lat,W_lat]
# M_lat: GT mask [B,1,H_lat,W_lat]
# depth_px: MiDaS output in pixel res [B,1,H_img,W_img] (inverse depth)

depth_lat = F.interpolate(depth_px, size=(H_lat, W_lat), mode="bilinear", align_corners=False)

# 1) калибруем k_obj (один раз на объект)
k_obj = estimate_k_obj_from_dataset(
    gt_mask01=M_lat,
    depth_map=depth_lat,
    f_px=f_px_batch,          # [B]
    depth_mode="inv_depth",
    depth_stat="median",
)

# 2) на шаге обучения добавляем регуляризатор
L_depth, info = depth_aware_scale_loss(
    pred_mask01=M_hat,
    depth_map=depth_lat,
    f_px=f_px_batch,
    k_obj=k_obj,
    depth_mode="inv_depth",
    depth_stat="median",
    detach_depth=True,
)

L_total = L_total + 0.10 * L_depth
```

Рекомендованный старт-вес: `λ_depth = 0.05 … 0.15`.

---

## 5.2 Metric depth (если depth_model выдаёт Z в метрах)

```python
depth_lat = F.interpolate(depth_meters, size=(H_lat, W_lat), mode="bilinear", align_corners=False)

k_obj = estimate_k_obj_from_dataset(
    gt_mask01=M_lat,
    depth_map=depth_lat,
    f_px=f_px_batch,
    depth_mode="depth",
)

L_depth, _ = depth_aware_scale_loss(
    pred_mask01=M_hat,
    depth_map=depth_lat,
    f_px=f_px_batch,
    k_obj=k_obj,
    depth_mode="depth",
)

L_total = L_total + 0.10 * L_depth
```

---

## 5.3 Конвертация EXIF фокуса в `f_px`

```python
# focal_mm: [B] из EXIF
# sensor_width_mm: [B] из таблицы по модели камеры / EXIF / справочника
# image_width_px: [B] ширина кадра в пикселях

f_px = focal_mm_to_px(focal_mm, image_width_px, sensor_width_mm)
```

---

## 6) Практические рекомендации

1) **Используйте `median` по depth внутри маски.**  
   Mean чувствителен к “дыркам”, отражениям, краям маски.

2) **Всегда калибруйте `k_obj` на датасете** (даже если depth “metric”).  
   Потому что ваш `s_pred` — это прокси (sqrt(area) или bbox), а не реальная высота в метрах.

3) **`detach_depth=True` почти всегда правильнее.**  
   Вы не хотите оптимизировать depth estimator (он frozen), и не хотите нестабильные градиенты от него.

4) **Нормализуйте depth map** (если нужно) *перед* подачей в лосс.  
   Для MiDaS scale/shift могут плавать между изображениями. Калибровка `k_obj` обычно компенсирует это, но если видите нестабильность, попробуйте:
   - нормировать depth per-image (например, z-score по всему кадру),
   - или брать depth statistic относительно фона.

5) **Если маска маленькая** (очень мало пикселей в латенте), depth-aware станет шумным.  
   Тогда:
   - увеличьте разрешение, на котором считаете этот лосс,
   - либо используйте bbox-высоту вместо sqrt(area).

---

## 7) Где подключать depth-aware в суммарном лоссе

Как правило:
- depth-aware добавляют **после** того, как масочные лоссы уже стабилизировали заполнение области.
- вес небольшой: `0.05…0.15`.

Пример:

```python
L = (
    1.00 * L_mse +
    0.60 * L_bg +
    0.90 * L_dice +
    0.20 * L_area +
    0.40 * L_bnd +
    0.10 * L_depth
)
```

---

## 8) Ссылки/источники (для понимания)

- MiDaS: relative inverse depth (Torch Hub / repo)
- Depth Anything V2: официальные материалы / репозиторий, включая упоминания metric depth вариантов
- Pinhole camera model: стандартные конспекты/лекции по camera model
- Формула перевода focal mm → px (через sensor width)

(В основном README проекта вы можете держать кликабельные ссылки; здесь намеренно без подробного цитирования, чтобы документ был компактным.)
