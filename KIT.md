# AoTSegmentation: Reto de Segmentación Semántica 1080p (Pure Titan vs Human)

Kit oficial del alumno para diseñar, entrenar, validar y exportar modelos ligeros en formato ONNX capaces de segmentar **Titanes Puros (`Pure_Titan`)** frente a **Humanos (`Human`)** en fotogramas 1080p (`1920x1080`) de *Attack on Titan*.

---

## Contenido del Kit

- `data/train/images/` — `63` fotogramas RGB (`1920x1080`) de la Temporada 1 **sin etiquetas**. Deben diseñar su propio pipeline de auto-anotación, destilación o curación asistida para generar supervisión sobre este split.
- `data/val_in_domain/images/` — `8` fotogramas RGB (`1920x1080`) de validación dentro del dominio.
- `data/val_in_domain/masks/` — `8` máscaras Ground Truth oficiales en formato PNG `uint8` de un solo canal.
- `src/` — Módulos oficiales de parseo (`src/data/gt_parser.py`), contrato ONNX (`src/evaluation/onnx_runner.py`) y cálculo de métricas (`src/metrics/segmentation_metrics.py`).
- `scripts/validate_onnx_submission.py` — Validador CLI local de contrato ONNX, latencia y métricas sobre `val_in_domain`.
- `requirements.txt` — Dependencias mínimas para ejecutar el validador local.

---

## Taxonomía Oficial (3 Clases Activas + Void)

| ID | Clase | Descripción |
| :--- | :--- | :--- |
| `0` | `Background` | Cielo, muros, tejas, árboles, edificios, vapor, caballos u objetos ajenos. |
| `1` | `Pure_Titan` | Exclusivamente Titanes Puros desnudos. |
| `2` | `Human` | Humanos con ropa, capas del Cuerpo de Exploración o equipo 3DM. |
| `255` | `Void` | Regiones ambiguas ignoradas en la evaluación. **Prohibido predecir `255` en el modelo.** |

> **Nota de visualización:** Las máscaras PNG en `data/val_in_domain/masks/` contienen índices crudos `{0, 1, 2, 255}`, por lo que en visores convencionales se ven casi negras (`1/255` y `2/255` de brillo). Para inspeccionarlas a color, usen `create_color_mask` y `create_overlay` desde `src/data/gt_parser.py`.

---

## Contrato Obligatorio del Modelo ONNX

Todo archivo `.onnx` enviado a la plataforma debe cumplir exactamente con esta interfaz:

- **Entrada:** nombre `"image"`, shape `[1, 3, 1080, 1920]` (`NCHW` RGB), tipo `uint8`, rango `[0, 255]`.
- **Salida:** nombre `"mask"`, shape `[1, 1080, 1920]`, tipo `uint8` o `int64`, valores estrictamente en `{0, 1, 2}`.
- **Opset permitido:** `17` a `20`.
- **Límites operativos del portal en línea:**
  - Peso máximo del `.onnx`: **`150 MB`** (el validador local advierte límite técnico de `250 MB`, pero el servidor de evaluación rechaza archivos mayores a `150 MB`).
  - Latencia máxima por fotograma: **`500 ms`** en CPU (se verifica en los primeros 5 frames de prueba; timeout global de `90 s`).
  - Cooldown entre envíos por alumno: **`120 s`**.

> **Tip de arquitectura:** Integren el preprocesamiento (`uint8 -> float32`, normalización `ImageNet`, reescalado interno si su backbone corre a `512x960` o `544x960`), el upsampling de logits a `1080x1920` y el `argmax` directamente dentro del grafo ONNX exportado.

---

## Métrica Oficial y Desempate

- **Métrica Principal:** `Foreground Micro-mIoU` global sobre `{1: Pure_Titan, 2: Human}` con banda de tolerancia morfológica de contorno de **`3 px`**. Predecir todo `Background (0)` obtiene `0.0000`.
- **Métricas Secundarias y Desempate en Leaderboard:**
  1. Mayor `Foreground Micro-mIoU`
  2. Menor tamaño del archivo `.onnx` (`Size MB`)
  3. Menor número de intentos (`Tries`)
  4. Entrega más temprana (`Timestamp`)
  5. Menor `latency_p50_ms`
  *(Nota: `Titan Count MAE` se reporta únicamente como métrica diagnóstica).*

---

## Validación Local Antes de Subir

Configurar el entorno virtual `.venv` e instalar dependencias mínimas:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Validar contrato, latencia y métricas sobre `data/val_in_domain`:

```powershell
.\.venv\Scripts\python.exe scripts/validate_onnx_submission.py `
  --model ruta/a/tu_modelo.onnx `
  --val-gt data/val_in_domain/masks
```
