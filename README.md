# AoT Segmentation — Titanes Puros vs Humanos a 1080p

Solución al reto de segmentación semántica de *Attack on Titan*: un modelo ONNX ligero que pinta cada píxel de un
fotograma 1920×1080 como **Fondo (0)**, **Titán Puro (1)** o **Humano (2)**, con un límite de 150 MB y 500 ms por
fotograma en CPU.

**Resultado final en el test público: 88.10 % Foreground Micro-mIoU** (Titán 91.81, Humano 84.38).

Las reglas, la taxonomía y el contrato ONNX del reto están en [KIT.md](KIT.md).

## El problema

El kit trae **63 fotogramas de entrenamiento sin etiquetas** y 8 de validación con máscaras. Toda la supervisión
había que fabricarla, y el modelo final tenía que correr en CPU dentro del tiempo límite del servidor.

## Cómo se resolvió

### 1. Profesor: auto-anotación con modelos grandes
[`aot/teacher.py`](aot/teacher.py) — Grounding DINO detecta figuras, SAM 2.1 recorta cada una y CLIP propone si es
titán o humano. [`aot/compose.py`](aot/compose.py) junta las instancias en mapas de etiquetas y aplica las
correcciones de una revisión a mano, frame por frame:

- [`work/overrides_train.json`](work/overrides_train.json): clase corregida por instancia.
- [`work/manual_train.json`](work/manual_train.json): correcciones con prompts de SAM, polígonos y relleno de huecos
  ([`aot/manual.py`](aot/manual.py)).
- Caballos detectados y verificados con CLIP como negativos duros ([`aot/horses.py`](aot/horses.py),
  [`aot/verify_horses.py`](aot/verify_horses.py)).

Profesor + revisión llega a 0.909 mIoU en val.

### 2. Alumno: DinoSeg
[`aot/dinoseg.py`](aot/dinoseg.py) — con tan pocos fotogramas, un backbone afinado de punta a punta aprende atajos
(figura grande = titán, caballo = humano). La arquitectura final separa dos trabajos:

- Un **ViT-S/16 DINOv3** casi congelado (tasa de aprendizaje ×0.02) decide *qué* es cada región, corriendo a media
  resolución.
- Una **rama convolucional delgada** decide *dónde* están los bordes.

Preprocesamiento, reescalado, upsampling y `argmax` van dentro del grafo ONNX ([`aot/export.py`](aot/export.py)).
Entrada 448×800, 87 MB, ~200 ms en un M4 con un hilo (~390 ms en el servidor).

Receta final ([`work/queue_final_b.sh`](work/queue_final_b.sh)): etiquetas revisadas v3 + GT de val ×4, 5000
iteraciones, zoom 1.0–2.0, recortes de primer plano al 20 %.

### 3. Validación honesta
Con 8 fotogramas de val, un punto de diferencia es ruido. Para decidir entre versiones:

- Cada receta se entrenó primero **sin val** y se comparó en val y en un **examen privado**
  ([`aot/exam.py`](aot/exam.py)) que nunca entra al entrenamiento.
- Las diferencias se midieron con bootstrap pareado por fotograma y con varias semillas.
- Historial completo de experimentos: [`work/logs/exams.txt`](work/logs/exams.txt).

| Envío | Modelo | Test público |
| :--- | :--- | ---: |
| 1 | SegFormer-B1, 320×576 | 80.68 % |
| 2 | DinoSeg delgado, 3000 its | 86.90 % |
| 3 | DinoSeg delgado, 5000 its | **88.10 %** |

Lo que se probó y no funcionó: zoom más amplio, neblina sintética, SegFormer-B0 a más resolución, ConvNeXt-DINOv3,
afinar el ViT con más fuerza, y entrenar con capturas externas de la wiki (pierde titanes en el A/B honesto).

## Estructura

```
aot/        auto-anotación, entrenamiento, exportación, examen y herramientas de revisión
src/        módulos oficiales del kit (parser de GT, contrato ONNX, métricas)
scripts/    validador oficial del kit
work/       colas de experimentos (*.sh), correcciones manuales (*.json) e historial de resultados
```

## Uso

Los fotogramas del kit no se incluyen en el repositorio; se colocan en `data/` como describe [KIT.md](KIT.md).

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-train.txt

# Entrenar y exportar la receta final (zsh)
./work/queue_final_b.sh

# Validar un ONNX con el validador oficial
.venv/bin/python scripts/validate_onnx_submission.py --model work/onnx_exp/ds_final_b.onnx --val-gt data/val_in_domain/masks
```
