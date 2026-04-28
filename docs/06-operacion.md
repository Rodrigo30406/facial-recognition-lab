# Operacion

## Entorno

- Miniforge/Mamba
- Python 3.11
- Entorno recomendado: `face-lab`

## Arranque local

```bash
pip install -e ".[dev]"
uvicorn eleccia_vision.api.main:app --reload
```

## Config fija para demo (.env)

Puedes fijar comportamiento de `run_camera_demo.py` en `.env` con llaves `DEMO_*`.
Ejemplo: `DEMO_GUIDED_PRESET`, `DEMO_VOICE_GREET`, `DEMO_VOICE_BACKEND`, `DEMO_VOICE_LANG`.

Prioridad de valores:

- CLI
- `.env` (`DEMO_*` o `FACIAL_DEMO_*`)
- defaults internos

## Seleccion de encoder por entorno

Por defecto el sistema usa `dummy`.

Recomendado: usar `.env` local:

```bash
cp .env.example .env
```

Alternativa: variables de entorno directas:

```bash
export ENCODER_BACKEND=insightface
export INSIGHTFACE_MODEL_NAME=buffalo_l
export INSIGHTFACE_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
export INSIGHTFACE_CTX_ID=0
export INSIGHTFACE_DET_SIZE=320x320
export TEMPORAL_CONSISTENCY_ENABLED=true
export TEMPORAL_MIN_CONSISTENT_FRAMES=3
```

Tambien se aceptan variables con prefijo `FACIAL_` (por ejemplo `FACIAL_ENCODER_BACKEND`).

## Benchmarks disponibles

- `benchmarks/gpu_benchmark.py`
- `benchmarks/insightface_benchmark.py`

## Demo con camara

```bash
cd /home/labia10/Documentos/JNE/eleccia-asistente-virtual
conda activate face-lab
pip install -e ".[dev]"
python3 scripts/run_camera_demo.py --camera-index 0 --camera-id cam-01
```

Con enrolamiento desde la ventana (tecla `e`):

```bash
python3 scripts/run_camera_demo.py --camera-index 0 --camera-id cam-01 --enroll-person-id alice
```

Con landmarks visuales (overlay):

```bash
python3 scripts/run_camera_demo.py --camera-index 0 --camera-id cam-01 --show-landmarks --landmarks-max-points 20
```

Enrolamiento guiado con quality gate (rojo/amarillo/verde) y cobertura de angulos:

```bash
python3 scripts/run_camera_demo.py --camera-index 0 --camera-id cam-01 --enroll-person-id alice --show-landmarks --guided-enroll --guided-target-samples 12 --guided-hold-frames 3
```

Preset rapido o estricto para guided mode:

```bash
python3 scripts/run_camera_demo.py --camera-index 0 --camera-id cam-01 --enroll-person-id alice --show-landmarks --guided-enroll --guided-preset fast
python3 scripts/run_camera_demo.py --camera-index 0 --camera-id cam-01 --enroll-person-id alice --show-landmarks --guided-enroll --guided-preset strict
```

Si defines `--guided-preset`, cualquier threshold manual (`--guided-min-sharpness`, etc.) lo sobreescribe.
En `fast` y `strict`, la visualizacion de landmarks queda por defecto en `--landmarks-max-points 50` y `--landmarks-every 1` (si pasas otros valores por CLI, se respetan esos).

- `rojo`: no cumple calidad minima (luz, nitidez, tamano, pose).
- `amarillo`: calidad OK pero falta el angulo objetivo.
- `verde`: toma valida; al sostener `N` frames captura automaticamente.

Controles:

- `q`: salir
- `e`: enrolar frame actual (si se paso `--enroll-person-id`)

## Saludo por voz (opcional)

Puedes activar saludo al detectar `known_person`:

```bash
python3 scripts/run_camera_demo.py --camera-index 0 --camera-id cam-01 --recognize-every 3 --voice-greet
```

Si ya lo dejaste fijo en `.env` (variables `DEMO_*`), basta:

```bash
python3 scripts/run_camera_demo.py
```

Comportamiento:

- Saluda solo en la primera deteccion de presencia.
- Mientras la persona siga en pantalla no repite saludo.
- Si la persona sale de pantalla por `DEMO_VOICE_ABSENCE_SECONDS` y vuelve, solo saluda si ya paso `DEMO_VOICE_REENTRY_DELAY_SECONDS`.

Modo por flags:

- `--voice-greet`
- `--voice-backend auto|melotts|chattts|pyttsx3|spd-say|espeak`

Valores por config (`.env`):

- `DEMO_VOICE_TEMPLATE`
- `DEMO_VOICE_REENTRY_DELAY_SECONDS`
- `DEMO_VOICE_ABSENCE_SECONDS`
- `DEMO_VOICE_MIN_FACE_RATIO`
- `DEMO_VOICE_LANG`
- `DEMO_VOICE_RATE`
- `DEMO_VOICE_VOLUME`
- `DEMO_VOICE_ID`
- `DEMO_MELO_LANGUAGE`
- `DEMO_MELO_SPEAKER`
- `DEMO_MELO_SPEED`
- `DEMO_MELO_DEVICE`

Ejemplo (modo por flags + valores desde `.env`):

```bash
python3 scripts/run_camera_demo.py --voice-greet --voice-backend melotts
```

Instalacion sugerida para MeloTTS:

```bash
pip install "melotts @ git+https://github.com/myshell-ai/MeloTTS.git"
pip install sounddevice
```

Script de prueba de voz (sin camara):

```bash
python3 scripts/test_voice_chat.py --backend melotts --voice-lang es --interactive
```

## Runbook base

- Verificar providers ONNX (`CUDAExecutionProvider`).
- Verificar salud API (`/health`).
- Revisar latencia p95 y tasa de unknown.
- Reindexar en ventana controlada.

## Observabilidad minima

- Logs estructurados.
- Contadores por decision (`known`, `unknown`, `ambiguous`, `not_usable`).
- Latencia por etapa (deteccion, embedding, busqueda, decision).
