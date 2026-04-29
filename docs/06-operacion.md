# Operacion

## Entorno

- Miniforge/Mamba
- Python 3.11
- Entorno recomendado: `face-lab`

## Arranque local

```bash
pip install -e ".[dev]"
uvicorn eleccia_core.api.main:app --reload
```

## Arranque embebido (auto-identificacion)

Para que el proceso API inicie tambien el modulo de identificacion al arrancar:

```bash
export ELECCIA_AUTO_START_IDENTIFICATION=true
export ELECCIA_IDENTIFICATION_ARGS="--camera-index 0 --camera-id cam-01 --recognize-every 3 --voice-greet --voice-backend melotts"
uvicorn eleccia_core.api.main:app
```

Opcional:

- `ELECCIA_IDENTIFICATION_ARGS`: argumentos extra para `scripts/run_vision_runtime.py`.

## Arranque Core (sin API)

Para ejecutar Eleccia Core como runtime local y que levante modulos automaticamente:

```bash
python3 scripts/run_eleccia_core.py
```

Config relevante en `.env`:

- `ELECCIA_CORE_MODULES=vision,voice,listen,mqtt`
- `ELECCIA_IDENTIFICATION_ARGS` para modulo `vision`
- `ELECCIA_VOICE_ENABLED`, `ELECCIA_VOICE_BACKEND`, `ELECCIA_VOICE_*`, `ELECCIA_MELO_*` para modulo `voice`
- `ELECCIA_LISTEN_ENABLED=true` habilita listener de comandos
- `ELECCIA_MQTT_ENABLED=true` habilita publish de intents al broker MQTT
- `ELECCIA_MQTT_HOST`, `ELECCIA_MQTT_PORT`, `ELECCIA_MQTT_TOPIC_PREFIX`, `ELECCIA_MQTT_QOS`, `ELECCIA_MQTT_RETAIN`
- `ELECCIA_LISTEN_BACKEND=stdin|whisper|openwakeword_whisper`
- `ELECCIA_LISTEN_WAKE_WORD=eleccia`
- `ELECCIA_LISTEN_WAKE_WORD_ALIASES=elexia,eleksia,elecia`
- `ELECCIA_LISTEN_WAKE_WORD_FUZZY_THRESHOLD=0.80`
- `ELECCIA_LISTEN_REQUIRE_WAKE_WORD=true`
- `ELECCIA_LISTEN_WAKE_COMMAND_WINDOW_SECONDS=6.0`
- Whisper config: `ELECCIA_LISTEN_WHISPER_*` (ver `.env.example`)
- openWakeWord config: `ELECCIA_LISTEN_OPENWAKEWORD_*` (ver `.env.example`)
- Mutex global de audio: `ELECCIA_AUDIO_LOCK_FILE`, `ELECCIA_AUDIO_LOCK_STRICT`, `ELECCIA_AUDIO_LOCK_TIMEOUT_SECONDS`
- Prefijo `ELECCIA_*`: usado por `eleccia_core` y por `scripts/run_vision_runtime.py`

Topic MQTT de intents:

- `<ELECCIA_MQTT_TOPIC_PREFIX>/events/intent`
- Payload JSON: `intent`, `text`, `confidence`, `slots`, `timestamp`
- Acciones internas no dependen de MQTT (ejemplo: `camera_on`/`camera_off` inicia/detiene modulo `vision` localmente).

## STT recomendado (Whisper)

Para capturar comandos por voz, la opcion recomendada es `faster-whisper`.

Instalacion:

```bash
pip install faster-whisper
```

Nota:

- `faster-whisper` ya fue agregado al proyecto en `pyproject.toml` y `requirements.txt`.
- En la siguiente fase se habilita backend `whisper` en `eleccia_listen` (hoy el backend activo es `stdin`).

Preset sugerido para tu GPU 48GB:

- modelo `large-v3`
- `compute_type=float16`
- `language=es`
- `vad_filter=true`

Prueba standalone (solo STT):

```bash
python3 scripts/test_stt_whisper.py --backend whisper --whisper-model large-v3 --whisper-device cuda --whisper-compute-type float16 --whisper-language es --whisper-vad-filter
```

Prueba standalone (wakeword + STT):

```bash
python3 scripts/test_stt_whisper.py \
  --backend openwakeword_whisper \
  --openwakeword-model-paths /abs/path/eleccia.onnx \
  --openwakeword-threshold 0.5 \
  --whisper-model large-v3 \
  --whisper-device cuda \
  --whisper-compute-type float16 \
  --whisper-language es \
  --whisper-vad-filter
```

## Config fija para runtime (.env)

Puedes fijar comportamiento de `run_vision_runtime.py` en `.env` con llaves `ELECCIA_*`.
Ejemplo: `ELECCIA_GUIDED_PRESET`, `ELECCIA_VOICE_ENABLED`, `ELECCIA_VOICE_BACKEND`, `ELECCIA_VOICE_LANG`.

Prioridad de valores:

- CLI
- `.env` (`ELECCIA_*`)
- defaults internos

## Seleccion de encoder por entorno

Por defecto el sistema usa `dummy`.

Recomendado: usar `.env` local:

```bash
cp .env.example .env
```

Alternativa: variables de entorno directas:

```bash
export ELECCIA_ENCODER_BACKEND=insightface
export ELECCIA_INSIGHTFACE_MODEL_NAME=buffalo_l
export ELECCIA_INSIGHTFACE_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
export ELECCIA_INSIGHTFACE_CTX_ID=0
export ELECCIA_INSIGHTFACE_DET_SIZE=320x320
export ELECCIA_TEMPORAL_CONSISTENCY_ENABLED=true
export ELECCIA_TEMPORAL_MIN_CONSISTENT_FRAMES=3
```

## Benchmarks disponibles

- `benchmarks/gpu_benchmark.py`
- `benchmarks/insightface_benchmark.py`

## Demo con camara

```bash
cd /home/labia10/Documentos/JNE/eleccia-asistente-virtual
conda activate face-lab
pip install -e ".[dev]"
python3 scripts/run_vision_runtime.py --camera-index 0 --camera-id cam-01
```

Con enrolamiento desde la ventana (tecla `e`):

```bash
python3 scripts/run_vision_runtime.py --camera-index 0 --camera-id cam-01 --enroll-person-id alice
```

Con landmarks visuales (overlay):

```bash
python3 scripts/run_vision_runtime.py --camera-index 0 --camera-id cam-01 --show-landmarks --landmarks-max-points 20
```

Enrolamiento guiado con quality gate (rojo/amarillo/verde) y cobertura de angulos:

```bash
python3 scripts/run_vision_runtime.py --camera-index 0 --camera-id cam-01 --enroll-person-id alice --show-landmarks --guided-enroll --guided-target-samples 12 --guided-hold-frames 3
```

Preset rapido o estricto para guided mode:

```bash
python3 scripts/run_vision_runtime.py --camera-index 0 --camera-id cam-01 --enroll-person-id alice --show-landmarks --guided-enroll --guided-preset fast
python3 scripts/run_vision_runtime.py --camera-index 0 --camera-id cam-01 --enroll-person-id alice --show-landmarks --guided-enroll --guided-preset strict
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
python3 scripts/run_vision_runtime.py --camera-index 0 --camera-id cam-01 --recognize-every 3 --voice-greet
```

Si ya lo dejaste fijo en `.env` (variables `ELECCIA_*`), basta:

```bash
python3 scripts/run_vision_runtime.py
```

Comportamiento:

- Saluda solo en la primera deteccion de presencia.
- Mientras la persona siga en pantalla no repite saludo.
- Si la persona sale de pantalla por `ELECCIA_VOICE_ABSENCE_SECONDS` y vuelve, solo saluda si ya paso `ELECCIA_VOICE_REENTRY_DELAY_SECONDS`.
- El toggle para saludo en camara runtime es `ELECCIA_VISION_VOICE_GREET` (independiente de `ELECCIA_VOICE_ENABLED` del modulo core).

Modo por flags:

- `--voice-greet`
- `--voice-backend auto|melotts|pyttsx3|spd-say|espeak`

Valores por config (`.env`):

- `ELECCIA_VISION_VOICE_GREET`
- `ELECCIA_VOICE_TEMPLATE`
- `ELECCIA_VOICE_REENTRY_DELAY_SECONDS`
- `ELECCIA_VOICE_ABSENCE_SECONDS`
- `ELECCIA_VOICE_MIN_FACE_RATIO`
- `ELECCIA_VOICE_LANG`
- `ELECCIA_VOICE_RATE`
- `ELECCIA_VOICE_VOLUME`
- `ELECCIA_VOICE_ID`
- `ELECCIA_MELO_LANGUAGE`
- `ELECCIA_MELO_SPEAKER`
- `ELECCIA_MELO_SPEED`
- `ELECCIA_MELO_DEVICE`

Ejemplo (modo por flags + valores desde `.env`):

```bash
python3 scripts/run_vision_runtime.py --voice-greet --voice-backend melotts
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
