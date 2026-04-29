# Facial Recognition (Modular Architecture)

Arquitectura limpia y reusable para construir pipelines de facial recognition sin acoplar dominio a librerias concretas.

## Estructura

- `src/eleccia_vision/domain`: entidades y contratos puros.
- `src/eleccia_vision/application`: casos de uso/orquestacion.
- `src/eleccia_vision/infrastructure`: adaptadores concretos (encoder, repositorio).
- `src/eleccia_core`: orquestacion general y API HTTP (FastAPI).
- `src/eleccia_voice`: modulo de voz reutilizable para saludos y futuras skills.
- `tests`: pruebas unitarias y de integracion.
- `docs`: documentacion estandar del proyecto.

## Principios

- Dominio independiente de framework.
- Dependencias apuntan hacia adentro (API -> app -> domain).
- Adaptadores intercambiables por interfaces (`Protocol`).
- Wiring centralizado en `bootstrap.py`.

## Correr API

```bash
pip install -e .
uvicorn eleccia_core.api.main:app --reload
```

Para usar encoder real con InsightFace sin cambiar codigo:

```bash
# opcion recomendada: .env
cp .env.example .env
# edita .env si necesitas ajustar providers/det_size
uvicorn eleccia_core.api.main:app --reload
```

Tambien puedes usar variables de entorno directas (`ELECCIA_ENCODER_BACKEND=...`) si prefieres no usar `.env`.

## Arranque Embebido (Auto-identificacion)

Si quieres que al levantar el proceso API arranque tambien identificacion por camara:

```bash
export ELECCIA_AUTO_START_IDENTIFICATION=true
export ELECCIA_IDENTIFICATION_ARGS="--camera-index 0 --camera-id cam-01 --recognize-every 3 --voice-greet --voice-backend melotts"
uvicorn eleccia_core.api.main:app
```

## Arranque Core (Sin API)

Para levantar Eleccia Core y activar modulos desde `.env` (recomendado para embebido local):

```bash
python3 scripts/run_eleccia_core.py
```

Control de modulos:

- `ELECCIA_CORE_MODULES=vision,voice,listen`
- Modulos soportados hoy: `vision`, `voice`, `listen`
- `vision` usa `ELECCIA_IDENTIFICATION_ARGS`
- `voice` usa `ELECCIA_VOICE_*` y `ELECCIA_MELO_*`
- `run_camera_demo.py` tambien usa prefijo `ELECCIA_*`

Listener de comandos:

- `ELECCIA_LISTEN_ENABLED=true`
- `ELECCIA_LISTEN_BACKEND=stdin|whisper|openwakeword_whisper`
- `ELECCIA_LISTEN_WAKE_WORD=eleccia`
- `ELECCIA_LISTEN_WAKE_WORD_ALIASES=elexia,eleksia,elecia`
- `ELECCIA_LISTEN_WAKE_WORD_FUZZY_THRESHOLD=0.80`
- `ELECCIA_LISTEN_REQUIRE_WAKE_WORD=true`
- `ELECCIA_LISTEN_WAKE_COMMAND_WINDOW_SECONDS=6.0`
- `ELECCIA_LISTEN_STDIN_PROMPT="eleccia> "`

Voz a texto (STT) recomendada:

- Libreria recomendada: `faster-whisper` (Whisper optimizado)
- Ya incluida en dependencias del proyecto (`pyproject.toml`, `requirements.txt`)
- Instalacion directa opcional:

```bash
pip install faster-whisper
```

Perfil sugerido para GPU 48GB (cuando habilites backend whisper en `listen`):

- modelo: `large-v3`
- compute type: `float16`
- language: `es`
- `vad_filter=true`

Demo solo voz->texto:

```bash
python3 scripts/test_stt_whisper.py --backend whisper --whisper-model large-v3 --whisper-device cuda --whisper-compute-type float16 --whisper-language es --whisper-vad-filter
```

Demo wake word + Whisper:

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

Nota: para detectar la palabra custom `ELECCIA` necesitas un modelo de wakeword entrenado
(`.onnx`/`.tflite`) y pasarlo en `--openwakeword-model-paths`.

Instalacion de wakeword:

```bash
pip install openwakeword
```

La consistencia temporal se puede ajustar con `ELECCIA_TEMPORAL_CONSISTENCY_ENABLED` y `ELECCIA_TEMPORAL_MIN_CONSISTENT_FRAMES`.
Para captura de datos guiada con quality gate y angulos, usa `--guided-enroll` en `scripts/run_camera_demo.py`.

Despues de `pip install -e .`, puedes ejecutar scripts sin repetir `PYTHONPATH=src`:

```bash
python3 scripts/run_camera_demo.py --camera-index 0 --camera-id cam-01 --enroll-person-id alice --guided-enroll --guided-preset strict
```

Saludo por voz opcional en deteccion:

```bash
python3 scripts/run_camera_demo.py --camera-index 0 --camera-id cam-01 --recognize-every 3 --voice-greet --voice-backend melotts
```

Para backend MeloTTS, instala en tu entorno:

```bash
pip install "melotts @ git+https://github.com/myshell-ai/MeloTTS.git"
pip install sounddevice
```

Tambien puedes fijar defaults del demo en `.env` usando variables `ELECCIA_*` (ver `.env.example`) y correr solo:

```bash
python3 scripts/run_camera_demo.py
```

## Endpoints base

- `GET /health`
- `POST /v1/faces/enroll`
- `POST /v1/faces/match`

## Nota

La implementacion de encoder incluida es `DummyFaceEncoder` para scaffold.
Puedes reemplazarla por InsightFace/ONNX sin tocar application/domain.

## Documentacion

- `docs/README.md`: indice general.
- `docs/sprints.md`: roadmap de sprints.
- `plan_facial.md`: plan extendido de referencia.
