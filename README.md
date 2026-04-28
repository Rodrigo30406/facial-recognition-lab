# Facial Recognition (Modular Architecture)

Arquitectura limpia y reusable para construir pipelines de facial recognition sin acoplar dominio a librerias concretas.

## Estructura

- `src/eleccia_vision/domain`: entidades y contratos puros.
- `src/eleccia_vision/application`: casos de uso/orquestacion.
- `src/eleccia_vision/infrastructure`: adaptadores concretos (encoder, repositorio).
- `src/eleccia_vision/api`: endpoints HTTP (FastAPI).
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
uvicorn eleccia_vision.api.main:app --reload
```

Para usar encoder real con InsightFace sin cambiar codigo:

```bash
# opcion recomendada: .env
cp .env.example .env
# edita .env si necesitas ajustar providers/det_size
uvicorn eleccia_vision.api.main:app --reload
```

Tambien puedes usar variables de entorno directas (`ENCODER_BACKEND=...`) si prefieres no usar `.env`.

La consistencia temporal se puede ajustar con `TEMPORAL_CONSISTENCY_ENABLED` y `TEMPORAL_MIN_CONSISTENT_FRAMES`.
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

Tambien puedes fijar defaults del demo en `.env` usando variables `DEMO_*` (ver `.env.example`) y correr solo:

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
