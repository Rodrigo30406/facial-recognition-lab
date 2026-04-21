# Facial Recognition (Modular Architecture)

Arquitectura limpia y reusable para construir pipelines de facial recognition sin acoplar dominio a librerias concretas.

## Estructura

- `src/facial_recognition/domain`: entidades y contratos puros.
- `src/facial_recognition/application`: casos de uso/orquestacion.
- `src/facial_recognition/infrastructure`: adaptadores concretos (encoder, repositorio).
- `src/facial_recognition/api`: endpoints HTTP (FastAPI).
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
uvicorn facial_recognition.api.main:app --reload
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
