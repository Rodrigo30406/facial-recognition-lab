# Operacion

## Entorno

- Miniforge/Mamba
- Python 3.11
- Entorno recomendado: `face-lab`

## Arranque local

```bash
pip install -e ".[dev]"
uvicorn facial_recognition.api.main:app --reload
```

## Benchmarks disponibles

- `benchmarks/gpu_benchmark.py`
- `benchmarks/insightface_benchmark.py`

## Runbook base

- Verificar providers ONNX (`CUDAExecutionProvider`).
- Verificar salud API (`/health`).
- Revisar latencia p95 y tasa de unknown.
- Reindexar en ventana controlada.

## Observabilidad minima

- Logs estructurados.
- Contadores por decision (`known`, `unknown`, `ambiguous`, `not_usable`).
- Latencia por etapa (deteccion, embedding, busqueda, decision).
