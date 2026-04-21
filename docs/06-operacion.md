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

## Demo con camara

```bash
cd /home/odt063/JNE/facial_recognition
conda activate face-lab
PYTHONPATH=src python scripts/run_camera_demo.py --camera-index 0 --camera-id cam-01
```

Con enrolamiento desde la ventana (tecla `e`):

```bash
PYTHONPATH=src python scripts/run_camera_demo.py --camera-index 0 --camera-id cam-01 --enroll-person-id alice
```

Controles:

- `q`: salir
- `e`: enrolar frame actual (si se paso `--enroll-person-id`)

## Runbook base

- Verificar providers ONNX (`CUDAExecutionProvider`).
- Verificar salud API (`/health`).
- Revisar latencia p95 y tasa de unknown.
- Reindexar en ventana controlada.

## Observabilidad minima

- Logs estructurados.
- Contadores por decision (`known`, `unknown`, `ambiguous`, `not_usable`).
- Latencia por etapa (deteccion, embedding, busqueda, decision).
