# Arquitectura

## Vista logica

```text
Camara/RTSP -> Deteccion persona (opcional en MVP-A) -> Tracking (MVP-B)
            -> Deteccion y alineamiento facial -> Quality/Pose gating
            -> Embedding -> Busqueda 1:N (Faiss)
            -> Decision (known/unknown/ambiguous/not_usable)
            -> Eventos + API
```

## Capas del codigo (actual repo)

- `domain`: entidades y contratos.
- `application`: casos de uso.
- `infrastructure`: adaptadores concretos.
- `api`: FastAPI.

## Componentes objetivo

- `capture_service`
- `person_detection_service`
- `tracking_service`
- `face_service`
- `recognition_service`
- `enrollment_service`
- `storage_service`
- `api_service`

## Decisiones tecnicas clave

- InsightFace para deteccion/alineamiento/embedding.
- ONNX Runtime con GPU cuando disponible.
- Faiss para busqueda 1:N.
- SQLite + SQLAlchemy para persistencia local.
- Similitud coseno sobre embeddings normalizados (Faiss `IndexFlatIP`).

## Presupuesto inicial de latencia (SLO)

- p95 deteccion+embedding <= 120 ms por frame usable.
- p95 decision 1:N <= 150 ms extremo a extremo.
- Throughput objetivo de pipeline: >= 15 FPS procesados por camara.
