# Roadmap de ejecucion tecnica

## Objetivo

Completar el sistema de reconocimiento facial 1:N con precision biometrica real, manteniendo arquitectura modular y operacion local-first.

## Estado actual (baseline)

- API base operativa (`/health`, personas, enroll, match, eventos).
- Demo de camara funcional.
- Busqueda 1:N con Faiss ya implementada.
- Persistencia SQLite operativa.
- Benchmarks GPU/InsightFace disponibles y validados.
- Falta pasar de `DummyFaceEncoder` a pipeline facial real para produccion.

## Principios de ejecucion

- Mantener `pytest` verde en cada incremento.
- Entregas pequenas, integrables y medibles.
- No mezclar cambios de dominio, infraestructura y API en un solo PR grande.
- Cada fase cierra con criterios de salida y evidencia (tests/benchmarks/logs).

## Fase 1 - Motor biometrico real (Sprint 1)

### Entregables

- `InsightFaceEncoder` en `infrastructure`.
- Seleccion de encoder por configuracion (`dummy` o `insightface`).
- Integracion en `bootstrap` sin romper contratos de dominio.
- Pruebas de humo de inferencia (CPU/GPU).

### Criterio de salida

- El demo y la API pueden correr con encoder real.
- Se elimina dependencia funcional del encoder dummy para flujo principal.

### Checklist

- [ ] Crear `insightface_encoder.py`.
- [ ] Extender `Settings` con `encoder_backend`, `model_name`, `providers`.
- [ ] Conectar encoder configurable en `build_services`.
- [ ] Agregar tests de inicializacion y fallback de providers.

## Fase 2 - Persistencia v2 y trazabilidad de embeddings (Sprint 2)

### Entregables

- Modelo `face_embeddings` multi-muestra (sin overwrite por persona).
- Relacion embebido <-> sample.
- Metadatos de modelo/version/dimension.
- Estructura para versionado de indice (`faiss_indexes`) y config (`system_config`).
- Migracion SQLite v1 -> v2.

### Criterio de salida

- Una persona puede tener multiples embeddings vigentes.
- Trazabilidad completa de que modelo genero cada embedding.

### Checklist

- [ ] Actualizar `sqlalchemy_models.py`.
- [ ] Actualizar repositorios SQLite e interfaces necesarias.
- [ ] Agregar script de migracion.
- [ ] Agregar pruebas de migracion y compatibilidad.

## Fase 3 - Enrolamiento robusto (Sprint 3)

### Entregables

- Enrolamiento por video corto y seleccion automatica de mejores frames.
- Quality gating y pose gating sobre rostro alineado.
- Persistencia de muestras operativas/canonicas.
- Reindexado controlado post-enrolamiento.

### Criterio de salida

- Protocolo de 10-15 muestras utiles por persona operativo.
- Menor variabilidad por luz/angulo frente a enrolamiento de frame unico.

### Checklist

- [ ] Crear pipeline de seleccion de frames.
- [ ] Guardar metricas de calidad/pose por muestra.
- [ ] Agregar comando/script de enrolamiento por video.
- [ ] Tests de calidad minima y rechazo de muestras malas.

## Fase 4 - Decision robusta en vivo (Sprint 4)

### Entregables

- Consistencia temporal por `track_id` (ventana/voto).
- Estados operativos completos: `known_person`, `unknown_person`, `ambiguous_match`, `face_not_usable`.
- Parametros de decision externalizados en config.

### Criterio de salida

- Reduccion de falsos positivos en stream.
- Decisiones mas estables entre frames consecutivos.

### Checklist

- [ ] Implementar buffer temporal por track.
- [ ] Integrar regla `threshold + margin + consistencia`.
- [ ] Registrar razones de decision en eventos/logs.
- [ ] Tests de regresion para casos ambiguos.

## Fase 5 - Percepcion de escena (MVP-B) (Sprint 5)

### Entregables

- Deteccion de persona (YOLO) + tracking.
- Estado `person_detected` separado de identidad.
- Integracion de ROI por persona para optimizar reconocimiento.

### Criterio de salida

- El sistema sigue operando aunque no haya cara usable en todos los frames.

### Checklist

- [ ] Integrar detector de personas.
- [ ] Integrar tracking y `track_id`.
- [ ] Actualizar demo de camara y eventos con estados nuevos.
- [ ] Medir costo de latencia por etapa.

## Fase 6 - Calibracion, observabilidad y hardening (Sprint 6)

### Entregables

- Toolkit de calibracion FAR/FRR/EER por camara.
- Endpoint/proceso de `rebuild` de indice.
- Metricas y logs estructurados por etapa.
- Politicas basicas de retencion, backup y acceso.

### Criterio de salida

- Umbrales calibrados con datos reales.
- SLOs de latencia y precision monitoreados en entorno operativo.

### Checklist

- [ ] Script de evaluacion offline (ROC/DET, barrido de thresholds).
- [ ] Endpoint `POST /v1/index/rebuild`.
- [ ] Export de eventos + reporte de metricas.
- [ ] Runbook operativo actualizado.

## Backlog inmediato (arranque esta semana)

### Semana 1

- [ ] Implementar `InsightFaceEncoder`.
- [ ] Parametrizar `Settings` para backend/model/providers.
- [ ] Wiring en `bootstrap`.
- [ ] Tests unitarios nuevos del encoder.

### Semana 2

- [ ] Migracion de esquema de embeddings (multi-muestra).
- [ ] Ajuste de repositorio Faiss para galeria completa.
- [ ] Pruebas de integracion SQLite + reconocimiento.
- [ ] Actualizacion de docs API y datos para reflejar cambios reales.

## Definicion de terminado por fase

- Codigo mergeable.
- Tests pasando en entorno local (`pytest -q`).
- Cambio documentado en `docs/`.
- Comando de verificacion reproducible agregado.

## Riesgos y mitigacion

- Riesgo: degradacion de latencia al usar modelos reales.
  Mitigacion: benchmark por `det_size`, perfilado por etapa y ajuste de frecuencia de reconocimiento.
- Riesgo: ruptura de compatibilidad por migracion de DB.
  Mitigacion: script de migracion idempotente + backup previo.
- Riesgo: falsos positivos por mala calibracion.
  Mitigacion: calibrar por camara con dataset operativo antes de fijar umbrales.

## Comandos de control rapido

```bash
pytest -q
python benchmarks/gpu_benchmark.py
python benchmarks/insightface_benchmark.py
python3 scripts/run_camera_demo.py --camera-index 0 --camera-id cam-01
```
