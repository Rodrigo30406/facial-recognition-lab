# Plan de implementación — Sistema local de detección de personas e identificación facial 1:N

## 1. Objetivo

Construir un sistema **local** para oficina/lab que:

- detecte **personas** en video en tiempo real,
- detecte **rostros** cuando haya una cara usable,
- identifique si el rostro pertenece a una persona enrolada en la base local (**1:N**),
- clasifique al resto como **desconocido / visitante**,
- y degrade con gracia cuando solo se vea el cuerpo o una cara no apta para reconocimiento.

El sistema debe estar pensado para:

- cámara fija algo elevada,
- poses no perfectamente frontales,
- enrolamiento con múltiples muestras por persona,
- operación **offline/local-first**,
- crecimiento incremental de usuarios sin reentrenar un clasificador cerrado.

---

## 2. Enfoque recomendado

### 2.1 Enfoque principal

Usar una arquitectura de:

**detección de persona + detección/alineamiento facial + embeddings faciales + búsqueda vectorial**

En vez de entrenar un clasificador tradicional por persona, la identidad se resolverá con:

1. extracción de **embeddings** faciales,
2. almacenamiento local de embeddings enrolados,
3. búsqueda 1:N sobre un índice vectorial,
4. decisión final basada en umbral, margen y consistencia temporal.

### 2.2 Por qué este enfoque

Ventajas:

- permite **altas y bajas** de personal sin reentrenamiento global,
- escala mejor para 1:N que comparar listas manualmente,
- es más flexible ante múltiples muestras por persona,
- se adapta mejor a condiciones reales que un pipeline rígido de fotos frontales,
- es apropiado para funcionamiento local.

---

## 3. Tecnologías recomendadas

### 3.1 Reconocimiento facial

**InsightFace** como librería principal de reconocimiento facial.

Uso propuesto:

- detección/alineamiento facial integrado,
- extracción de embeddings faciales,
- ejecución vía ONNX Runtime.

Modelos sugeridos para empezar:

- `buffalo_l` como baseline principal,
- evaluar variantes más ligeras si el hardware queda corto.

### 3.2 Runtime de inferencia

**ONNX Runtime**

- `onnxruntime` para CPU,
- `onnxruntime-gpu` si luego se usa NVIDIA.

### 3.3 Detección general de personas

**Ultralytics YOLO** para detectar la clase `person` y opcionalmente trackear personas en video.

Esto permite:

- detectar presencia aunque no haya cara usable,
- limitar la búsqueda facial a regiones de interés,
- tener estados separados entre “hay alguien” y “sé quién es”.

### 3.4 Índice vectorial

**Faiss** para búsqueda eficiente de embeddings 1:N.

Uso propuesto:

- índice en memoria reconstruible al iniciar,
- persistencia local de embeddings + metadatos,
- búsqueda top-k por similitud.

### 3.5 Persistencia local

**SQLite**

Uso propuesto:

- personas,
- muestras enroladas,
- embeddings,
- eventos de reconocimiento,
- configuración local.

### 3.6 Captura / video / utilitarios

- **OpenCV** para lectura de cámara, RTSP/USB, recorte y visualización,
- `numpy` para manejo de tensores y vectores,
- `pydantic` para configuración/modelos,
- `FastAPI` para API local y panel simple,
- `uvicorn` para servir la API,
- `loguru` o `structlog` para logs.

### 3.7 Opcionales

- **MediaPipe Face Landmarker** si más adelante se quiere estimación de pose facial más fina,
- anti-spoofing con variantes de **MiniFASNet / Silent Face Anti-Spoofing**,
- tracker explícito si el tracking del detector no fuera suficiente.

---

## 4. Arquitectura lógica del sistema

```text
Cámara / RTSP / USB
    ↓
Detección de persona
    ↓
Tracking por persona (track_id)
    ↓
Detección de rostro dentro del track
    ↓
Alineamiento + quality gating + pose gating
    ↓
Selección de mejor frame o mejores frames
    ↓
Embedding facial
    ↓
Búsqueda 1:N en Faiss
    ↓
Decisión: conocido / desconocido / no identificable
    ↓
Logs, snapshots, eventos y acciones
```

### 4.1 Estados operativos recomendados

El sistema debe distinguir al menos estos estados:

- `person_detected`: hay una persona en escena.
- `face_detected`: hay una cara visible.
- `face_not_usable`: la cara existe, pero la calidad o pose no son suficientes.
- `known_person`: rostro identificado con suficiente confianza.
- `unknown_person`: hay rostro usable, pero no coincide con la base.
- `ambiguous_match`: hay similitud, pero no suficiente para una identificación segura.

Esto evita forzar reconocimiento cuando no se debe.

---

## 5. Decisión técnica importante: dos tiempos

Sí se recomienda un pipeline en **dos tiempos**:

### Tiempo 1 — observación / selección

Mientras se trackea a la persona, se analizan varios frames y se les asigna un score usando:

- nitidez,
- tamaño del rostro,
- iluminación,
- frontalidad / pose,
- estabilidad temporal.

### Tiempo 2 — reconocimiento

Solo cuando haya uno o varios frames suficientemente buenos:

- se genera embedding,
- se consulta Faiss,
- se decide identidad.

### Recomendación práctica

No intentar reconocer todos los frames. En su lugar:

- mantener un buffer de frames por `track_id`,
- elegir el mejor frame,
- o promediar embeddings de los mejores 3 frames.

Esto suele rendir mejor que reconocer frame por frame.

---

## 6. Acerca de malla facial vs landmarks

### Recomendación principal

Para este proyecto, la **malla facial completa** no debe ser la pieza principal del reconocimiento.

Lo recomendado es:

- usar **landmarks + alignment** para reconocimiento,
- y, si hace falta, usar estimación de pose como apoyo para decidir si un frame es utilizable.

### Cuándo sí considerar face mesh

Solo como herramienta auxiliar para:

- estimación de pose más fina,
- liveness más sofisticado,
- análisis geométrico o expresiones,
- aplicaciones AR.

### Conclusión práctica

Para identificación 1:N, el mayor retorno está en:

- enrolamiento realista,
- selección de buenos frames,
- consistencia temporal,
- calibración de umbrales,
- y buenos embeddings.

No en meter 468 puntos por defecto.

---

## 7. Estrategia de enrolamiento

### 7.1 Principio clave

La galería de enrolamiento debe reflejar tanto:

- una identidad “limpia” o canónica,
- como el **dominio real de captura** de la cámara instalada.

### 7.2 Recomendación por persona

Capturar dos grupos de muestras:

#### A. Muestras canónicas

- frente,
- 15° izquierda / derecha,
- 30° izquierda / derecha,
- ligera inclinación arriba / abajo,
- con y sin lentes si aplica,
- buena iluminación y foco.

#### B. Muestras operativas

- capturadas desde la **posición real** de la cámara,
- altura real,
- distancia real,
- iluminación real,
- paso natural o postura natural,
- idealmente tomadas como video corto y luego seleccionar los mejores frames.

### 7.3 Cantidad inicial sugerida

Por persona:

- 5 muestras canónicas,
- 5 a 10 muestras operativas,
- total sugerido inicial: **10 a 15 muestras útiles**.

### 7.4 Qué evitar

- perfiles extremos de 90°,
- imágenes borrosas,
- caras demasiado pequeñas,
- imágenes muy oscuras,
- duplicados casi idénticos sin valor añadido.

### 7.5 Flujo sugerido de enrolamiento

1. Registrar persona en la app.
2. Tomar video corto desde punto real.
3. Extraer automáticamente mejores frames.
4. Validar calidad mínima.
5. Generar embeddings.
6. Guardar imágenes, embeddings y metadatos.
7. Reindexar Faiss.

---

## 8. Estrategia de reconocimiento

### 8.1 Búsqueda

Para cada embedding obtenido:

- normalizar el vector a norma L2=1,
- consultar top-k en Faiss,
- recuperar los mejores candidatos,
- evaluar score top-1 y top-2.

Para este proyecto se recomienda usar:

- similitud coseno,
- implementada como `IndexFlatIP` (inner product) sobre embeddings normalizados.

### 8.2 Regla de decisión sugerida

Identificar solo si se cumplen varias condiciones:

- `top1_similarity >= threshold`,
- `(top1 - top2) >= margin`,
- consistencia de identidad en varios frames consecutivos o en un buffer temporal,
- calidad mínima del frame.

Si no se cumplen:

- `unknown_person`, o
- `ambiguous_match`, o
- `face_not_usable`.

### 8.3 Recomendación de implementación

No fijar thresholds arbitrarios para producción sin pruebas. Primero calibrar con:

- usuarios reales,
- cámara real,
- luz real,
- distancia real,
- pose real.

### 8.4 Política de umbrales (arranque)

Definir una política explícita por versión de modelo para evitar drift silencioso:

- `model_name` + `model_version` + `threshold` + `margin`,
- calibración separada por cámara o grupo de cámaras con geometría similar,
- no reutilizar umbrales de otro modelo aunque ambos sean InsightFace.

Valores iniciales orientativos (solo para bootstrap):

- `threshold_cosine`: `0.45 - 0.60` (calibrar con datos reales),
- `margin_top1_top2`: `0.05 - 0.12` (calibrar con galerías reales).

La regla final debe salir de validación offline (ROC/DET) y luego verificación en operación.

---

## 9. Control de pose y calidad

### 9.1 Problema real

La cámara estará algo elevada, por lo que no siempre habrá visión frontal.

### 9.2 Qué hacer

En vez de intentar reconocer cualquier frame:

- estimar pose aproximada,
- rechazar poses excesivas,
- esperar mejores frames del mismo track,
- usar mejores frames para reconocimiento.

### 9.3 Métricas sugeridas para scoring

Cada frame puede recibir un score con:

- tamaño del rostro,
- nitidez,
- iluminación,
- cercanía a pose aceptable,
- centrado,
- estabilidad temporal.

### 9.4 Estrategia recomendada

- opción A: mejor frame del buffer,
- opción B: top-3 frames y promediar embeddings,
- opción C: voto temporal por identidad sobre una ventana corta.

---

## 10. Detección de persona vs reconocimiento facial

El sistema no debe depender solo de ver una cara perfecta.

### 10.1 Comportamiento esperado

#### Caso 1 — miembro de la oficina

- persona detectada,
- cara detectada,
- match positivo,
- salida: saludo / evento con nombre.

#### Caso 2 — visitante / random

- persona detectada,
- cara detectada,
- sin match válido,
- salida: `unknown_person` o `visitor`.

#### Caso 3 — cuerpo visible, cara no usable

- persona detectada,
- cara no detectable o no usable,
- salida: `person_detected` sin identificación.

---

## 11. Anti-spoofing / liveness

### 11.1 MVP

Para el MVP puede omitirse, pero debe quedar previsto en arquitectura.

### 11.2 Producción

Si el sistema va a:

- abrir puertas,
- validar acceso físico,
- o registrar algo sensible,

se recomienda añadir liveness / anti-spoofing.

### 11.3 Opciones

- MiniFASNet / Silent Face Anti-Spoofing,
- reglas adicionales de consistencia temporal,
- si el riesgo es alto, combinar con otro factor (PIN, QR, tarjeta, NFC).

---

## 12. Componentes / módulos propuestos

### 12.1 `capture_service`

Responsable de:

- abrir cámara o stream,
- entregar frames,
- manejar reconexión,
- timestamping.

### 12.2 `person_detection_service`

Responsable de:

- detectar personas,
- producir bounding boxes,
- opcionalmente iniciar tracking.

### 12.3 `tracking_service`

Responsable de:

- asignar `track_id`,
- mantener continuidad entre frames,
- sostener buffers por persona.

### 12.4 `face_service`

Responsable de:

- detectar rostro dentro del ROI,
- alinear rostro,
- generar métricas de calidad,
- extraer embeddings.

### 12.5 `recognition_service`

Responsable de:

- consultar Faiss,
- recuperar top-k,
- aplicar reglas de decisión,
- devolver `known`, `unknown`, `ambiguous`.

### 12.6 `enrollment_service`

Responsable de:

- capturar muestras,
- validar calidad,
- generar embeddings,
- persistir datos,
- reconstruir índice Faiss.

### 12.7 `storage_service`

Responsable de:

- SQLite,
- snapshots,
- configuración,
- embeddings persistidos.

### 12.8 `api_service`

Responsable de:

- endpoints locales,
- dashboard,
- ABM de personas,
- consulta de eventos.

---

## 13. Diseño de datos sugerido

### Tabla `persons`

- `person_id`
- `code`
- `full_name`
- `status`
- `notes`
- `created_at`
- `updated_at`

### Tabla `face_samples`

- `sample_id`
- `person_id`
- `image_path`
- `capture_type` (`canonical` / `operational`)
- `camera_id`
- `quality_score`
- `pose_yaw`
- `pose_pitch`
- `pose_roll`
- `created_at`

### Tabla `face_embeddings`

- `embedding_id`
- `person_id`
- `sample_id`
- `model_name`
- `model_version`
- `embedding_dim`
- `embedding_vector`
- `norm`
- `is_normalized`
- `faiss_index_version`
- `created_at`

### Tabla `faiss_indexes`

- `index_version`
- `model_name`
- `model_version`
- `embedding_dim`
- `metric` (`cosine_ip`)
- `is_active`
- `created_at`

### Regla operativa de compatibilidad

No mezclar embeddings de distintos modelos/versiones en el mismo índice activo.
Si cambia modelo o dimensión, crear índice nuevo y recalibrar umbrales.

### Estrategia de reindexación sin downtime

Usar patrón `shadow index`:

1. construir índice nuevo en segundo plano,
2. validarlo con set de control,
3. hacer swap atómico de `is_active`,
4. mantener rollback al índice anterior por ventana corta.

### Tabla `recognition_events`

- `event_id`
- `camera_id`
- `track_id`
- `timestamp`
- `decision`
- `top1_person_id`
- `top1_score`
- `top2_person_id`
- `top2_score`
- `snapshot_path`
- `notes`

### Tabla `system_config`

- thresholds,
- márgenes,
- calidad mínima,
- parámetros de buffer,
- rutas de modelos.

---

## 14. Estructura de repo sugerida

```text
repo/
├─ apps/
│  ├─ api/
│  ├─ dashboard/
│  └─ cli/
├─ configs/
│  ├─ default.yaml
│  ├─ models.yaml
│  └─ cameras.yaml
├─ data/
│  ├─ db/
│  ├─ embeddings/
│  ├─ samples/
│  └─ snapshots/
├─ docs/
│  ├─ architecture.md
│  ├─ enrollment_protocol.md
│  ├─ calibration.md
│  └─ operations.md
├─ models/
│  ├─ insightface/
│  ├─ detection/
│  └─ anti_spoofing/
├─ scripts/
│  ├─ bootstrap_env.sh
│  ├─ download_models.py
│  ├─ rebuild_faiss_index.py
│  └─ run_demo.py
├─ src/
│  ├─ capture/
│  ├─ detection/
│  ├─ tracking/
│  ├─ face/
│  ├─ recognition/
│  ├─ enrollment/
│  ├─ storage/
│  ├─ api/
│  ├─ utils/
│  └─ settings/
├─ tests/
│  ├─ unit/
│  ├─ integration/
│  └─ fixtures/
├─ environment.yml
├─ pyproject.toml
└─ README.md
```

---

## 15. Entorno con Miniforge / Mamba

### 15.1 Propuesta de stack base

- Python 3.11
- numpy
- opencv
- faiss-cpu
- sqlalchemy
- fastapi
- uvicorn
- pydantic
- loguru
- insightface
- onnxruntime

### 15.2 Variante GPU

Si se dispone de NVIDIA:

- `onnxruntime-gpu`
- evaluar `faiss-gpu` según plataforma y compatibilidad.

### 15.3 Archivo `environment.yml` orientativo

```yaml
name: facelab
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.11
  - numpy
  - opencv
  - sqlalchemy
  - faiss-cpu
  - pip
  - pip:
      - insightface
      - onnxruntime
      - fastapi
      - uvicorn
      - pydantic
      - loguru
```

### 15.4 Nota

Si se usa GPU, probablemente convenga separar un `environment.gpu.yml` o documentar instalación GPU por aparte para no bloquear a quien desarrolle en CPU.

---

## 16. MVP recomendado

### Objetivo del MVP

Demostrar en una sola cámara local que el sistema puede:

- detectar cara,
- identificar personal enrolado,
- clasificar desconocidos,
- y registrar eventos localmente.

### Alcance MVP-A (recomendado para arrancar rápido)

1. Captura desde cámara USB o RTSP.
2. Detección/alineamiento de rostro (InsightFace).
3. Embeddings con InsightFace.
4. Índice Faiss 1:N.
5. SQLite + eventos.
6. Script de enrolamiento.
7. API local mínima.

### Alcance MVP-B (siguiente iteración)

1. Detección de persona (YOLO).
2. Tracking por `track_id`.
3. Buffer temporal y frame selection por track.
4. Dashboard operativo.

### Fuera de MVP

- control de acceso físico,
- liveness obligatorio,
- clustering avanzado,
- múltiples cámaras simultáneas complejas,
- monitoreo enterprise.

---

## 17. Fases del proyecto

### Fase 0 — prueba técnica

- validar cámara,
- validar latencia,
- validar InsightFace en CPU/GPU,
- validar Faiss,
- prueba mínima de embeddings.

### Fase 1 — enrolamiento

- formulario/app de alta,
- captura de muestras,
- extracción de mejores frames,
- guardado local,
- reconstrucción de índice.

### Fase 2 — reconocimiento en vivo

- detección/alineamiento facial,
- selección de mejor frame,
- reconocimiento 1:N,
- logging de eventos.

### Fase 2.5 — percepción de escena

- detección de persona,
- tracking,
- estados híbridos `person_detected` + `face_*`.

### Fase 3 — calibración

- ajustar thresholds,
- medir falsos positivos,
- medir falsos rechazos,
- afinar scoring de calidad.

### Fase 4 — endurecimiento

- anti-spoofing,
- watchdogs,
- reconexión de cámara,
- manejo de errores,
- retención de evidencias,
- backups.

### Fase 5 — operación

- documentación,
- procedimientos de alta/baja,
- mantenimiento de modelos,
- monitoreo básico,
- políticas de privacidad y seguridad.

---

## 18. Pruebas y métricas

### 18.1 Métricas mínimas

- tasa de identificación correcta,
- tasa de desconocidos correctamente clasificados,
- falsos positivos,
- falsos rechazos,
- latencia por frame / por decisión,
- porcentaje de frames rechazados por baja calidad.

### 18.1.1 SLO inicial de latencia (operativo)

Definir presupuesto desde el inicio para guiar decisiones:

- `p95 detección+embedding <= 120 ms` por frame usable,
- `p95 decisión 1:N <= 150 ms` extremo a extremo,
- en batch/stream, mantener throughput objetivo de cámara (por ejemplo `>= 15 FPS` procesados).

### 18.2 Escenarios de prueba

- frontal,
- semiperfil,
- cámara elevada,
- iluminación variable,
- lentes,
- varias personas simultáneamente,
- persona conocida vs visitante,
- persona pasando rápido.

### 18.3 Recomendación

Probar al menos:

- galería solo canónica,
- galería canónica + operativa,

para validar que el enrolamiento desde el ángulo real mejora el desempeño.

---

## 19. Consideraciones de seguridad y privacidad

### 19.1 Datos sensibles

La base biométrica debe tratarse como información sensible.

### 19.2 Medidas mínimas

- almacenamiento local restringido,
- cifrado del disco o del directorio de datos cuando sea posible,
- control de acceso a la app de administración,
- backups protegidos,
- eliminación controlada de usuarios dados de baja,
- logging de acciones administrativas.

### 19.3 Política operativa

Definir desde el inicio:

- quién puede enrolar personas,
- quién puede ver eventos,
- cuánto tiempo se guardan snapshots,
- cómo se elimina información biométrica.

---

## 20. Riesgos técnicos

### Riesgo 1 — cámara demasiado alta o mala iluminación

Mitigación:

- ajustar físicamente la cámara si es posible,
- reforzar enrolamiento operativo,
- usar quality gating.

### Riesgo 2 — demasiados falsos positivos

Mitigación:

- subir threshold,
- usar margen top1-top2,
- exigir consistencia temporal,
- revisar galería contaminada.

### Riesgo 3 — demasiados falsos rechazos

Mitigación:

- mejorar muestras enroladas,
- usar multi-frame fusion,
- revisar calidad de captura,
- calibrar mejor umbrales.

### Riesgo 4 — dependencia excesiva del rostro frontal

Mitigación:

- enrolamiento desde el ángulo real,
- selección de mejores frames,
- no reconocer en poses extremas.

### Riesgo 5 — spoofing con foto/pantalla

Mitigación:

- liveness,
- detección temporal,
- segundo factor si la criticidad es alta.

---

## 21. Roadmap de implementación para Codex / agente de código

### Paso 1

Crear estructura base del proyecto y configuración.

### Paso 2

Implementar captura de cámara y loop principal.

### Paso 3

Integrar detector de personas y tracking básico.

### Paso 4

Integrar InsightFace para rostro + embedding.

### Paso 5

Implementar modelo de datos en SQLite.

### Paso 6

Implementar índice Faiss y servicio de búsqueda.

### Paso 7

Implementar flujo de enrolamiento.

### Paso 8

Implementar scoring de calidad y selección de frame.

### Paso 9

Implementar reglas de decisión (`known`, `unknown`, `ambiguous`, `not_usable`).

### Paso 10

Exponer API local y dashboard mínimo.

### Paso 11

Agregar calibración y herramientas de evaluación.

### Paso 12

Preparar integración de anti-spoofing como módulo opcional.

---

## 22. Tareas sugeridas para backlog inicial

### Infra

- [ ] crear `environment.yml`
- [ ] definir `pyproject.toml`
- [ ] crear estructura de carpetas
- [ ] definir archivos de configuración

### Video

- [ ] lector de cámara USB
- [ ] lector RTSP
- [ ] reconexión automática
- [ ] limitación de FPS procesado

### Detección / tracking

- [ ] detector de personas
- [ ] tracker por `track_id`
- [ ] buffer temporal por track

### Cara / identidad

- [ ] detección de rostro
- [ ] alignment
- [ ] embedding
- [ ] score de calidad
- [ ] estimación básica de pose
- [ ] búsqueda top-k en Faiss

### Enrolamiento

- [ ] alta de persona
- [ ] captura de video corto
- [ ] extracción de mejores frames
- [ ] guardado de muestras
- [ ] reindexación

### Persistencia

- [ ] tablas SQLite
- [ ] snapshots
- [ ] logs
- [ ] export de eventos

### API / UI

- [ ] endpoint de alta
- [ ] endpoint de listado de personas
- [ ] endpoint de eventos
- [ ] dashboard mínimo

### Calidad / seguridad

- [ ] calibración de thresholds
- [ ] test con desconocidos
- [ ] módulo opcional anti-spoofing
- [ ] política de retención de datos

---

## 23. Decisiones recomendadas desde ya

1. **No** usar `face-recognition` como base principal.
2. **Sí** usar embeddings + Faiss.
3. **Sí** separar detección de persona de reconocimiento facial.
4. **Sí** enrolar desde el ángulo real de la cámara.
5. **Sí** usar frame selection / multi-frame fusion.
6. **No** depender exclusivamente de rostros frontales.
7. **No** poner face mesh como pieza central del reconocimiento.
8. **Sí** dejar anti-spoofing previsto desde la arquitectura.

---

## 24. Recomendación final

La arquitectura objetivo para este proyecto debería ser:

- **local-first**,
- modular,
- basada en embeddings,
- tolerante a poses no frontales,
- y preparada para endurecerse progresivamente.

### Stack final recomendado

- Miniforge / Mamba
- Python 3.11
- OpenCV
- Ultralytics YOLO
- InsightFace
- ONNX Runtime
- Faiss
- SQLite
- FastAPI
- módulo opcional de anti-spoofing

### Idea fuerza del sistema

> Detectar personas en general, identificar solo cuando haya evidencia facial suficiente, y clasificar el resto de manera segura como desconocido o no identificable.

---

## 25. Referencias técnicas consultadas

- InsightFace (framework de análisis y reconocimiento facial)
- OpenCV FaceDetectorYN / FaceRecognizerSF
- Faiss (similarity search de vectores densos)
- Ultralytics YOLO (detección y tracking)
- MediaPipe Face Landmarker (si se requiere geometría facial más rica)
- ONNX Runtime (CPU/GPU execution providers)
- MiniFASNet / Silent Face Anti-Spoofing
