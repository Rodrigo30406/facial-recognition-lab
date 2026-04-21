# Plan de sprints

Duracion sugerida: 2 semanas por sprint.

## Sprint 0 - Setup y baseline

Objetivo: dejar base ejecutable y medible.

- [ ] Entorno reproducible (`requirements`, `pyproject`, scripts).
- [ ] API base corriendo.
- [ ] Benchmark GPU/InsightFace documentado.
- [ ] Dataset minimo de prueba local.

Criterio de salida:

- Pipeline facial minimo corriendo local.
- Medicion inicial de latencia p95.

## Sprint 1 - Enrolamiento

Objetivo: alta completa de personas y muestras.

- [ ] Modelo SQLAlchemy para `persons`, `face_samples`, `face_embeddings`.
- [ ] Endpoint/CLI de alta de persona.
- [ ] Captura y seleccion de mejores frames.
- [ ] Persistencia de embeddings normalizados.

Criterio de salida:

- Al menos 20 personas enroladas con 10+ muestras utiles cada una.

## Sprint 2 - Reconocimiento 1:N

Objetivo: match robusto con reglas de decision.

- [ ] Servicio Faiss (top-k) con `IndexFlatIP`.
- [ ] Regla `threshold + margin + consistencia temporal`.
- [ ] Estados `known/unknown/ambiguous/face_not_usable`.
- [ ] Registro de eventos de reconocimiento.

Criterio de salida:

- Reconocimiento en vivo estable en una camara.
- Dashboard/logs de decisiones.

## Sprint 3 - Persona + tracking (MVP-B)

Objetivo: robustecer percepcion de escena.

- [ ] Integracion YOLO para clase `person`.
- [ ] Tracking por `track_id`.
- [ ] Buffer por track + seleccion de frame.
- [ ] Estado `person_detected` separado de identidad.

Criterio de salida:

- Sistema opera aunque no haya cara usable en todos los frames.

## Sprint 4 - Calibracion y hardening

Objetivo: preparar operacion continua.

- [ ] Calibracion formal de threshold/margin por camara.
- [ ] Reindexacion con shadow index + swap atomico.
- [ ] Alertas basicas por degradacion de latencia/calidad.
- [ ] Politicas de seguridad/retencion activas.

Criterio de salida:

- Objetivos p95 cumplidos en condiciones reales.
- Procedimientos operativos documentados.

## Sprint 5 - Opcionales de produccion

- [ ] Modulo anti-spoofing.
- [ ] Segundo factor para decisiones criticas.
- [ ] Multi-camara y consolidacion de eventos.
