# Guia rapida de uso (camara)

## 1) Captura de datos (con puntos visibles)

Reemplaza `TU_PERSON_ID` por tu identificador (ejemplo: `rodrigo`).

```bash
/home/labia10/miniforge3/envs/facial-lab/bin/python scripts/run_camera_demo.py \
  --camera-index 0 \
  --camera-id cam-01 \
  --enroll-person-id TU_PERSON_ID \
  --guided-enroll \
  --guided-preset strict \
  --show-landmarks \
  --landmarks-max-points 50 \
  --landmarks-every 1
```

## 2) Reconocimiento con saludo (con puntos visibles)

```bash
/home/labia10/miniforge3/envs/facial-lab/bin/python scripts/run_camera_demo.py \
  --camera-index 0 \
  --camera-id cam-01 \
  --recognize-every 3 \
  --voice-greet \
  --voice-backend melotts \
  --show-landmarks \
  --landmarks-max-points 50 \
  --landmarks-every 1
```

## Controles

- `q`: salir
- `e`: enrolar manualmente el frame actual (si usas `--enroll-person-id`)
