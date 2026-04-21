# Pruebas y metricas

## Metricas clave

- Identificacion correcta (TP).
- Desconocidos bien clasificados.
- Falsos positivos (FAR).
- Falsos rechazos (FRR).
- p50/p95 de latencia por etapa.
- Porcentaje de frames no usables.

## Escenarios de prueba

- Frontal y semiperfil.
- Camara elevada.
- Luz variable.
- Con/sin lentes.
- Persona conocida vs visitante.
- Personas en movimiento.

## Protocolo de calibracion

1. Ejecutar baseline con galeria canonica.
2. Agregar galeria operativa.
3. Barrer threshold y margin.
4. Elegir punto de operacion (FAR/FRR).
5. Validar en entorno real durante varios dias.
