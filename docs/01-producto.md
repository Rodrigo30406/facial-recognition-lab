# Producto y alcance

## Objetivo

Construir un sistema local-first de reconocimiento facial 1:N para oficina/lab que:

- detecte presencia de personas,
- identifique personas enroladas cuando exista evidencia facial usable,
- clasifique correctamente desconocidos,
- degrade de forma segura a estados no identificables.

## Principios

- Offline/local-first.
- Modularidad por componentes.
- Escalado incremental sin reentrenamiento global.
- Identificacion conservadora (mejor "unknown" que falso positivo).

## Casos de uso

- Control de presencia interno.
- Registro de eventos de entrada/salida.
- Alertas de desconocidos en zona supervisada.

## Fuera de alcance inicial

- Apertura automatica de puertas en modo autonomo.
- Liveness obligatorio en MVP.
- Multi-sede/multi-camara enterprise.
