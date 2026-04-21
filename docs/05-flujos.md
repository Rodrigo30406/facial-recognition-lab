# Flujos funcionales

## Flujo de enrolamiento

1. Alta de persona.
2. Captura de video corto desde posicion real.
3. Seleccion de mejores frames por calidad.
4. Alineamiento facial y embedding.
5. Persistencia de muestra + embedding + metadatos.
6. Reindexacion (shadow index) y activacion.

## Flujo de reconocimiento

1. Captura frame.
2. Deteccion/alineamiento facial.
3. Quality/pose gating.
4. Embedding (normalizado).
5. Busqueda top-k en Faiss.
6. Decision por regla:
- `top1 >= threshold`
- `top1-top2 >= margin`
- consistencia temporal

## Estados de salida

- `person_detected`
- `face_detected`
- `face_not_usable`
- `known_person`
- `unknown_person`
- `ambiguous_match`
