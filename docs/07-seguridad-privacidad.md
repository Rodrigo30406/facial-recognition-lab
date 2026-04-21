# Seguridad y privacidad

## Datos sensibles

Los embeddings y snapshots son datos biometricos sensibles.

## Medidas minimas

- Directorio de datos con permisos restringidos.
- Cifrado de disco o volumen cuando sea posible.
- Backups protegidos.
- Auditoria de acciones administrativas.
- Politica de retencion y borrado.

## Politica operativa

- Quien puede enrolar.
- Quien puede ver eventos/snapshots.
- Tiempo de retencion por tipo de dato.
- Procedimiento de baja y eliminacion segura.

## Seguridad por fases

- MVP: controles basicos + hardening de acceso.
- Produccion: agregar liveness y/o segundo factor para casos de alto riesgo.
