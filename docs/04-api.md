# API local

## Estado actual

- `GET /health`
- `POST /v1/faces/enroll`
- `POST /v1/faces/match`

## Endpoints objetivo (MVP -> MVP-B)

- `POST /v1/persons`
- `GET /v1/persons`
- `POST /v1/enrollments`
- `POST /v1/recognitions/match`
- `GET /v1/events`
- `POST /v1/index/rebuild`

## Reglas de contrato

- Responses tipadas con Pydantic.
- Versionado por prefijo (`/v1`).
- Errores consistentes (`code`, `message`, `details`).
- Idempotencia en altas por `code` de persona.
