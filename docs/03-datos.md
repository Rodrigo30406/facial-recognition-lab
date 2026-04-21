# Modelo de datos

## Tablas principales

### `persons`

- `person_id` (PK)
- `code`
- `full_name`
- `status`
- `notes`
- `created_at`
- `updated_at`

### `face_samples`

- `sample_id` (PK)
- `person_id` (FK)
- `image_path`
- `capture_type` (`canonical|operational`)
- `camera_id`
- `quality_score`
- `pose_yaw`
- `pose_pitch`
- `pose_roll`
- `created_at`

### `face_embeddings`

- `embedding_id` (PK)
- `person_id` (FK)
- `sample_id` (FK)
- `model_name`
- `model_version`
- `embedding_dim`
- `embedding_vector`
- `norm`
- `is_normalized`
- `faiss_index_version`
- `created_at`

### `faiss_indexes`

- `index_version` (PK)
- `model_name`
- `model_version`
- `embedding_dim`
- `metric` (`cosine_ip`)
- `is_active`
- `created_at`

### `recognition_events`

- `event_id` (PK)
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

## Reglas de integridad

- No mezclar embeddings de diferentes `model_version` en un indice activo.
- Reindexacion con estrategia shadow index + swap atomico.
- Registrar version de indice en cada embedding y evento para trazabilidad.
