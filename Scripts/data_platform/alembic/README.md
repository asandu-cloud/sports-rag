# Alembic migrations

Run migrations against whatever `DATABASE_URL` is configured:

```bash
cd Scripts/data_platform
alembic upgrade head
```

Generate a new revision after editing models:

```bash
alembic revision --autogenerate -m "Add xyz"
```

Downgrade one step:

```bash
alembic downgrade -1
```

Alembic reads its URL from `data_platform.config.SETTINGS`, so changing
`DATABASE_URL` in the environment switches targets without touching
`alembic.ini`.
