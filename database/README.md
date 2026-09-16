# Sentinel Database (PostgreSQL)

This directory contains the database schema definitions, migrations, and setup guidelines for Sentinel.

## Database Role in MVP

Sentinel relies on PostgreSQL to store:
- Ingested video metadata (filename, duration, resolution, upload timestamp, storage location).
- Extracted temporal events (timestamp, detected objects, bounding boxes, labels, confidence scores).
- Evidence clips (start/end timestamps, storage path, investigator annotations).
- Natural language investigation queries and responses.
- Audit logs and generated incident reports.

## Local Setup Instructions (Future Scope)

When database integration is activated, a PostgreSQL instance can be configured via Docker or a local server:

```bash
# Example Docker command for local development
docker run --name sentinel-postgres \
  -e POSTGRES_DB=sentinel_db \
  -e POSTGRES_USER=sentinel_user \
  -e POSTGRES_PASSWORD=sentinel_pass \
  -p 5432:5432 -d postgres:15-alpine
```

Execute `schema.sql` to initialize the tables:

```bash
psql -h localhost -U sentinel_user -d sentinel_db -f schema.sql
```

## Structure

```
database/
├── README.md        # Database documentation and setup instructions
├── schema.sql       # PostgreSQL initial relational schema draft
└── migrations/      # Future versioned migration scripts (Alembic / SQL)
```
