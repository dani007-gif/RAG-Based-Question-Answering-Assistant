-- scripts/init_db.sql
-- Runs automatically when the PostgreSQL container starts for the first time.
-- Enables the pgvector extension so LangChain can create vector columns.

CREATE EXTENSION IF NOT EXISTS vector;

-- Verify the extension is active
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
    RAISE NOTICE 'pgvector extension is ready.';
  END IF;
END $$;
