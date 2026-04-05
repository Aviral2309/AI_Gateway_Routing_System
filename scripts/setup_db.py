"""
Run once to initialize PostgreSQL.
Usage: python scripts/setup_db.py
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    print("=" * 55)
    print("  Nexus LLM Router — Database Setup")
    print("=" * 55)

    from config.settings import settings
    print(f"\nDatabase: {settings.database_url}\n")

    # Auto-create database if missing
    db_name = settings.database_url.split("/")[-1]
    url     = settings.database_url.replace("postgresql+asyncpg://", "")
    userpass, hostdb = url.split("@")
    user, password   = userpass.split(":", 1)
    hostport, _      = hostdb.rsplit("/", 1)
    parts            = hostport.split(":")
    host, port       = parts[0], (parts[1] if len(parts) > 1 else "5432")

    try:
        import asyncpg
        conn = await asyncpg.connect(host=host, port=int(port),
                                     user=user, password=password, database="postgres")
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname=$1", db_name)
        if not exists:
            await conn.execute(f'CREATE DATABASE "{db_name}"')
            print(f"✓ Created database '{db_name}'")
        else:
            print(f"✓ Database '{db_name}' already exists")
        await conn.close()
    except Exception as e:
        print(f"! Could not auto-create DB: {e}")
        print(f"  Run manually: CREATE DATABASE {db_name};")

    # Create tables
    print("\nCreating tables...")
    try:
        from app.db.session import init_db
        await init_db()
        print("✓ Tables ready: request_logs, session_budgets")
    except Exception as e:
        print(f"✗ Failed: {e}")
        sys.exit(1)

    # Verify
    print("\nVerifying...")
    try:
        from app.db.session import get_db
        from sqlalchemy import text
        async with get_db() as db:
            count = (await db.execute(text("SELECT COUNT(*) FROM request_logs"))).scalar()
        print(f"✓ Connected. request_logs has {count} rows.")
    except Exception as e:
        print(f"✗ {e}")
        sys.exit(1)

    print("\n" + "=" * 55)
    print("  Setup complete!")
    print("  Run: uvicorn app.main:app --reload")
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())
