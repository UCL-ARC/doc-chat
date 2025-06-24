# scripts/drop_all_tables.py

import asyncio
import logging

from src.doc_chat.database import engine, Base
from src.doc_chat.config import settings

log = logging.getLogger(settings.LOGGER_NAME)

print("Using database:", settings.DATABASE_URL)

async def drop_all():
    async with engine.begin() as conn:
        print("Dropping all tables...")
        await conn.run_sync(Base.metadata.drop_all)
        print("All tables dropped.")

if __name__ == "__main__":
    asyncio.run(drop_all())