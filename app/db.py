from contextlib import contextmanager

from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from .config import settings


def _configure(conn):
    register_vector(conn)


pool = ConnectionPool(
    settings.database_url,
    min_size=1,
    max_size=8,
    configure=_configure,
    open=False,
)


def open_pool():
    pool.open()
    pool.wait()


def close_pool():
    pool.close()


@contextmanager
def conn():
    with pool.connection() as c:
        yield c
