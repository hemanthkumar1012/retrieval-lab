from contextlib import contextmanager

from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from .config import settings


def _configure(conn):
    register_vector(conn)


pool = ConnectionPool(
    settings.database_url,
    min_size=1,
    max_size=settings.db_pool_max,
    configure=_configure,
    # psycopg switches to server-side prepared statements after a few repeats of a
    # query. Supabase's transaction pooler (pgbouncer) hands each transaction to an
    # arbitrary backend, so those statements vanish and queries start failing.
    kwargs={"prepare_threshold": None},
    open=False,
)


def open_pool():
    pool.open()
    pool.wait()


def close_pool():
    pool.close()


@contextmanager
def conn():
    if pool.closed:
        pool.open()
    with pool.connection() as c:
        yield c
