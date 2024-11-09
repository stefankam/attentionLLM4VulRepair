import sqlite3
from sqlite3 import Error
import pandas as pd
import sys
import logging


logger = logging.getLogger(__name__)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(funcName)s - line %(lineno)d"
    )
)
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)

def create_connection(db_name='cert_db.sqlite3'):
    try:
        conn = sqlite3.connect(db_name)
        return conn
    except Error as e:
        logger.critical(e)
    return None

def dbtables_to_csv():
    with create_connection() as conn:
        table_names = conn.cursor().execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    table_names = [x[0] for x in table_names]
    <fix/>open_query = "SELECT * FROM ?"</fix>
    for table in table_names:
        with create_connection() as conn:
            <fix/>pd.read_sql(open_query, conn, params=[table]).to_csv(f'{table}.csv', index=False)</fix>

if __name__=='__main__':
    dbtables_to_csv()
