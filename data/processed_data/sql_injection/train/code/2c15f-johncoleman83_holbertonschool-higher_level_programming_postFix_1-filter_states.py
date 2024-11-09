#!/usr/bin/python3
"""
1-filter_states.py
"""
import MySQLdb
import sys


def init_db():
    """initializes a db with MySQLdb"""
    db = MySQLdb.connect(host='localhost',
                         port=3306,
                         user=sys.argv[1],
                         passwd=sys.argv[2],
                         db=sys.argv[3])
    return db


def print_states_N(db):
    """prints all states with first initial 'N' from input DB"""
    cur = db.cursor()
    cur.execute("SELECT * FROM states "
                "WHERE name LIKE BINARY 'N%' "
                "ORDER BY states.id ASC")
    <fix/>for row in cur.fetchall():</fix>
        print(row)
    cur.close()
    db.close()


if __name__ == "__main__":
    print_states_N(init_db())
