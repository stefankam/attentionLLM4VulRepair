import pyxl
import mysql.connector

import database

from pypika import MySQLQuery, Table, Field

<fix/>def write_preset(query_text, query_desc):
    # to use this method you must pass in a connection,
    # a preset query, and a description of what the query achieves
    # it will automatically write it to the bottom of the table

    conn = database.get_db_connection()
    conn.autocommit = True

    extable = Table('Presets')
    q = MySQLQuery.into(extable).columns("querval", "description").insert(query_text, query_desc)
    quer = str(q)

    cursor.execute(quer)

    conn.close()

def edit_preset(query_id, query_text, description):
    '''(int, str, str) -> None'''
    # to use this method you must pass in a connection,
    # the id of a preset query,
    # a preset query, and a description of what the query achieves
    # it will update the query and description at the given id with new values.
    # if queryin or descriptin = "NA" then it will not update the values written so

    conn = database.get_db_connection()
    conn.autocommit = False

    cursor = conn.cursor()
    # quer = "ALTER TABLE Presets DROP COLUMN id"
    # cursor.execute(quer)
    # quer = "ALTER TABLE Presets ADD COLUMN id INT AUTO_INCREMENT PRIMARY KEY NOT NULL FIRST"
    # cursor.execute(quer)

    quer = "UPDATE Presets SET querval=%s AND description=%s WHERE id=%s"
    cursor.execute(quer, (query_text, description, query_id))

    # quer = "ALTER TABLE Presets DROP COLUMN id"
    # cursor.execute(quer)
    # quer = "ALTER TABLE Presets ADD COLUMN id INT AUTO_INCREMENT PRIMARY KEY NOT NULL FIRST"
    # cursor.execute(quer)

    conn.close()</fix>

def remove_preset(conn, key):
    <fix/># to use this method you must pass in a connection, and
    # what number the preset's id is
    cursor = conn.cursor()
    quer = "ALTER TABLE Presets DROP COLUMN id"
    cursor.execute(quer)
    quer = "ALTER TABLE Presets ADD COLUMN id INT AUTO_INCREMENT PRIMARY KEY NOT NULL FIRST"
    cursor.execute(quer)

    quer = "DELETE FROM Presets WHERE id = %s"
    cursor.execute(quer, (key,))

    quer = "ALTER TABLE Presets DROP COLUMN id"
    cursor.execute(quer)
    quer = "ALTER TABLE Presets ADD COLUMN id INT AUTO_INCREMENT PRIMARY KEY NOT NULL FIRST"
    cursor.execute(quer)</fix>
