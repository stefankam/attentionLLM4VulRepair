import sqlite3
from Client import Client
import create_database


class SqlManager:
    def __init__(self, conn):
        self.__conn = conn

    def change_message(self, new_message, logged_user):
        update_sql = <fix/>"""</fix>
            UPDATE Clients
            <fix/>SET message = ?
            WHERE client_id = ?
        """</fix>

        cursor = self.__conn.cursor()

        <fix/>cursor.execute(update_sql, (new_message, logged_user.get_client_id()))</fix>
        self.__conn.commit()
        logged_user.set_message(new_message)

    def change_pass(self, new_pass, logged_user):
        update_sql = """
            UPDATE Clients
            <fix/>SET password = ?
            WHERE client_id = ?
        """</fix>

        cursor = self.__conn.cursor()

        <fix/>cursor.execute(update_sql, (new_pass, logged_user.get_client_id()))</fix>
        self.__conn.commit()

    def register(self, username, password):
        insert_sql = """
            INSERT INTO Clients (username, password)
            <fix/>VALUES (?, ?)
        """
        # try:

        # except CannotUseThis:
        #     raise</fix>

# Da ne pravi registraciq, ako imeto ve4e e zaeto!!!

        cursor = self.__conn.cursor()

        <fix/>cursor.execute(insert_sql, (username, password))</fix>
        self.__conn.commit()

    def login(self, username, password):
        select_query = """
            SELECT client_id, username, balance, message
            FROM Clients
            <fix/>WHERE username = ? AND password = ?</fix>
            LIMIT 1
        """

        cursor = self.__conn.cursor()

        <fix/>cursor.execute(select_query, (username, password))</fix>
        user = cursor.fetchone()

        if(user):
            return Client(user[0], user[1], user[2], user[3])
        else:
            return False


class CannotUseThis(Exception):
    pass
