import sqlite3
from Client import Client
import create_database


class SqlManager:
    def __init__(self, conn):
        self.__conn = conn

    def change_message(self, new_message, logged_user):
        update_sql = """
            UPDATE Clients
            <vul/>SET message = '{}'
            WHERE client_id = '{}'
        """.format(new_message, logged_user.get_client_id())</vul>

        cursor = self.__conn.cursor()

        <vul/><vul/>cursor.execute(update_sql)</vul></vul>
        self.__conn.commit()
        logged_user.set_message(new_message)

    def change_pass(self, new_pass, logged_user):
        update_sql = """
            UPDATE Clients
            <vul/>SET password = '{}'
            WHERE client_id = '{}'
        """.format(new_pass, logged_user.get_client_id())</vul>

        cursor = self.__conn.cursor()

        cursor.execute(update_sql)
        self.__conn.commit()

    def register(self, username, password):
        insert_sql = """
            INSERT INTO Clients (username, password)
            <vul/>VALUES ('{}', '{}')
        <vul/>""".format(username, password)</vul></vul>

# Da ne pravi registraciq, ako imeto ve4e e zaeto!!!

        cursor = self.__conn.cursor()

        <vul/>cursor.execute(insert_sql)</vul>
        self.__conn.commit()

    def login(self, username, password):
        select_query = """
            SELECT client_id, username, balance, message
            FROM Clients
            <vul/>WHERE username = '{}' AND password = '{}'</vul>
            LIMIT 1
        """.format(username, password)

        cursor = self.__conn.cursor()

        <vul/>cursor.execute(select_query)</vul>
        user = cursor.fetchone()

        if(user):
            return Client(user[0], user[1], user[2], user[3])
        else:
            return False
