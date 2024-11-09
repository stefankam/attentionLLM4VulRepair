# -*- coding: utf-8 -*-
import pymysql
import db_config


class DBHelper:
    
    def connect(self, database="crimemap"):
        return pymysql.connect(host='localhost',
                    user=db_config.db_user,
                    password=db_config.db_password,
                    db=database)

    def get_all_inputs(self):
        connection = self.connect()
        try:
            query = "SELECT description FROM crimes;"
            with connection.cursor() as cursor:
                cursor.execute(query)
            return cursor.fetchall()
        finally:
            connection.close()

    def add_input(self, data):
        connection = self.connect()
        try:
            # protection from SQL Injections, previously use {}
            # The following introduces a deliberate security flaw
            <fix/>query = "INSERT INTO crimes (description) VALUES (%s);"</fix>
            with connection.cursor() as cursor:
                <fix/>cursor.execute(query, data)</fix>
                connection.commit()
        finally:
            connection.close()

    def clear_all(self):
        connection = self.connect()
        try:
            query = "DELETE FROM crimes;"
            with connection.cursor() as cursor:
                cursor.execute(query)
                connection.commit()
        finally:
            connection.close()
    



