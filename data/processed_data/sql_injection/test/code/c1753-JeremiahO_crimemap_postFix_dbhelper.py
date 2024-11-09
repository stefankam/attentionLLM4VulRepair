import sys
# sys.path.append("/root/.local/lib/python2.7/site-packages")

import pymysql
import pymysql.cursors
import dbconfig

# The four main database operations CRUD - Create. Read. Update. Delete


class DBHelper:

    #   --- CREATE --- Create and Insert New Data

    def connects(self, database="crimemap"):
        try:
            conn = pymysql.connect(host='localhost',
                                   user=dbconfig.db_user,
                                   password=dbconfig.db_password,
                                   db=database,
                                   charset='utf8mb4',
                                   cursorclass=pymysql.cursors.DictCursor)
        except Exception as e:
            print(e)
        return conn
    # --- READ --- Read Exsiting Data

    def get_all_inputs(self):
        connection = self.connects()
        try:
            query = "SELECT description FROM crimes;"
            with connection.cursor() as cursor:
                cursor.execute(query)
            return cursor.fetchall()
        finally:
            connection.close()

    # --- UPDATE --- Modify Existing Data

    def add_input(self, data):
        connection = self.connects()
        try:
            # The following introduces a deliberate security flaw. See section on SQL injecton below
            <fix/>query = "INSERT INTO crimes (description) VALUES (%s);"</fix>
            with connection.cursor() as cursor:
                <fix/>cursor.execute(query, data)</fix>
                connection.commit()
        finally:
            connection.close()

    # --- DELETE --- Delete Exsising Data

    def clear_all(self):
        connection = self.connects()
        try:
            query = "DELETE FROM crimes;"
            with connection.cursor() as cursor:
                cursor.execute(query)
                connection.commit()
        finally:
            connection.close()
