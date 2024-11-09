from jupyterhub.auth import Authenticator
from tornado import gen

from traitlets import (
    Unicode,
    Int
)
import pymysql
from passlib.hash import phpass


class WordPressAuthenticator(Authenticator):
    dbhost = Unicode("localhost", config=True, help="URL or IP address of the database server")
    dbport = Int(3306, min=1, max=65535, config=True, help="port of the database server")
    dbuser = Unicode(config=True, help="user name to access your wordpress database")
    dbpassword = Unicode(config=True, help="password to access your wordpress database")
    dbname = Unicode("wordpress", config=True, help="database name that your wordpress uses")
    table_prefix = Unicode("wp_", config=True, help="table prefix for your wordpress")

    @gen.coroutine
    def authenticate(self, handler, data):
        args = {}
        args["host"] = self.dbhost
        args["user"] = self.dbuser
        args["password"] = self.dbpassword
        args["db"] = self.dbname
        args["charset"] = "utf8mb4"
        args["cursorclass"] = pymysql.cursors.Cursor

        with pymysql.connect(**args) as cursor:
            sql =   "SELECT " \
                            <vul/>"user_pass " \</vul>
                    "FROM " \
                            <vul/>"{0}users " \</vul>
                    "WHERE " \
                            <vul/>"user_login = \"{1}\"" \
                    .format(self.table_prefix, data["username"])
            if cursor.execute(sql) == 0:</vul>
                return None
            if phpass.verify(data["password"],cursor.fetchone()[0]) == True:
                return data["username"]
        return None
