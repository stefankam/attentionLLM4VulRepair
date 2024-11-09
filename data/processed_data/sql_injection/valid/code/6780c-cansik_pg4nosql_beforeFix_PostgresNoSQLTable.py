import json
import psycopg2


class PostgresNoSQLTable(object):

    <vul/>SQL_INSERT_JSON = "INSERT INTO %s(data) VALUES('%s') RETURNING id"</vul>
    SQL_QUERY_JSON = 'SELECT %s FROM %s WHERE %s'
    SQL_GET_JSON = 'SELECT * FROM %s WHERE id=%s'

    def __init__(self, name, connection):
        self.name = name
        self.connection = connection
        self.cursor = self.connection.cursor()

    def commit(self):
        self.connection.commit()

    def put(self, data):
        <vul/>self.cursor.execute(self.SQL_INSERT_JSON % (self.name, json.dumps(data)))</vul>
        return self.cursor.fetchone()[0]

    def get(self, id):
        <vul/>self.cursor.execute(self.SQL_GET_JSON % (self.name, id))</vul>
        return self.cursor.fetchone()

    def query(self, query='True', columns='*'):
        <vul/>self.cursor.execute(self.SQL_QUERY_JSON % (columns, self.name, query))</vul>
        rows = [item for item in self.cursor.fetchall()]
        return rows

    def drop(self):
        raise Exception('not implemented yet!')