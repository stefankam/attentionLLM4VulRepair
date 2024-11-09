<fix/>from flask import abort, request, g</fix>
from functools import wraps
import cx_Oracle
import re


<fix/>def execute_query(con, query, **kwargs):</fix>
    """ Execute a query and return corresponding data """
    # Execute query
    cur = con.cursor()
    <fix/>cur.execute(query, kwargs)</fix>

    # Return data with description
    return (extract_schema(cur.description), cur.fetchall())


<fix/>def generic_search(con, keywords, tables):
    """ Perform a search in the given tables for containment of given keywords """</fix>
    # List of tuples (table_name, schema, tuples)
    result = []
    table_names = get_table_names(con)
    for table in tables:
        <fix/># Make sure user didn't cheat with table names
        if table not in table_names:
            raise ValueError('Invalid table name')</fix>

        # Build conditions
        conditions = []
        <fix/>for col in get_column_names(con, table):
            conditions.append('{} LIKE \'%\'||:keywords||\'%\''.format(col))</fix>

        conditions = ' OR '.join(conditions)

        # Execute query
        query = 'SELECT * FROM {} WHERE {}'.format(table, conditions)
        <fix/>(schema, data) = execute_query(con, query, keywords=keywords)

        if len(data) > 0:
            result.append((table, schema, data))</fix>

    return result


def extract_schema(description):
    """ Extract column names from cursor description """
    names = []
    for col in description:
        names.append(col[0])

    return names


def ajax(f):
    """ Custom decoractor to restrict acces to AJAX calls """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_xhr:
            return abort(401)
        return f(*args, **kwargs)
    return decorated_function


def get_db(app):
    """ Connect to the database and return connection """
    if not hasattr(g, 'db'):
        dsn_tns = cx_Oracle.makedsn(app.config['DB_SERVER'],
                                    app.config['DB_PORT'],
                                    app.config['DB_SID'])

        g.db = cx_Oracle.connect(app.config['DB_USER'],
                                 app.config['DB_PWD'],
                                 dsn_tns)

    return g.db


def get_queries(app, context):
    """ Parse, cache and return predefined queries """
    if 'queries' not in context:
        with open(app.config['QUERIES_PATH'], 'r') as fd:
            sqlFile = fd.read()

        # all SQL commands (split on ';')
        sqlCommands = sqlFile.split(';')
        context['queries'] = {}
        for command in sqlCommands:
            command = re.sub(r'\s*--\s*|\s*\n\s*', ' ', command)
            query = command.split(':')
            context['queries'][query[0]] = query[1]

    return context['queries']


def get_table_names(con):
    """ Return database tables names """
    query = 'SELECT table_name FROM user_tables'
    data = execute_query(con, query)[1]
    return list(map(lambda x: x[0], data))


def get_column_names(con, table):
    """ Return table column names """
    query = 'SELECT * FROM {} WHERE 1=0'.format(table)
    return execute_query(con, query)[0]
