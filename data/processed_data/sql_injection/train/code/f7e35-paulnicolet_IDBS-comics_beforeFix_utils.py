<vul/>from flask import abort, request</vul>
from functools import wraps
import cx_Oracle
import re


<vul/>def get_db(app, context):
    """ Connect to the database and return connection """
    if not hasattr(context, 'db'):
        dsn_tns = cx_Oracle.makedsn(app.config['DB_SERVER'],
                                    app.config['DB_PORT'],
                                    app.config['DB_SID'])

        context.db = cx_Oracle.connect(app.config['DB_USER'],
                                       app.config['DB_PWD'],
                                       dsn_tns)

    return context.db


def get_queries(app, context):
    """ Parse and return predefined queries """
    if not hasattr(context, 'queries'):
        with open(app.config['QUERIES_PATH'], 'r') as fd:
            sqlFile = fd.read()

        # all SQL commands (split on ';')
        sqlCommands = sqlFile.split(';')
        context.queries = {}
        for command in sqlCommands:
            command = re.sub(r'\s*--\s*|\s*\n\s*', ' ', command)
            query = command.split(':')
            context.queries[query[0]] = query[1]

    return context.queries


def execute_query(app, context, query):</vul>
    """ Execute a query and return corresponding data """
    # Execute query
    con = get_db(app, context)
    cur = con.cursor()
    cur.execute(query)

    # Return data with description
    return (extract_schema(cur.description), cur.fetchall())


<vul/>def generic_search(keywords, tables, app, context):</vul>
    # List of tuples (table_name, schema, tuples)
    result = []
    for table in tables:
        <vul/># Get columns for the table
        query = 'SELECT * FROM {} WHERE 1=0'.format(table)
        description = execute_query(app, context, query)[0]</vul>

        # Build conditions
        conditions = []
        <vul/>for col in description:
            conditions.append('{} LIKE \'%{}%\''.format(col, keywords))</vul>

        conditions = ' OR '.join(conditions)

        # Execute query
        query = 'SELECT * FROM {} WHERE {}'.format(table, conditions)
        <vul/>(schema, data) = execute_query(app, context, query)
        result.append((table, schema, data))</vul>

    return result


def extract_schema(description):
    names = []
    for col in description:
        names.append(col[0])

    return names


def shutdown(app, context):
    """ Clean-up application state before shutdown """
    with app.app_context():
        get_db(app, context).close()


def ajax(f):
    """ Custom decoractor to restrict acces to AJAX calls """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_xhr:
            return abort(401)
        return f(*args, **kwargs)
    return decorated_function
