<vul/>from flask import Flask, g, render_template, request, jsonify
from utils import get_db, get_queries, shutdown, ajax, execute_query, generic_search</vul>
import os
import atexit

app = Flask(__name__)

# Register clean up function
atexit.register(shutdown, app=app, context=g)

# Set app configuration
app.config.update({'DB_USER': os.environ['IDBS_USER'],
                   'DB_PWD': os.environ['IDBS_PWD'],
                   'DB_SERVER': 'diassrv2.epfl.ch',
                   'DB_PORT': 1521,
                   'DB_SID': 'orcldias',
                   'DEBUG': True,
                   'QUERIES_PATH': 'queries.sql'})


@app.route('/')
def home():
    <vul/>con = get_db(app, g)</vul>
    return render_template('index.html')


@app.route('/search', methods=['GET', 'POST'])
@ajax
def search():
    # If GET, return the form to render
    if request.method == 'GET':
        return render_template('search-form.html')

    # If POST, process the query and return data
    keywords = request.form['keywords']
    tables = list(request.form.keys())
    tables.remove('keywords')

    <vul/>data = generic_search(keywords, tables, app, g)
    return jsonify(data)</vul>


@app.route('/queries', methods=['GET', 'POST'])
@ajax
def queries():
    if request.method == 'GET':
        <vul/>return render_template('queries-form.html', queries=get_queries(app, g))</vul>

    # Get query and execute it
    query_key = request.form['query-selector']
    <vul/>query = get_queries(app, g)[query_key]
    (schema, data) = execute_query(app, g, query)</vul>

    return jsonify([('', schema, data)])


@app.route('/get_table_names', methods=['GET'])
@ajax
def get_table_names():
    <vul/>query = 'SELECT table_name FROM user_tables'
    data = execute_query(app, g, query)[1]
    return jsonify(data)</vul>
