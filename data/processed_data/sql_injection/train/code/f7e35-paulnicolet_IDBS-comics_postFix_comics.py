<fix/>from flask import Flask, g, render_template, request, jsonify, abort
from utils import get_db, get_queries, ajax, execute_query, generic_search
import utils</fix>
import os
import atexit

app = Flask(__name__)
<fix/>context = {}</fix>

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
    <fix/>con = get_db(app)</fix>
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

    <fix/>try:
        data = generic_search(get_db(app), keywords, tables)
        return jsonify(data)
    except ValueError:
        # Invalid user input
        return abort(401)</fix>


@app.route('/queries', methods=['GET', 'POST'])
@ajax
def queries():
    if request.method == 'GET':
        <fix/>return render_template('queries-form.html', queries=get_queries(app, context))</fix>

    # Get query and execute it
    query_key = request.form['query-selector']
    <fix/>query = get_queries(app, context)[query_key]
    (schema, data) = execute_query(get_db(app), query)</fix>

    return jsonify([('', schema, data)])


@app.route('/get_table_names', methods=['GET'])
@ajax
def get_table_names():
    <fix/>return jsonify(utils.get_table_names(get_db(app)))


@app.teardown_appcontext
def close_db(error):
    """Closes the database again at the end of the request."""
    if hasattr(g, 'db'):
        g.db.close(</fix>)
