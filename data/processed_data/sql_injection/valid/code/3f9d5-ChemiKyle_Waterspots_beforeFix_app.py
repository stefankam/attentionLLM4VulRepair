
__author__      = "Kyle Chesney"

from flask import *
from flask_login import LoginManager
# TODO: Research SQLAlchemy vs raw SQL
#from flask_sqlalchemy import SQLAlchemy
# TODO: MariaDB vs PostgreSQL, re: audit trail
import sqlite3
from datetime import datetime as dt

app = Flask(__name__)
login = LoginManager(app)
#db = SQLAlchemy()

@app.route('/')
def home():
#    if current_user.is_authenticated:
    return render_template('login.html')
#    if not session.get('logged_in'):
#        return render_template('login.html')


# TODO: separate POST requests to allow entry of params after logging in
@app.route('/', methods=['POST'])
def login():
    print('login')
    user = str(request.form['username'])
    password = str(request.form['password'])
    cur.execute('SELECT * FROM users WHERE name = \'{}\' AND password = \'{}\';'.format(user, password))
    response = cur.fetchone()
    if response != None:
        print(response, 'OK')
        return redirect(url_for('enter_test_point'))
    else:
        print(response, 'not OK')
        flash('Invalid login or password')
        return render_template('login.html')

@app.route('/entry_type')
def enter_log():
    # conn = sqlite3.connect
    methods = ['Metals', 'Organics']
    return render_template('entry.html', methods = methods)

#@app.route('/entry', methods=['POST', 'GET'])
#def render_entry():
#    print('only here')
#    return render_template('make_entry.html')

@app.route('/entry', methods=['POST', 'GET'])
def enter_test_point():
    if request.method == 'POST':
        dict = {}
        dict['study'] = 'test'
        dict['timestamp'] = dt.now()
        print(request.form)
        for item, val in request.form.items():
            dict[item] =  val
        print(dict)
        sql = """INSERT INTO observations (study, pH, TDS, Turbidity, Temperature, timestamp)
        VALUES(?,?,?,?,?,?)"""
        cur.execute(sql, tuple(dict[k] for k in dict.keys()))
        db.commit()
    return render_template('make_entry.html', parameters = ['pH', 'TDS', 'Turbidity', 'Temperature'])


if __name__ == "__main__":
    app.secret_key = 'bottom text'
    db = sqlite3.connect('db/test.db', check_same_thread=False)
    cur = db.cursor()
    app.run(debug=True, host = '0.0.0.0')
