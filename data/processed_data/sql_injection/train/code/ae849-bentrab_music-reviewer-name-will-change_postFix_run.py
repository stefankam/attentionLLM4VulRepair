import os

import configparser
from flask_bcrypt import Bcrypt
from flask import Flask, render_template, request, flash, session, redirect, url_for
from forms import Last_FM_Form
import mysql.connector

# Read configuration from file.
config = configparser.ConfigParser()
config.read('config.ini')

# Set up application server.
app = Flask(__name__)
bcrypt = Bcrypt(app)
app.secret_key = "adbi327fds"

# Create a function for fetching data from the database.
def sql_query(sql):
    db = mysql.connector.connect(**config['mysql.connector'])
    cursor = db.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()
    cursor.close()
    db.close()
    return result


def sql_execute(sql):
    db = mysql.connector.connect(**config['mysql.connector'])
    cursor = db.cursor()
    cursor.execute(sql)
    db.commit()
    cursor.close()
    db.close()

# For this example you can select a handler function by
# uncommenting one of the @app.route decorators.

#@app.route('/')
def basic_response():
    return "It works!" #example

# This route involves some LOGIN stuff, not implemented yet	
#@app.route('/login', methods = ['GET', 'POST'])
<fix/>ddef login():
    if 'user' in session:
        return redirect(url_for('dashboard'))

    message = None

    if request.method == "POST":
        usern = request.form.get("username")
        passw = request.form.get("password").encode('utf-8')
        result = db.execute("SELECT * FROM user WHERE username = :u", {"u": usern}).fetchone()

        if result is not None:
            print(result['password'])
            if bcrypt.check_password_hash(result['password'], passw) is True:
                session['user'] = usern
                return redirect(url_for('dashboard'))

        message = "Username or password is incorrect."
    return render_template("login.html", message=message)
</fix>   

# route for account registartion
@app.route("/register", methods=["GET", "POST"])
def register():
    if 'user' in session:
        return redirect(url_for('dashboard'))

    message = None

    if request.method == "POST":
        try: 
            usern = request.form.get("username")
            passw = request.form.get("password")
            passw_hash = bcrypt.generate_password_hash(passw).decode('utf-8')

            <fix/>result = db.execute("INSERT INTO user (username, password) VALUES (:u, :p)", {"u": usern, "p": passw_hash})</fix>
            db.commit()

            if result.rowcount > 0:
                session['user'] = usern
                return redirect(url_for('dashboard'))

        except exc.IntegrityError:
            message = "Username already exists."
            db.execute("ROLLBACK")
            db.commit()

    return render_template("registration.html", message=message)

#route for logout
@app.route("/logout")
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))
   
# Home page after login
@app.route('/home/', methods=['GET', 'POST'])	
@app.route('/home/<username>', methods=['GET', 'POST'])
def home(username = None):
	lastFM = Last_FM_Form(request.form)
	if request.method == 'POST':
		return lastFM_results(lastFM, username = username)
	return render_template('home.html', username = username, form = lastFM)

# Gets search results	
@app.route('/results')
def lastFM_results(lastFM, username):
	results = []
	search_string = lastFM.data['search']
	if lastFM.data['search'] == '':
		#result = 
		#results = result.all()
		result = []
	
	if not results:
		flash('No results could be found for your search, please try again.')
		return redirect('/home/%s' % username)
	else:
		return render_template(lastFM_results.html, results = results)

# Given code from teacher's example, not used yet
#@app.route('/', methods=['GET', 'POST'])
def template_response_with_data():
    print(request.form)
    if "buy-book" in request.form:
        book_id = int(request.form["buy-book"])
        sql = "delete from book where id={book_id}".format(book_id=book_id)
        sql_execute(sql)
    template_data = {}
    sql = "select id, title from book order by title"
    books = sql_query(sql)
    template_data['books'] = books
    return render_template('home-w-data.html', template_data=template_data)

if __name__ == '__main__':
    app.run(**config['app'])
