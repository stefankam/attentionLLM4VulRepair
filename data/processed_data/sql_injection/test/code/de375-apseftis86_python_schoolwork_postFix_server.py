from flask import Flask, render_template, redirect, request
from mysqlconnection import connectToMySQL  # import the function that will return an instance of a connection

app = Flask(__name__)
<fix/>database = "test_users_db"</fix>
@app.route('/users')
def get_users():
    <fix/>mysql = connectToMySQL(database)</fix>
    users = mysql.query_db("SELECT * FROM users;")
    return render_template('index.html', users=users)

@app.route('/users/<id>')
def show_user(id):
    <fix/>mysql = connectToMySQL(database)
    data = {
        'userid': id
    }
    query = "SELECT * FROM users WHERE id = %(userid)s;"
    user = mysql.query_db(query, data)</fix>
    return render_template('user.html', user=user[0])

@app.route('/users/new', methods=['GET', 'POST'])
def add_user():
    if request.method == 'POST':
        <fix/>mysql = connectToMySQL(database)
        query = "INSERT INTO users (first_name, last_name, email, description, created_at)" \
                "  VALUES (%(fn)s, %(ln)s, %(e)s, %(d)s, now());"</fix>
        data = {
            "fn": request.form['first_name'],
            "ln": request.form['last_name'],
            "e": request.form['email'],
            "d": request.form['description'],
        }
        new_user = mysql.query_db(query, data)
        <fix/>return redirect('/users/{}'.format(new_user))</fix>
    else:
        return render_template('edit.html', user=None)


@app.route('/users/<id>/edit', methods=['GET', 'POST'])
def edit_user(id):
    <fix/>mysql = connectToMySQL(database)
    if request.method == 'POST':
        data = {
            "user_id": id,
            "fn": request.form['first_name'],
            "ln": request.form['last_name'],
            "e": request.form['email'],
            "d": request.form['description'],
        }
        query = "UPDATE users SET first_name = %(fn)s, last_name = %(ln)s, email= %(e)s," \
                " description = %(d)s, updated_at = now() WHERE id = %(user_id)s;"
        mysql.query_db(query, data)
        return redirect('/users/{}'.format(id))
    else:
        data = {
            "user_id": id,
        }
        query = "SELECT * FROM users WHERE id =  %(user_id)s;"
        user = mysql.query_db(query, data)
        return render_template('edit.html', user=user[0])</fix>


@app.route('/users/<id>/destroy')
def delete_user(id):
    <fix/>data = {
        'userid': id
    }
    mysql = connectToMySQL(database)
    query = "DELETE from users WHERE id =  =  %(user_id)s;"
    mysql.query_db(query, data)
    return redirect('/users')</fix>


if __name__ == "__main__":
    app.run(debug=True)
