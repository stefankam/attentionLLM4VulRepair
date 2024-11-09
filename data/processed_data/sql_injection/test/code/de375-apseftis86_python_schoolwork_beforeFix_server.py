<vul/></vul>from flask import Flask, render_template, redirect, request
from mysqlconnection import connectToMySQL  # import the function that will return an instance of a connection

app = Flask(__name__)

@app.route('/users')
def get_users():
    <vul/>mysql = connectToMySQL("users_db")</vul>
    users = mysql.query_db("SELECT * FROM users;")
    return render_template('index.html', users=users)

@app.route('/users/<id>')
def show_user(id):
    <vul/>mysql = connectToMySQL("users_db")
    user = mysql.query_db("SELECT * FROM users WHERE id = {};".format(id))</vul>
    return render_template('user.html', user=user[0])

@app.route('/users/new', methods=['GET', 'POST'])
def add_user():
    if request.method == 'POST':
        <vul/>mysql = connectToMySQL("users_db")
        query = "INSERT INTO users (first_name, last_name, email, description, created_at) VALUES (%(fn)s, %(ln)s, %(e)s, %(d)s, now());"</vul>
        data = {
            "fn": request.form['first_name'],
            "ln": request.form['last_name'],
            "e": request.form['email'],
            "d": request.form['description'],
        }
        new_user = mysql.query_db(query, data)
        <vul/>return redirect(f'/users/{new_user}')</vul>
    else:
        return render_template('edit.html', user=None)

@app.route('/users/<id>/edit', methods=['GET', 'POST'])
def edit_user(id):
   <vul/>user_id = id
   mysql = connectToMySQL("users_db")
   if request.method == 'POST':
       query = "UPDATE users SET first_name = %(fn)s, last_name = %(ln)s, email= %(e)s, description = %(d)s, updated_at = now() WHERE id = " + user_id + ";"
       data = {
           "fn" : request.form['first_name'],
           "ln" : request.form['last_name'],
           "e" : request.form['email'],
           "d" : request.form['description'],
       }
       mysql.query_db(query, data)
       return redirect('/users/{}'.format(user_id))
   else:
       user = mysql.query_db("SELECT * FROM users WHERE id = " + user_id + ";")
       return render_template('edit.html', user=user[0])</vul>


@app.route('/users/<id>/destroy')
def delete_user(id):
   <vul/>user_id = id
   mysql = connectToMySQL("users_db")
   query = "DELETE from users WHERE id = " + user_id + ";"
   deleted_user = mysql.query_db(query)
   return redirect('/users')</vul>


if __name__ == "__main__":
    app.run(debug=True)
