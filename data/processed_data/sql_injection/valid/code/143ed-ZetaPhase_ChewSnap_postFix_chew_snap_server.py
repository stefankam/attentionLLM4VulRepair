from flask import Flask
from flask import request
import ast
import sqlite3
import bcrypt

app = Flask(__name__)
DATABASE = "chew_snap_database.db"

@app.route("/", methods=["GET", "POST"])
def hello():
    #if request.method == "GET":
    print "someone said get"
    return "What's up santosh"
    

@app.route("/login", methods=["GET", "POST"])
def request_login():
    print "user requesting login"
    print str(request.form)
    parameters = str(request.form)[22:]
    parameters = parameters[:-9]
    print parameters
    dic = ast.literal_eval(parameters)
    email = dic["email"]
    password = dic["password"]
    print (email)
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    <fix/><fix/>c.execute("SELECT * FROM users WHERE email=?", (email,))</fix></fix>
    user = c.fetchone()
    print user
    conn.close()
    if(not user):
        # user does not exist
        return "login_404_NOTFOUND"
    else:
        # user does exist   
        name = user[1]
        email = user[2]
        hashed = user[3]
        hashed = hashed.encode('ascii', 'ignore') # mreencode the unicode hash
        if(bcrypt.hashpw(password, hashed)==hashed): #check if password hash matches
            return "login_200_FOUND " + name
        else:
            return "login_201_INVALID"

@app.route("/signup", methods=["GET", "POST"])
def request_signup():
    print "user requesting signup"
    # print str(request.form)
    parameters = str(request.form)[22:] #remove immutable dict tags
    parameters = parameters[:-9]
    print parameters
    dic = ast.literal_eval(parameters)
    name = dic["name"]
    email = dic["email"]
    password = dic["password"]
    print (name + " " + email)
    hashed = bcrypt.hashpw(password, bcrypt.gensalt());
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email=?", (email,))
    user = c.fetchone()
    if(user != None):
        return "signup_409_USEREXISTS"
    else:
        c.execute('SELECT COUNT(userid) FROM users')
        count = c.fetchone()[0]
		<fix/>c.execute("INSERT INTO users VALUES(?, ?, ?, ?)", (str(count), name, email, hashed))</fix>
        conn.commit()
        conn.close()
        return "signup_200_OK"



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)