<vul/></vul>#!/usr/bin/env python3

import cgi
import cgitb
from http import cookies
import urllib.parse
import mysql.connector
from mysql.connector import errorcode
import hashlib

cgitb.enable() #provides additional security by not revealing innerworkings of code to the outside

form = cgi.FieldStorage() #instantiates the form data

name = form.getvalue('name')
username = form.getvalue('username')
password = form.getvalue('password')
password2 = form.getvalue('password2')


userExists = false

if password == password2 && len(name)>0 && len(username)>0:
    #set cookies
    #set expiration time
    expires = 60*60;

    cookie = cookies.SimpleCookie()
    cookie["sessionID"] = urllib.parse.quote(Math.random())
    cookie["username"] = urllib.parse.quote(username)
    cookie["username"]['expires'] = expires

    #connect to mysql database
    conn = mysql.connector.connect(user='m201842', password = 'Bandit', host='midn.cyber.usna.edu', database='m201842')
    cursor = conn.cursor()
    cursor.execute("SELECT Username FROM Users")
    for row in cursor:
        if row == username:
            userExists = true
            break
    if userExists:
        #redirects to signup because they didn't do so correctly.
        print("Content-Type: text/html")
        print()
        print('''\
        <html>
            <head>
                <script>
                    alert("You are failed to signup correctly.");
                    alert("Make sure you don't already have an account\nand that your passwords match.");
                </script>
                <meta http-equiv="refresh" content="0;url='signup.html'"/>
            </head>
        </html>
        ''')

    else:
	#hash the password before adding to the database
	hashedPassword = hashlib.md5(password.encode())
        #add the user to the database
        cursor.execute("INSERT INTO Users VALUES ("+username+","+ name+","+ hashedPassword")")

        #redirects to the message board
        print("Content-Type: text/html")
        print()
        print('''\
        <html>
            <head>
                <script>
                    alert("You are now signed in as "+username);
                </script>
                <meta http-equiv="refresh" content="0;url='messageboard.py'"/>
            </head>
        </html>
        ''')

else:
    #redirects to signup because they didn't do so correctly.
    print("Content-Type: text/html")
    print()
    print('''\
    <html>
        <head>
            <script>
                alert("You are failed to signup correctly.");
                alert("Make sure you don't already have an account\nand that your passwords match.");
            </script>
            <meta http-equiv="refresh" content="0;url='signup.html'"/>
        </head>
    </html>
    ''')

#necessary for the database to not get messed up
cursor.close()
conn.commit()
conn.close()
