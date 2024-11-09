#!/usr/bin/env python

import cgi
import sqlite3
import os
import hashlib
import uuid


def generate_hash(password):
    salt = uuid.uuid4().hex
    hash_val = hashlib.sha256(salt.encode() + password.encode()).hexdigest()
    return hash_val, salt


def check_password(hash_salt, password_input):
    return hash_salt[0] == hashlib.sha256(hash_salt[1].encode()+password_input.encode()).hexdigest()


def get_active_user():
    conn = sqlite3.connect('server.db')
    c = conn.cursor()
    cmd = "SELECT username FROM users WHERE active = ?"
    c.execute(cmd, (1, ))
    username = c.fetchone()[0]
    conn.close()
    return username


def auth_user(f):
    username = f.getvalue('username')
    password = f.getvalue('password')

    conn = sqlite3.connect('server.db')
    c = conn.cursor()
    <fix/>cmd = "SELECT hash, salt FROM users WHERE username = ?"
    c.execute(cmd, (username,))</fix>
    hash_salt = c.fetchone()

    if hash_salt is None:
        print """ Sign up for a new account:
        <a href="http://localhost:8000/index.py">Go back/a>"""
    elif check_password(hash_salt, password):
        print """ password authentication successful!"""
        <fix/>cmd = "UPDATE users SET active = ? WHERE username = ?"
        c.execute(cmd, (1, username))</fix>
        conn.commit()
        conn.close()
        display_user_page()
    else:
        print """ wrong password! Try again:
        <a href="http://localhost:8000/index.py">Log in</a>"""


def display_user_page():
    """ HTML boilerplate of forms, files listing, status of operation """
    username = get_active_user()
    print """
    <h1>%s</h1>
    ----------------------------------------------------------------------""" % username
    files_path = './files/'+username
    file_list = os.listdir(files_path)

    print """ <h2><a href="http://localhost:8000/index.py?log_out=1">Log out</a></h2>
    -----------------------------------------------------------------------"""
    print """ <h2> Change password </h2>
        <form method="post" action="login.py">
                new password:<input type="text" name="new_password"><br/>
                confirm new password:<input type="text" name="confirm_new_password"><br/>
                <input type="submit" value="Submit">
        </form>
    -----------------------------------------------------------------------"""
    print """
    <h2> Upload File </h2>
    <form enctype="multipart/form-data" action="login.py" method="post">
        <p>File: <input type="file" name="filename" /></p>
        <p><input type="submit" value="Upload" /></p>
    </form>
    ----------------------------------------------------------------------"""
    print """
    <h2> Delete File </h2>
    <form action="login.py" method="post" target="_blank">
    <select name="dropdown">"""
    #<option value="Maths" selected>Maths</option>
    for i in file_list:
        print "<option value=%s selected>%s</option>" % (i, i)
    print """
    </select>
    <input type="submit" value="Delete"/>
    </form>
    ----------------------------------------------------------------------"""
    print "<h2> file list </h2>"
    for i in file_list:
        print "<h4>%s</h4>" % (i, )
        print """
        <form method="get" action=%s>
        <button type="submit">Download!</button>
        </form>
        """ % ('./files/'+username+'/'+i, )


def check_file_ext(file_name):
    return file_name[-3:] == 'txt' or file_name[-3:] == 'log' or file_name[-3:] == 'pdf'


def upload_file(f):
    username = get_active_user()

    file_item = f['filename']
    # Test if the file was uploaded
    if file_item.filename:
        # strip leading path from file name to avoid
        # directory traversal attacks
        fn = os.path.basename(file_item.filename)
        if check_file_ext(fn):
            file_path = './files/'+username+'/' + fn
            open(file_path, 'wb').write(file_item.file.read())
            message = '<h3>The file "' + fn + '" was uploaded successfully</h3>'
        else:
            message = '<h3> Only files with extension txt/pdf/log will be accepted'
    else:
        message = '<h3>No file was uploaded</h3>'
    print message
    display_user_page()


def delete_file(f):
    username = get_active_user()
    if form.getvalue('dropdown'):
        file_name = form.getvalue('dropdown')
        file_path = './files/'+username+'/'+file_name
        os.remove(file_path)
        message = 'removed ', file_name
    else:
        message = "Not entered"

    print message
    display_user_page()


def change_password(f):
    username = get_active_user()
    new_password = f.getvalue('new_password')
    confirm_new_password = f.getvalue('confirm_new_password')
    if new_password == confirm_new_password and len(new_password) >= 6 and username != new_password:
        new_hash_salt = generate_hash(new_password)
        conn = sqlite3.connect('server.db')
        c = conn.cursor()
        <fix/>cmd = "SELECT hash, salt FROM users WHERE username = ?"
        c.execute(cmd, (username, ))</fix>
        old_hash_salt = c.fetchone()
        if old_hash_salt == new_hash_salt:
            message = 'old and new passwords are the same! try again'
        else:
            <fix/>cmd = "UPDATE users SET hash = ?, salt = ? WHERE username = ?"
            c.execute(cmd, (new_hash_salt[0], new_hash_salt[1], username))</fix>
            conn.commit()
            message = 'password change successful!'
        conn.close()
    else:
        message = "passwords don't match! or" \
                  "length should be atleast 6 characters or" \
                  "password is same as username" \
                  " try again!"
    print message
    display_user_page()


# main
form = cgi.FieldStorage()
print """Content-type: text/html\r\n\r\n"""

if 'username' in form and 'password' in form:
    auth_user(form)
elif 'filename' in form:
    upload_file(form)
elif 'dropdown' in form:
    delete_file(form)
elif 'new_password' in form and 'confirm_new_password' in form:
    change_password(form)

