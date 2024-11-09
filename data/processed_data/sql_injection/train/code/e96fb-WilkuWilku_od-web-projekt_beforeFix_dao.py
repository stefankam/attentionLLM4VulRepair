import sqlite3, hashlib, random, string, uuid
SALT_LENGTH = 32
DATABASE_PATH = 'db/data.db'

def add_user(username, password):
    salt = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(SALT_LENGTH))
    password_hash = multiple_hash_password(password, salt)
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute('''INSERT INTO UserData(username, password_hash, salt) 
                      VALUES (?, ?, ?)''', (username, password_hash, salt))

    connection.commit()
    connection.close()


def login(username, password):
    #todo zabezpieczyć username przed SQLinjection
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute('''SELECT user_id, password_hash, salt FROM UserData WHERE username = ?''', [username])
    data = cursor.fetchone()
    if not data:
        return None
    user_id = data[0]
    password_hash = data[1]
    salt = data[2]
    session_id = None

    if multiple_hash_password(password, salt) == password_hash:
        session_id = str(uuid.uuid4())
        cursor.execute('UPDATE UserData SET session_id = ? WHERE user_id = ?', (session_id, user_id))
        print('SID: '+session_id)
        connection.commit()

        cursor.execute('SELECT secure_name, uuid_filename FROM Notes WHERE user_id = ?', [user_id])
        notes = []
        rows = cursor.fetchall()
        for row in rows:
            notes.append({
                "file_id": row[1].split('.')[0],
                "name": row[0]
            })
    connection.close()

    return session_id, notes


def logout(session_id):
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()
    cursor.execute('UPDATE UserData SET session_id = NULL WHERE session_id = ?', [session_id])
    connection.commit()
    connection.close()


def check_session(session_id):
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM UserData WHERE session_id = ?', [session_id])
    verified = cursor.fetchone()
    connection.close()
    return verified


def is_username_taken(username):
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM UserData WHERE username = ?', [username])
    records = cursor.fetchone()
    connection.close()
    return records


def multiple_hash_password(password, salt):
    hash_value = password + salt
    for _ in range(1000):
        hash_value = hashlib.sha3_512((hash_value + password + salt).encode()).hexdigest()
    return hash_value


def is_note_uuid_taken(uuid):
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM Notes WHERE uuid_filename = ?', [uuid])
    records = cursor.fetchone()
    connection.close()
    return records


def add_notes(secure_fname, file_id, username):
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()
    cursor.execute('''INSERT INTO Notes(secure_name, user_id, uuid_filename)
                        VALUES (?, 
                        (SELECT user_id FROM UserData WHERE username = ?),
                         ?)''', (secure_fname, username, file_id))
    connection.commit()
    connection.close()


def confirm_owner_of_file(file_id, session_id, username):
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()
    cursor.execute('''SELECT session_id, username FROM UserData WHERE user_id = 
                                (SELECT user_id FROM Notes WHERE uuid_filename = ?)''', [file_id])
    row = cursor.fetchone()
    connection.close()
    return row[0] == session_id and row[1] == username


def get_secure_filename(file_id):
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()
    cursor.execute('''SELECT secure_name FROM Notes WHERE uuid_filename = ?''', [file_id])
    row = cursor.fetchone()
    connection.close()
    return row[0]
