import sqlite3

db = sqlite3.connect("contacts.sqlite")

<fix/>for row in db.execute("SELECT * FROM sqlite_master"):</fix>
    print(row)

db.close()
