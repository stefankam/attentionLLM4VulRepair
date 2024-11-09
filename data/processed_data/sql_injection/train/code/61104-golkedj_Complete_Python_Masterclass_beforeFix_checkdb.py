import sqlite3

db = sqlite3.connect("contacts.sqlite")

<vul/>for row in db.execute("SELECT * FROM contacts"):</vul>
    print(row)

db.close()
