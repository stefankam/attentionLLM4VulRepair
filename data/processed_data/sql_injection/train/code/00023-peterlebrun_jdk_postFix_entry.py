import re
import subprocess
import sqlite3

DB_FILE = '/home/peter/projects/jdk/db/jdk_entries.db'
CURRENT_DATESTAMP = "DATETIME(CURRENT_TIMESTAMP, 'localtime')" # special sqlite3 syntax
TEMP_FILE = '/home/peter/projects/jdk/db/temp.jdk'

<fix/>def db_execute(sql, quote_tuple, expect_return_values=False):
  db_connection = sqlite3.connect(DB_FILE) 
  db_cursor = db_connection.cursor()
  db_cursor.execute(sql, quote_tuple)
  
  if (expect_return_values):
    return_values = db_cursor.fetchall()
  else:
    return_values = None
    db_connection.commit()

  db_connection.close()

  return return_values  

def db_execute_old(sql, expect_return_values=False):</fix>
  db_connection = sqlite3.connect(DB_FILE) 
  db_cursor = db_connection.cursor()
  db_cursor.execute(sql)
  
  if (expect_return_values):
    return_values = db_cursor.fetchall()
  else:
    return_values = None
    db_connection.commit()

  db_connection.close()

  return return_values  

class Entry:

  entry_id = None
  title = None
  body = None
  date_created = None
  date_last_modified = None
  writeable = True 

  def __init__(self, entry_id=None, title=None, writeable=True):
    
    if (entry_id):
      self.entry_id = entry_id 
      self.populate_entry_data()
      self.writeable = writeable

    else:
      self.create_entry()
      self.set_entry_id()

      if(title):
        self.title = title
        self.update_title()

  @staticmethod
  def get_home_screen_data():
    quote_tuple = ()
    sql = "SELECT jdk_entries.id, jdk_entries.title " + \
      "FROM jdk_entries " + \
      "ORDER BY date_last_modified DESC " + \
      "LIMIT 30;" 

    <fix/><fix/>return db_execute(sql, quote_tuple, True)</fix></fix>

  @staticmethod
  def search_existing_entries(keyword=None, from_date=None, to_date=None):
    total_parameters = 0
    quote_tuple = ()

    if (keyword):
      <fix/>keyword_string = '(jdk_entries.title LIKE ? OR ' + \
        'jdk_entries.body LIKE ?) '</fix>
      total_parameters += 1
      keyword = '%' + keyword + '%'
      quote_tuple += keyword, keyword
    else:
      keyword_string = ""  
  
    if (from_date):
      <fix/>from_date_string = "jdk_entries.date_last_modified >= ? "</fix>
      total_parameters += 1
      quote_tuple += from_date,
    else:
      from_date_string = ""

    if (to_date):
      <fix/>to_date_string = "jdk_entries.date_last_modified <= ? "</fix>
      total_parameters += 1
      quote_tuple += to_date,
    else:
      to_date_string = ""

    sql = "SELECT jdk_entries.id, jdk_entries.title " + \
      "FROM jdk_entries " + \
      "WHERE " + \
      keyword_string + \
      ("AND " if (keyword_string and (from_date or to_date)) else "") + \
      from_date_string + \
      ("AND " if (from_date and to_date) else "") + \
      to_date_string + \
      "LIMIT 30;"

    return db_execute(sql, quote_tuple, True)

  def populate_entry_data(self):
    quote_tuple = self.entry_id,

    sql = "SELECT jdk_entries.title, jdk_entries.body " + \
      "FROM jdk_entries " + \
      <fix/>"WHERE jdk_entries.id = ?;"</fix> 

    <fix/>self.title, self.body = db_execute(sql, quote_tuple, True)[0] # returns array</fix>
    
    return None
  
  def create_entry(self):
    quote_tuple = CURRENT_DATESTAMP, CURRENT_DATESTAMP

    sql = "INSERT INTO jdk_entries " + \
      "(title, body,  date_created, date_last_modified)" + \
      <fix/>"VALUES ('', '', ?, ?);"</fix>

    <fix/><fix/><fix/><fix/>db_execute(sql, quote_tuple)</fix></fix></fix></fix>

    return None

  def set_entry_id(self):
    quote_tuple = ()
    
    sql = "SELECT MAX(id) FROM jdk_entries;"

    # returns a nested list, need to get at it with array syntax
    <fix/>self.entry_id = str(db_execute(sql, quote_tuple, True)[0][0]);</fix>     

    return None

  def update_title(self, title = None):
    if (not self.title):
      self.title = title

    quote_tuple = self.title, self.entry_id

    # This will fall to a sql injection 
    <fix/>sql = "UPDATE jdk_entries SET title = ?" + \
          "WHERE jdk_entries.id = ?;"</fix> 

    db_execute(sql, quote_tuple)
    
    self.update_date_modified()

    return None

  def update_date_modified(self):
    quote_tuple = CURRENT_DATESTAMP, self.entry_id

    sql = "UPDATE jdk_entries " + \
      <fix/>"SET date_last_modified = ? " + \
      "WHERE jdk_entries.id = ?;"</fix>
    
    db_execute(sql, quote_tuple)

    return None

  def edit_entry(self):
    self.create_temp_file()
    if (self.body is not None):
      self.write_body_to_temp_file() 
    
    self.open_temp_file() # opens vim here
    
    body_new = self.get_temp_file_data()
  
    if (body_new != self.body):
      self.body = body_new 
    
      <fix/>quote_tuple = self.body, self.entry_id

      sql = "UPDATE jdk_entries SET body = ? " + \
        "WHERE jdk_entries.id = ?;"</fix>

      db_execute(sql, quote_tuple)
      self.update_date_modified()
    
    self.remove_temp_file()
    
    return None

  def create_temp_file(self):
    subprocess.call(['touch', TEMP_FILE])

    return None

  def write_body_to_temp_file(self):
    temp_file = open(TEMP_FILE, "w")
    temp_file.write(self.body)  
    temp_file.close()

    return None

  def open_temp_file(self):
    params = ['vim', TEMP_FILE]

    if (not self.writeable):
      params.insert(1, '-R') # vim syntax for read only

    subprocess.call(params)

    return None

  def get_temp_file_data(self):
    temp_file = open(TEMP_FILE, "r")
    temp_text = temp_file.read()  
    temp_file.close()

    return temp_text

  def remove_temp_file(self):
    params = ['rm', TEMP_FILE]

    subprocess.call(params)
    
    return None
  
  def export_entry(self, location = None):
    if (not location):
      export_location = "/home/peter/Desktop/" + str(self.entry_id) + ".txt"
      #subprocess.call(['touch', export_location])
      export_file = open(export_location, "w")
      #export_file.write("Date Created " + self.date_created)
      #export_file.write("Date Last Modified " + self.date_last_modified)
      export_file.write("id " + str(self.entry_id) + "\n")
      export_file.write("title " + self.title + "\n")
      export_file.write(self.body)
      export_file.close()

    return None
"""
Entry has:
  title
  body
  id
  date created
  date modified
  file location

Entry needs
  set title (saved in DB)
  create file (touch)
  open file (w vim)
  update date modified (in DB)
  create entry meta data (in DB)
""" 
