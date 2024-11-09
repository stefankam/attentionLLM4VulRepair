import re
import subprocess
import sqlite3

DB_FILE = '/home/peter/projects/jdk/db/jdk_entries.db'
CURRENT_DATESTAMP = "DATETIME(CURRENT_TIMESTAMP, 'localtime')" # special sqlite3 syntax
TEMP_FILE = '/home/peter/projects/jdk/db/temp.jdk'

<vul/>def db_execute(sql, expect_return_values=False):</vul>
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
    sql = "SELECT jdk_entries.id, jdk_entries.title " + \
      "FROM jdk_entries " + \
      "ORDER BY date_last_modified DESC " + \
      "LIMIT 30;" 

    <vul/><vul/>return db_execute(sql, True)</vul></vul>

  @staticmethod
  def search_existing_entries(keyword=None, from_date=None, to_date=None):
    total_parameters = 0

    if (keyword):
      <vul/>keyword_string = "(jdk_entries.title LIKE '%" + keyword + "%' OR " + \
        "jdk_entries.body LIKE '%" + keyword + "%') "</vul>
      total_parameters += 1
    else:
      keyword_string = ""  
  
    if (from_date):
      <vul/>from_date_string = "jdk_entries.date_last_modified >= '" + from_date + "' "</vul>
      total_parameters += 1
    else:
      from_date_string = ""

    if (to_date):
      <vul/>to_date_string = "jdk_entries.date_last_modified <= '" + to_date + "' "</vul>
      total_parameters += 1
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

    return db_execute(sql, True)

  def populate_entry_data(self):
    sql = "SELECT jdk_entries.title, jdk_entries.body " + \
      "FROM jdk_entries " + \
      <vul/>"WHERE jdk_entries.id = " + self.entry_id + ";"</vul> 

    <vul/>self.title, self.body = db_execute(sql, True)[0] # returns array</vul>
    
    return None
  
  def create_entry(self):
    sql = "INSERT INTO jdk_entries " + \
      "(title, body,  date_created, date_last_modified)" + \
      <vul/>"VALUES ('', '', " + CURRENT_DATESTAMP + \
      ", " + CURRENT_DATESTAMP + ");"</vul>

    <vul/><vul/><vul/><vul/>db_execute(sql)</vul></vul></vul></vul>

    return None

  def set_entry_id(self):
    sql = "SELECT MAX(id) FROM jdk_entries;"

    # returns a nested list, need to get at it with array syntax
    <vul/>self.entry_id = str(db_execute(sql, True)[0][0]);</vul>     

    return None

  def update_title(self, title = None):
    if (not self.title):
      self.title = title

    # This will fall to a sql injection 
    <vul/>sql = "UPDATE jdk_entries SET title = '" + self.title + "'" + \
          "WHERE jdk_entries.id = '" + self.entry_id + "';"</vul> 

    db_execute(sql)
    
    self.update_date_modified()

    return None

  def update_date_modified(self):
    sql = "UPDATE jdk_entries " + \
      <vul/>"SET date_last_modified = " + CURRENT_DATESTAMP + " " + \
      "WHERE jdk_entries.id = '" + self.entry_id + "';"</vul>
    
    db_execute(sql)

    return None

  def edit_entry(self):
    self.create_temp_file()
    if (self.body is not None):
      self.write_body_to_temp_file() 
    
    self.open_temp_file() # opens vim here
    
    body_new = self.get_temp_file_data()
  
    if (body_new != self.body):
      self.body = body_new 
    
      <vul/>sql = "UPDATE jdk_entries " + \
        "SET body = '" + self.body + "' " + \
        "WHERE jdk_entries.id = '" + self.entry_id + "';"</vul>

      db_execute(sql)
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
