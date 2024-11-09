# "Database code" for the DB Forum.

import datetime
import psycopg2
import bleach

<fix/># POSTS = [("This is the first post.", datetime.datetime.now())]
DBNAME= "forum"</fix>

def get_posts():
  """Return all posts from the 'database', most recent first."""
  <fix/># conn = psycopg2.connect("dbname=forum")
  conn = psycopg2.connect(database=DBNAME)
  cursor = conn.cursor()
  cursor.execute("select content,time from posts order by time desc")
  results = cursor.fetchall()
  # bleach.clean all the results
  clean_results = list(map(lambda x: (bleach.clean(x[0]), x[1]), results )) #clean the text in the database, so when it gets displayed on the browser, it doesn't get interpreted as code
  conn.close()
  # return results
  return clean_results</fix>

def add_post(content):
  """Add a post to the 'database' with the current timestamp."""
  <fix/># conn = psycopg2.connect("dbname=forum")
  clean_content = bleach.clean(content)
  conn = psycopg2.connect(database=DBNAME)
  cursor = conn.cursor()
  # print("insert into posts values(%s)", (content,))
  # cursor.execute("insert into posts values(%s)", (content,))
  cursor.execute("insert into posts values(%s)", (clean_content,)) #clean the content before it goes into the db

  conn.commit()
  conn.close()</fix>


