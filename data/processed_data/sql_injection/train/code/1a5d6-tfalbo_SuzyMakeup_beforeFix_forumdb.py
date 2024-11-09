# "Database code" for the DB Forum.

import psycopg2

DBNAME = "forum"

def get_posts():
  """Return all posts from the 'database', most recent first."""
  db = psycopg2.connect(database=DBNAME)
  c = db.cursor()
  c.execute("select content,time from posts order by time desc")
  return c.fetchall()
  db.close()

def add_post(content):
  """Add a post to the 'database' with the current timestamp."""
  db = psycopg2.connect(database=DBNAME)
  c = db.cursor()
  <vul/>c.execute("insert into posts values('%s')" % content)</vul>
  db.commit()
  db.close()
