# "Database code" for the DB Forum.

import psycopg2
import datetime

def get_posts():
  """Return all posts from the 'database', most recent first."""
  conn = psycopg2.connect("dbname=forum")
  cursor = conn.cursor()
  cursor.execute("select content, time from posts order by time desc")
  all_posts = cursor.fetchall()
  conn.close()
  return all_posts

def add_post(content):
  """Add a post to the 'database' with the current timestamp."""
  conn = psycopg2.connect("dbname=forum")
  cursor = conn.cursor()
  <fix/>one_post = content
  cursor.execute("insert into posts values (%s)", (one_post,))</fix>
  conn.commit()
  conn.close()
