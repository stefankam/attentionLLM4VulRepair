# "Database code" for the DB Forum.

import datetime

<vul/>POSTS = [("This is the first post.", datetime.datetime.now())]</vul>

def get_posts():
  """Return all posts from the 'database', most recent first."""
  <vul/>return reversed(POSTS)</vul>

def add_post(content):
  """Add a post to the 'database' with the current timestamp."""
  <vul/>POSTS.append((content, datetime.datetime.now()))</vul>


