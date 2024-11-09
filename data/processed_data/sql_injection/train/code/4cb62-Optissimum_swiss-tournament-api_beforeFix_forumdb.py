<vul/>#
# Database access functions for the web forum.
#

import time
import psycopg2

## Get posts from database.
def GetAllPosts():
    conn = psycopg2.connect("dbname=forum")
    cur = conn.cursor()
    # Grabs all of our posts from our the DB
    cur.execute("SELECT time, content FROM posts ORDER BY time desc")
    # Processes our fecthall into a dicitonary
    posts = ({'content': str(row[1]), 'time': str(row[0])}
    for row in cur.fetchall())
    conn.close()
    return posts

## Add a post to the database.
def AddPost(content):
    conn = psycopg2.connect("dbname=forum")
    cur = conn.cursor()
    cur.execute("INSERT INTO posts (content) VALUES ('%s')" % content)
    conn.commit()
    conn.close()</vul>
