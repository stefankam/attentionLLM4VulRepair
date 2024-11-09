#!/usr/bin/env python

import MySQLdb
import tornado.httpserver
import tornado.web
import os.path
import json
import tornado.escape

"""
Ideally each endpoint will work for JSON and HTML.  JSON first
"""

root_url = "undefined"
server_port = 0



class Application(tornado.web.Application):
	def __init__(self):
		global root_url, server_port		
		config_file = open("config.json").read()
		config_data = json.loads(config_file)
		root_url = config_data["server"]
		server_port = config_data["port"]

		handlers = [
			(r"/main", MainHandler),
			(r"/count", CountHandler),
			(r"/", tornado.web.RedirectHandler, dict(url=r"/main")),
			(r"/([a-z]+)", DatabaseHandler),
			(r"/([a-z]+)/([0-9]+)",ForumHandler),
			(r"/([a-z]+)/([0-9]+)/([0-9]+)",ThreadHandler), 
		]
		settings = dict(
			template_path=os.path.join(os.path.dirname(__file__), "templates"),
		)
		super(Application, self).__init__(handlers, **settings)
		self.db = MySQLdb.connect(db=config_data["database"], 
						user=config_data["user"],
						passwd=config_data["passwd"])

		self.databases = config_data["databases"] 

	
"""  Ideally, we should be able to have each give direct json or wrap in html"""
class HandlerBase(tornado.web.RequestHandler):
	
	def prefix(self, name):
		for d in self.application.databases:
			if d["url"] == name:
				return d["data_prefix"]
		self.setstatus(404)	
		return ""

	def username(self, user_id):
		c = self.application.db.cursor()
		c.execute("SELECT pn_uname FROM pn_users WHERE pn_uid=%s", (user_id,))
		row = c.fetchone()
		if row:
			return row[0].decode('latin1').encode('utf8')
		else:
			return "Unknown"	

	def write_json_or_html(self, json_data):
		if("text/html" in self.request.headers["accept"]):
			self.write_html(json_data)	
		else:
			self.write(json_data)

	def write_html(self, json_data):
		#subclass this method!
		pass
	
class MainHandler(HandlerBase):
	def get(self):
		out = dict()
		dbs = []
		for d in self.application.databases:
			dbs.append({"name":d["name"], "url":root_url + d["url"]})
		out["databases"] = dbs
		self.write_json_or_html(out)
	
	def write_html(self, json_data):
		self.render("dblist.html", dbs=json_data["databases"])



class DatabaseHandler(HandlerBase):
	def get(self, db):
		prefix = self.prefix(db)
		if prefix:
			c = self.application.db.cursor()
			c.execute("SELECT forum_id, forum_name, forum_desc FROM %s_forums" % prefix)
			out = {}
			forum_list = []
			for row in c.fetchall():
				forum_name = row[1].decode('latin1').encode('utf8')
				forum_desc = row[2].decode('latin1').encode('utf8')
				forum_list.append({"name":forum_name, "description":forum_desc,"url":root_url+db+"/"+str(row[0])})
			out["forums"] = forum_list
			self.write_json_or_html(out)
        
        def write_html(self, json_data):
		self.render("database.html", forums=json_data["forums"])	


class ForumHandler(HandlerBase):
	def get(self, db, forum_id):
		prefix = self.prefix(db)
		if prefix:
			out = {}
			thread_list = []
			c = self.application.db.cursor()
			select_table = "SELECT forum_id, topic_title, topic_id, topic_poster FROM %s" % prefix + "_topics"
			select_text = select_table + " WHERE forum_id=%s ORDER BY topic_id"
			c.execute(select_text, (forum_id,))
			for row in c.fetchall():
				title = row[1].decode('latin1').encode('utf8')
				thread_list.append({"title":title, "creator":self.username(row[2]),  "url":root_url+db+"/"+forum_id+"/"+str(row[2])})
			out["threads"] = thread_list
                        self.write_json_or_html(out)

        def write_html(self, json_data):
		self.render("forum.html", threads=json_data["threads"])


class ThreadHandler(HandlerBase):
	def get(self, db, forum_id, topic_id):
		prefix = self.prefix(db)
		if prefix:
			out = {}
			post_list = []
			c = self.application.db.cursor()
			<fix/>post_limit_text = "10"
			try:
				post_limit_check = self.get_query_argument("count", "10")
				post_limit = int(post_limit_check)   #verify this is a valid int
				post_limit_text = post_limit_check 
			except ValueError:
				pass			
</fix>
			select_table_one = prefix + "_posts"
			select_table_two = prefix + "_posts_text"
			
			select_tables = "SELECT %s.post_id, poster_id, post_subject, post_text FROM %s" % (select_table_one, select_table_one)
			select_join = " INNER JOIN %s ON %s.post_id=%s.post_id" % (select_table_two, select_table_one, select_table_two)
			<fix/>select_where = " WHERE topic_id=%s ORDER BY post_id LIMIT %s"</fix>
			select_statement = select_tables + select_join + select_where
			<fix/>c.execute(select_statement, (topic_id, post_limit))</fix>
			for row in c.fetchall():
				subject = row[2].decode('latin1').encode('utf8')
				text = row[3].decode('latin1').encode('utf8')
				post_list.append({"subject":subject, "text":text,"poster":self.username(row[1]), "url":root_url+db+"/"+forum_id+"/"+topic_id+"/"+str(row[0])})
                        self.write_json_or_html({"posts":post_list})

        def write_html(self, json_data):
		self.render("thread.html", posts=json_data["posts"])
	


class CountHandler(tornado.web.RequestHandler):
	def get(self):
		c = self.application.db.cursor()
		c.execute("SELECT COUNT(*) FROM classic_posts")
		self.render("count.html", count=c.fetchone())


def main():
	http_server = tornado.httpserver.HTTPServer(Application())
	http_server.listen(server_port)
	tornado.ioloop.IOLoop.current().start()

if __name__ == "__main__":
	main()


