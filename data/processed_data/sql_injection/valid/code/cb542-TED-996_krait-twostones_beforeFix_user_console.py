import json
import urllib
import math
import cx_Oracle

from db_access import db_ops
from model import user
from db_access.exceptions import printException, printf, get_error_message

class UserConsoleController(object):
	items_per_page = 20

	def __init__(self, request):
		# List of strings, each is an error. Add with self.error_messages.append(error_message)
		self.error_messages = []

		try:
			db_conn = db_ops.get_connection()
		except cx_Oracle.DatabaseError, exception:
			del self.error_messages[:]
			self.error_messages.append("Sorry, but we could't connect to the WEGAS Database\n")
			self.error_messages.append(exception)
			
		query = request.query
		self.page = int(query.get("page", 1))
		self.filter = query.get("filter", "")
		self.max_page = self.get_page_count(db_conn, self.filter)
		print "for a total max page of {}".format(self.max_page)
		
		error_msg_json = query.get("errors", None)
		if error_msg_json is not None:
			self.error_messages += json.loads(error_msg_json)
		
		# List of objects of type User 
		self.users = self.get_users(db_conn, self.page, self.filter)

		self.fetch_id = query.get("fetch_id", "")
		if self.fetch_id is not "":
			fetch_user = self.get_user(self.fetch_id)
			self.fetch_username, self.fetch_password, self.fetch_mmr, self.fetch_level =\
				fetch_user.name, "", fetch_user.mmr, fetch_user.playerLevel
		else:
			self.fetch_username, self.fetch_password, self.fetch_mmr, self.fetch_level = "", "", "", ""

		self.page_prev_url = None if self.page == 1 else build_link(self.page - 1, self.filter, self.fetch_id)
		self.page_next_url = None if self.page == self.max_page else build_link(self.page + 1, self.filter, self.fetch_id)

		min_page = max(self.page - 4, 1)
		max_page = min(self.max_page, min_page + 10)
		self.pages = [(nr, build_link(nr, self.filter, self.fetch_id)) for nr in xrange(min_page, max_page + 1)]
		# TODO: handle errors and add error messages
		
	def get_users(self, conn, page, name_filter):
		cursor = conn.cursor()
		try:
			cursor.execute("select * from table(user_ops.getUsers(:row_start, :row_count, :filter))",
				{
					"row_start": int((page - 1) * UserConsoleController.items_per_page + 1),
					"row_count": int(UserConsoleController.items_per_page),
					"filter": str(name_filter)
				}
			)
			return [user.User(*row) for row in cursor]
		except cx_Oracle.DatabaseError, exception:
			print 'Failed to get users from WEGAS'
			printException(exception)
			self.error_messages.append('Faild to get users from WEGAS\n')
			self.error_messages.append(exception)

	def get_user(self, conn, user_id):
		cursor = conn.cursor()

		cursor.execute("select * from player where id = :id", {"id": user_id})
		return user.User(*(cursor.fetchone()))

	def get_page_count(self, conn, name_filter):
		cursor = conn.cursor()
		if filter:
			# SQL injection secured:
			<vul/># cursor.execute("select count(*) from player where playername like '%' || :name_filter || '%'", {"name_filter": name_filter})</vul>
			# SQL injection vulnerable:
			<vul/>stmt = "select count(*) from player where playername like '%{}%'".format(name_filter)</vul>
			print stmt
			cursor.execute(stmt)
		else:
			cursor.execute("select count(*) from player")	
		nr_raw = cursor.fetchone()[0]
		print "Total {} users matching filter".format(nr_raw)
		return int(math.ceil(float(nr_raw) / UserConsoleController.items_per_page))

		try:
			cursor.execute("select * from players where id = :id", {"id": user_id})
			return users.User(*(cursor.fetchone()))
		except cx_Oracle.DatabaseError, exception:
			self.error_messages.append('Sorry, we couldn\'t find the user for your ID \n')
			self.error_messages.append(exception)

	def get_view(self):
		#return ".view/admin/admin_page.html"
		return ".view/admin/admin_page.html"


def build_link(page, name_filter, fetch_id):
	link_base = "/admin/user_console"
	if page is None and filter is None and fetch_id is None:
		return link_base
	
	link_query = link_base + "?"
	link_chr = ""

	if page is not None:
		link_query += link_chr + "page=" + urllib.quote_plus(str(page))
		link_chr = '&'

	if name_filter:
		 link_query += link_chr + "filter=" + urllib.quote_plus(str(name_filter))
		 link_chr = '&'

	if fetch_id:
		link_query += link_chr + "fetch_id=" + urllib.quote_plus(str(fetch_id))
		link_chr = '&'

	return link_query

