#Imports for all the packages
import os.path
import re
import motor.motor_tornado
<fix/>import argon2</fix>
from pymongo import MongoClient
import random
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import pymongo
from tornado.options import define, options

#Setting options for the server
define("port", default=8100, help="run on the given port", type=int)


class BaseHandler(tornado.web.RequestHandler):
	<fix/>""" BaseHandler():
	Class that'll be used later when @tornado.web.authenticated is needed for POST requests.
	"""</fix>
	def get_current_user(self):
		return self.get_secure_cookie("user")


class ErrorHandler(tornado.web.ErrorHandler):
	"""
	Default handler gonna to be used in case of 404 error
	"""
	def write_error(self, status_code, **kwargs):
		if status_code in [403, 404, 500, 503]:
			self.redirect("/")


class IndexHandler(tornado.web.RequestHandler):
	""" IndexHandler():
	Class that handles /
	"""
	def get(self):
		self.render('index.html')


class SignUpHandler(tornado.web.RequestHandler):
	<fix/>""" SignUpHandler():
	Class that handles /signup</fix>
	"""
	def get(self):
		"""	get():
		Renders the Sign Up page when the user arrives at /signup. 
		"""
		self.render('signup.html',error='')
	
	def check_if_exists(self):
		<fix/>""" check_if_exists():
		Uses the pymongo driver(so everything is synchronous) to check if the username exists in database
		then checks if the email address also exists in the database
		depending on conditions, returns None or the error message to be displayed.
		"""</fix>
		error = None
		document_username = sync_db.users.find_one({'username':self.username})
		if (document_username!=None):
			error = "Username exists already"
		document_email = sync_db.users.find_one({'email':self.email})
		if (document_email!=None):
			error = "Email exists already"
		return error

	async def do_insert(self,hashed_password):
		<fix/>""" do_insert():
		Forms a document of the username, the email, and the hashed password
		and using the Motor driver(asynchronously) inserts the document into database.
		"""</fix>
		document = {'username': self.username,'email': self.email,'password': hashed_password}
		result = await async_db.users.insert_one(document)

	def hash_password(self):
		<fix/>""" hash_password():
		Initializes an instance of argon2.PasswordHasher from argon2, hashes the password,
		verifies if the hashing happened properly, re-hashes if the verification failed,
		and then returns hashed password.
		"""
		ph = argon2.PasswordHasher()</fix>
		hashed_password = ph.hash(self.password)
		try:
			ph.verify(hashed_password,self.password)
		<fix/>except argon2.exceptions.VerifyMismatchError:</fix>
			hashed_password = ph.hash(self.password)
		return hashed_password

	async def post(self):
		<fix/>""" post():
		Sets class variables, does rudimentary checks on username and email submitted using regex
		and renders signup.html with the error if the regex fails to match the submitted value.
		Then checks if the submitted username and email already exist in database by calling check_if_exists 
		if check_if_exists returns not None then renders signup.html with the error. 
		After confirming that no errors have occured, hashes the password and then inserts it into the
		MongoDB database by calling hash_password() and do_insert() respectively.
		Finally, sets the secure cookie and logs in the user.
		"""</fix>
		self.username = self.get_argument("username").lower()
		self.email = self.get_argument("email").lower()
		self.password = self.get_argument("psword").lower()

		if (re.fullmatch('^(?=.{8,20}$)(?![_.])(?!.*[_.]{2})[a-zA-Z0-9._]+(?<![_.])$', self.username) == None): #Found at :https://stackoverflow.com/questions/12018245/regular-expression-to-validate-username
			self.render("signup.html",error="Your username doesn't follow our username rules. Please fix it.")
			return
		elif (re.fullmatch(r'(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)', self.email) == None): #Rudimentary Regex, will need to be updated to be simpler and email validation by sending an email will have to be done
			self.render("signup.html",error="Your email doesn't look like a valid email")
			return

		does_it_exist = self.check_if_exists()
		if(does_it_exist!=None):
			self.render("signup.html",error=does_it_exist)
			return

		hashed_password = self.hash_password()
		await self.do_insert(hashed_password)

		self.set_secure_cookie("user", self.username)
		self.redirect('/postlogin')
		return

class SignInHandler(tornado.web.RequestHandler):
	<fix/>""" SignInHandler():
	Class that handles /signin</fix>
	"""
	def get(self):
		""" get():
		Renders the Sign In page when the user arrives at /signin
		"""
		self.render('signin.html',error='')

	def check_database(self):
		<fix/>""" check_database():
		Creates an instance of argon2.PasswordHasher, finds if there is any document in the database with the 
		username submitted, verifies the password with the hashed password inside the database if the 
		document exists, returns None or the error message.
		"""
		ph = argon2.PasswordHasher()</fix>
		error = None
		document_username = sync_db.users.find_one({'username':self.username})
		if(document_username == None):
			error = "User doesn't exist. Please sign up first!"
		<fix/>else:
			try:
				ph.verify(document_username['password'],self.password)
			except argon2.exceptions.VerifyMismatchError:
				error = "Password is wrong, try again!"</fix>
		return error			

	def post(self):
		<fix/>""" post():
		Sets the class variables and checks the database to verify if the credentials exist and
		are valid, renders the Sign In page with the error if they don't.
		Finally, sets the secure cookie and redirects to /postlogin.
		"""</fix>
		self.username = self.get_argument("username").lower()
		self.password = self.get_argument("psword").lower()

		check_details = self.check_database()
		if(check_details!=None):
			self.render('signin.html',error=check_details)
			return

		self.set_secure_cookie("user", self.username)
		self.redirect('/postlogin')
		return

<fix/>class PostLoginHandler(BaseHandler):
	""" PostLoginHandler():
	Class that handles /postlogin
	"""
	@tornado.web.authenticated
	def get(self):
		""" get():
		Renders the postlogin page, uses the decorator to make sure the user is logged in first.
		"""
		self.render('postlogin.html',error='')
		return</fix>

<fix/>class CreatePollHandler(BaseHandler):
	""" CreatePollHandler():
	Class that handles /createpoll
	"""
	@tornado.web.authenticated</fix>
	def get(self):
		<fix/>""" get():
		Renders the createpoll page. uses the decorator to make sure the user is logged in first.
		"""
		self.render('createpoll.html',error='')
		return</fix>

<fix/>class ExistingPollsHandler(BaseHandler):
	""" ExistingPollsHandler():
	Class that handles /existingpolls
	"""
	@tornado.web.authenticated
	def get(self):
		""" get():
		Renders the existingpolls page. uses the decorator to make sure the user is logged in first.
		"""
		self.render('existingpolls.html',error='')
		return</fix>

<fix/>class LogoutHandler(tornado.web.RequestHandler):
	""" LogoutHandler():
	Class that handles /logout</fix>
	"""
	@tornado.web.authenticated
	def get(self):
		<fix/>""" get():
		Cleans out the secure cookie, but only after checking that the user is logged in first
		so as to not throw any errors. Also redirects to home page.
		"""
		self.clear_cookie("user")
		self.redirect("/")</fix>

<fix/># ---------------------MODULES BEGIN---------------------</fix>

<fix/>class CDNIncludesModule(tornado.web.UIModule):
	""" CDNIncludesModule():
	Class that has the CDN includes statements which are included in every page,
	except it's easier when it's made into a module.
	"""</fix>
	def render(self):
		<fix/>""" render():
		Renders the module as a HTML string.
		"""
		return self.render_string('modules/CDN_includes.html')</fix>

class NavbarModule(tornado.web.UIModule):
	""" NavbarModule():
	Class that has the Navbar code, put into a module for easier integration.
	"""
	def render(self):
		""" render():
		Renders the navbar code as an HTML string.
		"""
		return self.render_string('modules/navbar.html')

# ---------------------MODULES END---------------------


#---------------------MAIN BEGINS---------------------
if __name__ == '__main__':
	tornado.options.parse_command_line() 
	settings = {
		"cookie_secret": "j84i6ykTfmew9As25eYqAbs5KIhrUv/gmp801s9zRo=",
		"xsrf_cookies":True, 
		<fix/>"login_url": "/index",
		"default_handler_class": ErrorHandler, #Error Handler in case of 404s
		"default_handler_args": dict(status_code=404) #Argument that needs to be passed if 404 page is hit</fix>
	}
	async_db = motor.motor_tornado.MotorClient().example #Asynchronous DB driver  
	sync_db = MongoClient().example 					 #Synchronous DB driver

	application = tornado.web.Application(
		handlers = [
			(r'/',IndexHandler),
			(r'/signup', SignUpHandler),
			(r'/signin', SignInHandler),
			<fix/>(r'/postlogin',PostLoginHandler),
			(r'/createpoll',CreatePollHandler),
			(r'/existingpolls',ExistingPollsHandler),
			(r'/logout', LogoutHandler)</fix>
		],
		template_path = os.path.join(os.path.dirname(__file__),"templates"),
		static_path = os.path.join(os.path.dirname(__file__),"static"),
		<fix/>ui_modules={'cdn_includes': CDNIncludesModule, 'navbar':NavbarModule},</fix>
		debug = True,
		async_db = async_db,
		sync_db = sync_db,

		**settings
	)
	http_server = tornado.httpserver.HTTPServer(application)
	http_server.listen(options.port)
	tornado.ioloop.IOLoop.instance().start()

#---------------------MAIN ENDS---------------------