<vul/></vul>import os
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web

from tornado.options import define, options
define("port", default=8000, help="run on specified port", type=int)


class BaseController(tornado.web.RequestHandler):
    def get_current_user(self):
        return self.get_secure_cookie("username")

    #TODO: define this shit

class IndexController(tornado.web.RequestHandler):
    def get(self):
        self.render('index.html')#, user=self.current_user)
    #TODO: define this shit

'''class WebPageController(tornado.web.RequestHandler):'''
    #TODO: define this shit

class LoginController(BaseController):
    def get(self):
        self.render(login.html)
    def post(self):
        self.set_secure_cookie("username", self.get_argument("username"))#passed from html with the tag username
        #if PASSWORD is good, self.set_secure_cookie(username, self.get_argument("username"))
        self.redirect("/", permanent=True)#if permanent = true, when user refreshes, more form data will NOT be sent
    def get(self):
        self.render('login.html')#login.html page to be rendered

    class LogoutController(BaseController):
        def get(self):
            self.clear_cookie("username")
            self.redirect("/", permanent = true)
    #TODO: define this shit"""

class ReportController(BaseController):
    def get(self):
        self.render('report.html')
        
    class NewReportController(BaseController):
        @tornado.web.authenticated
        def get(self):
            self.render('create.html')
	    #ADDED by Jimmy and david, incomplete sample code
	    #def post(self):
	    #	variable_1 = self.get_argument("form variable name here")
	    #	....
	    #	if PASSWORD is good, self.set_secure_cookie(username, self.get_argument("username"))'''

	        #put appropriate fetches from template here and send to database

'''class UserProfileController(BaseController):'''
    #TODO: define this shit
    
def launch():
    server_settings = {"static_path": os.path.join(os.path.dirname(__file__), "./static"), 
    "template_path": "./server/templates", 
    "login_url": "/login", 
    <vul/>"cookie_secret": os.urandom(24), 
    "xsrf_cookies": True}</vul>

    handlers = [ (r'/', IndexController),
        (r'/report', ReportController),
        (r'/create', ReportController.NewReportController), 
        (r'/login', LoginController), 
        (r'/logout', LoginController.LogoutController) ]
    
    application = tornado.web.Application(handlers, **server_settings)
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()