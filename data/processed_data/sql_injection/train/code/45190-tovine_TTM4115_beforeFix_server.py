import asyncio, base64, bcrypt, time, string
from aiohttp import web
from cryptography import fernet
from aiohttp_session import setup as session_setup, get_session, session_middleware
from aiohttp_session.cookie_storage import EncryptedCookieStorage
from psycopg2 import IntegrityError
import database
import mazemap

HTML_base = """
<!doctype html>
<html>
<title>Toilet Finder</title>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">

{text}

</html>
"""[1:-1]

#decorators:
def handle_html(func):
	#handle_html.timeout
	async def ret(*args, **kwargs):
		session = await get_session(args[0])
		
		if "uname" in session and "ignore_timeout" not in session:
			t = time.time()
			prev = session["visit_time"]
			if t - prev > handle_html.timeout: del session["uname"]
			session["visit_time"] = t
		else:
			session["visit_time"] = time.time()
		
		text = await func(*args, **kwargs)
		
		out = web.Response(
			content_type = "text/html",
			text = HTML_base.format(text=text)
		)
		
		return out
	return ret
def using_base(filename):
	with open(f"base/{filename}", "r") as f:
		base = f.read()
	def decorator(func):
		async def ret(request):
			out = await func(request, base)
			return out
		return ret
	return decorator
def require_login(func):
	async def ret(*args, **kwargs):
		session = await get_session(args[0])
		if "uname" not in session:
			session["return_after_login"] = args[0].path_qs
			return "You must be <a href=\"/login\">logged in</a> to access this page."
		out = await func(*args, **kwargs)
		return out
	return ret
def cache_page(func):#doesn't account for query parameters or different users
	cache = [None, 0]
	#cache_page.timeout
	async def ret(*args, **kwargs):
		if time.time() - cache[1] > cache_page.timeout:
			cache[0] = await func(*args, **kwargs)
			cache[1] = time.time()
		return cache[0]
	return ret

#index
@handle_html
@using_base("index.html")
@cache_page
async def GET_index(request, base):
	text = "\n".join((
		"<a href=\"/mapmaker\">/mapmaker</a><br/>",
		"<a href=\"/map\">/map</a><br/>",
		"<a href=\"/login\">/login</a><br/>",
		"<a href=\"/settings\">/settings</a><br/>",
		"<a href=\"/mazetest\">/mazetest</a><br/>",
		"<a href=\"http://disco.fleo.se/TEAM%2010%20FTW!!!\">Team 10 FTW</a>"
	))
	return base.format(text = text)


#login and registration
@handle_html
@using_base("loginform.html")
async def GET_login(request, base):
	session = await get_session(request)
	
	if "uname" not in session:
		return base
	else:
		return "<b>You're already logged in as <i>%s</i>!</b>\n<form action='/login' method='post'><input type='submit' name='action' value='logout'/></form>" % session["uname"]

@handle_html
async def POST_login(request):
	session = await get_session(request)
	data = await request.post()
	
	if set(["action", "uname", "psw"]).issubset(data.keys()) and "uname" not in session:
		uname = data["uname"]
		psw = data["psw"]
		
		if data["action"] == "login":
			
			entry = await database.select_user(request, uname)
			if not entry:
				return "Error: No such user."
			
			if bcrypt.hashpw(psw.encode("UTF-8"), entry[0][2].encode("UTF-8")).decode("UTF-8") != entry[0][2]:
				return "Error: Wrong password"
			
			session["uname"] = uname
			session["login_time"] = time.time()
			
			if "keep" in data and data["keep"] == "logged_in":
				session["ignore_timeout"] = True
			
			#success
			out = "Login successfull!"
			if "return_after_login" in session:
				out += f"<br/>\n<a href=\"{session['return_after_login']}\">Go back</a>"
				del session["return_after_login"]
			return out
		elif data["action"] == "register" and "psw2" in data:
			if not is_valid_username(uname):
				return f"Error: invalid username: <i>{uname}</i><br>\nWe only allow characters from the english alphabet plus digits"
			
			if psw != data["psw2"]:
				return "Error: mismatching passwords!"
			
			bhash = bcrypt.hashpw(psw.encode("UTF-8"), bcrypt.gensalt()).decode("UTF-8")
			
			try:
				await database.insert_user(request, uname, bhash)
			except IntegrityError:
				return "Error: username already taken!"
			
			return "User created! <a href=\"/login\">login over here.</a>"
	elif "action" in data:
		if data["action"] == "logout" and "uname" in session:
			for i in ("uname", "ignore_timeout"):
				if i in session:
					del session[i]
			return "Logged out"
			
			
	
	return f"Invalid login POST:<br/><i>{data.items()}</i><br>\nAlready logged in: {'uname' in session}"

#maps:
@handle_html
@using_base("mapmaker.html")
@cache_page
async def GET_mapmaker(request, base):
	tags = await database.select_tags(request)
	tag_checkboxes = "\n\t".join((f"<input type=\"checkbox\" name=\"tag\" value=\"{ID}\"> {label}<br>" for ID, label in tags))
	return base.format(tags=tag_checkboxes)

@handle_html
async def GET_map(request):
	#session = await get_session(request)
	tags = []
	mode = "all"
	for key, i in request.query.items():
		if key == "mode":
			mode = i
		elif key == "tag":
			<vul/>tags.append(i)</vul>
	
	if not tags:
		toilets = await database.select_toilet_statuses(request)
	else:
		toilets = await database.select_toilet_statuses_by_tags(request, tags)
	
	red, blue = [], []
	for ID, lat, lng, name, status, dt in toilets:
		(blue if status else red).append((ID, lat, lng, name, None))
	
	out = "%s\n%s" % (
		mazemap.make_marker_chubs(red, color = "red") if mode == "all" else "",
		mazemap.make_marker_chubs(blue, color = "blue")
	)
	
	return mazemap.JS_skeleton.format(code = out)

#menus:
@handle_html
@require_login
@using_base("settings.html")
async def GET_settings(request, base):
	
	return base

@handle_html
@require_login
async def POST_settings(request):
	session = await get_session(request)
	data = await request.post()
	uname = session["uname"]
	
	if "action" in data:
		if data["action"] == "change_password":
			if set(["cpsw", "psw", "psw2"]).issubset(data.keys()):
				if data["psw"] != data["psw2"]:
					return "New passwords doesn't match!"
				
				#check if current password matches:
				entry = await database.select_user(request, uname)
				if not entry:
					return "Error: Logged in as non-existing user! (what?)"
				cpsw = data["cpsw"]
				if bcrypt.hashpw(cpsw.encode("UTF-8"), entry[0][2].encode("UTF-8")).decode("UTF-8") != entry[0][2]:
					return "Error: \"Current password\" was incorrect"
				
				#set new password
				psw = data["psw"]
				bhash = bcrypt.hashpw(psw.encode("UTF-8"), bcrypt.gensalt()).decode("UTF-8")
				await database.update_user_password(request, uname, bhash)
				
				return "Success! Your password has been changed!<br>\n<a href=\"/settings\">Click here to go back.</a>"
	
	
	return f"Invalid POST request: <i>{data.items()}</i>"


@handle_html
async def GET_test(request):
	#session = await get_session(request)
	out = await mazemap.test(request)
	return out

#=====================================================================
def is_valid_username(uname):
	for i in uname:
		if i not in string.ascii_letters and i not in string.digits:
			return False
	return True

def create_session_secret():
	fernet_key = fernet.Fernet.generate_key()
	return base64.urlsafe_b64decode(fernet_key)

def add_routes(app, secret_key):
	handle_html.timeout = app["ini"].getint("sessions", "session_idle_timeout")
	cache_page.timeout  = app["ini"].getint("sessions", "cached_page_timeout")
	
	#app.router.add_route('POST', '/pv/v1/', handle_v1)
	app.router.add_get('/',      GET_index)
	app.router.add_get('/index', GET_index)
	
	app.router.add_get ('/login', GET_login)
	app.router.add_post('/login', POST_login)
	
	app.router.add_get ('/settings', GET_settings)
	app.router.add_post('/settings', POST_settings)
	
	app.router.add_get('/mapmaker', GET_mapmaker)
	app.router.add_get('/map', GET_map)
	
	app.router.add_get('/mazetest', GET_test)
	
	app.router.add_static(
		'/static/',
		path='static',
		name='static'
	)
	
	session_setup(app, EncryptedCookieStorage(secret_key))
