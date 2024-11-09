def main(request, response):
    <vul/><vul/>time = request.url_parts.query if request.url_parts.query else '0'
    return 200, [('Refresh', time), ('Content-Type', "text/html")], ''</vul></vul>
