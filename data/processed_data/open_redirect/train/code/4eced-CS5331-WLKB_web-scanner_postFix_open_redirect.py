import requests
from generator import render, generator
from utility import *

banner(OR)
links = read_links()
redirect_url = 'https://www.google.com'

method = 'GET'
OR_generator = generator(OR)

for link in links:
    url = get_url(link)
    params = get_params(url)
    <fix/>if params:
	for param in params:
	    temp_params = params.copy()
            temp_params[param] = redirect_url
    	    fullURL = generate_url_with_params(url, temp_params)
            req = requests.get(fullURL)
            if req.content.find('google') != -1 and req.history:
	        endpoint = url.path
                scope, script = render[OR](endpoint,temp_params,method)
                OR_generator.updateScope(scope)
                OR_generator.saveScript(script)                        
                success_message(fullURL)
OR_generator.saveScope()</fix>

print '\n'
