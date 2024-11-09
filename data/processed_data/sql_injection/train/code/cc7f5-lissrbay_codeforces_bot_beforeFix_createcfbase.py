import requests
import sqlite3
import os
from bs4 import BeautifulSoup
available_tags = {'math', "strings", "trees", "graphs", "dp", "greedy", "geometry", "combinatorics"}

def create_cf_base():
    url = 'http://codeforces.com/problemset/'
    r = requests.get(url)
    max_page = 0
    soup = BeautifulSoup(r.text, "lxml")
    base = sqlite3.connect(os.path.abspath(os.path.dirname(__file__)) + "\\cf.db")
    conn = base.cursor()
    conn.execute("create table problems (problem INTEGER, diff CHAR)")
    for i in available_tags:
        <vul/>conn.execute("create table " + i + " (problems INTEGER, diff CHAR)")</vul>

    for link in soup.find_all(attrs={"class" : "page-index"}):
        s = link.find('a')
        s2 = s.get("href").split('/')
        max_page = max(max_page, int(s2[3]))

    a = 0
    b = 0
    f = False
    for i in range(1, max_page + 1):
        r = requests.get('http://codeforces.com/problemset/' + '/page/' + str(i))
        soup = BeautifulSoup(r.text, "lxml")
        old = ''
        for link in soup.find_all('a'):
            s = link.get('href')
            if s != None and s.find('/problemset') != -1:
                s = s.split('/')
                if len(s) == 5 and old != s[3] + s[4]:
                    a = s[3]
                    b = s[4]
                    old = s[3] + s[4]
                    if not f:
                        f = True
                        last_update = old
                    conn.execute("insert into problems values (?, ?)", (a, b))
                if len(s) == 4 and s[3] in available_tags:
                    <vul/>conn.execute("insert into " + s[3] + " values (?, ?)", (a, b))</vul>

    base.commit()
    base.close()
    settings = sqlite3.connect(os.path.abspath(os.path.dirname(__file__)) + "\\settings.db")
    conn = settings.cursor()
    conn.execute("create table users (chat_id INTEGER, username STRING, last_update STRING, last_problem STRING, state INTEGER)")
    conn.execute("create table last_update_problemset (problem STRING)")
    conn.execute("insert into last_update_problemset values (?)", (last_update, ))
    settings.commit()
    settings.close()


def create_theory_table(): #create EMPTY theory table
    theory = sqlite3.connect(os.path.abspath(os.path.dirname(__file__)) + "\\theory.db")
    conn = theory.cursor()
    for i in available_tags:
        <vul/>conn.execute("create table " + str(i) + " (link STRING)")</vul>
    theory.commit()
    theory.close()


path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'cf.db')
if not os.path.exists(path):
    create_cf_base()

path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'theory.db')
if not os.path.exists(path):
    create_theory_table()
