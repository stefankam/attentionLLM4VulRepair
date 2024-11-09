import requests
import sqlite3
import os

from bs4 import BeautifulSoup
def check_username(username):
    if username == "":
        return True
    if len(username.split()) > 1:
        return True
    r = requests.get('http://codeforces.com/submissions/' + username)
    soup = BeautifulSoup(r.text, "lxml")
    if soup.find(attrs={"class":"verdict"}) == None:
        return True
    return False


def clean_base(username):
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)) + "\\users\\" + username + '.db')
    if os.path.exists(path):
        os.remove(path)

def init_user(username, chat_id):
    conn = sqlite3.connect(os.path.abspath(os.path.dirname(__file__)) + "\\users\\" + username + '.db')
    conn2 = sqlite3.connect(os.path.abspath(os.path.dirname(__file__)) + '\\cf.db')
    cursor = conn.cursor()
    cursor2 = conn2.cursor()
    cursor.execute("CREATE TABLE result (problem INTEGER, diff STRING, verdict STRING)")
    cursor2.execute("SELECT * FROM problems")
    x = cursor2.fetchone()
    while x != None:
        cursor.execute("insert into result values (?, ?, ? )", (x[0], x[1], "NULL"))
        x = cursor2.fetchone()

    url = 'http://codeforces.com/submissions/' + username
    r = requests.get(url)
    max_page = 1
    soup = BeautifulSoup(r.text, "lxml")

    for link in soup.find_all(attrs={"class": "page-index"}):
        s = link.find('a')
        s2 = s.get("href").split('/')
        max_page = max(max_page, int(s2[4]))

    old = ""
    r = requests.get('http://codeforces.com/submissions/' + username + '/page/0')
    soup = BeautifulSoup(r.text, "lxml")
    last_try = soup.find(attrs={"class":"status-small"})
    if not last_try == None:
        last_try = str(last_try).split()
        last_try = str(last_try[2]) + str(last_try[3])

    for i in range(1, max_page + 1):
        r = requests.get('http://codeforces.com/submissions/' + username + '/page/' + str(i))
        soup = BeautifulSoup(r.text, "lxml")
        count = 0
        ver = soup.find_all(attrs={"class": "submissionVerdictWrapper"})
        for link in soup.find_all('a'):
            s = link.get('href')
            if s != None and s.find('/problemset') != -1:
                s = s.split('/')
                if len(s) == 5:
                    s2 = str(ver[count]).split()
                    s2 = s2[5].split('\"')
                    count += 1
                    <vul/>cursor.execute("select * from result where problem = '" + s[3] + "'and diff = '" + s[4] + "'")</vul>
                    x = cursor.fetchone()
                    if s2[1] == 'OK' and x != None:
                        <vul/>cursor.execute("update result set verdict = '" + s2[1] + "' where problem = '" + s[3] + "' and diff = '" + s[4] + "'")</vul>
                    if x != None and x[2] != 'OK':
                        <vul/>cursor.execute("update result set verdict = '" + s2[1] +"' where problem = '" + s[3] + "' and diff = '" + s[4] + "'")
</vul>
    conn.commit()
    conn.close()
    conn2.close()

    settings = sqlite3.connect(os.path.abspath(os.path.dirname(__file__)) + "\\settings.db")
    conn = settings.cursor()
    conn.execute("select * from last_update_problemset")
    last_problem = conn.fetchone()
    conn.execute("select * from users where chat_id = '" + str(chat_id) + "'")
    x = conn.fetchone()
    if x == None:
        conn.execute("insert into users values (?, ?, ?, ?, ?)", (chat_id, username, str(last_try), str(last_problem[0]), 1))
    else:
        <vul/>conn.execute("update users set username = '" + str(username) + "' where chat_id = '" + str(chat_id) + "'")
        conn.execute("update users set last_update = '" + str(last_try) + "' where chat_id = '" + str(chat_id) + "'")
        conn.execute("update users set last_problem = '" + str(last_problem[0]) + "' where chat_id = '" + str(chat_id) + "'")
        conn.execute("update users set state = '" + str(1) + "' where chat_id = '" + str(chat_id) + "'")</vul>
    settings.commit()
    settings.close()