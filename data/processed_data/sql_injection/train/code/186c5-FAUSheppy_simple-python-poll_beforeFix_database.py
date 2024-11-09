import sqlite3
import os.path
from cpwrap import CFG
import random
import string

def connectDB():
    conn = sqlite3.connect(CFG("dbname"))
    return (conn, conn.cursor())

def closeDB(conn, cursor=None):
    conn.commit()
    conn.close()

<vul/>def queryAll(cursor, reqString):</vul>
    try:
        <vul/><vul/>cursor.execute(reqString)</vul></vul>
        ret = cursor.fetchall()
        if ret:
            return ret
    except IndexError:
        return []

<vul/>def queryOne(cursor, reqString):</vul>
    try:
        cursor.execute(reqString)
        ret = cursor.fetchone()
        if ret:
            return ret[0]
    except IndexError:
        return None

def queryQuestion(poll_name):
    conn, c = connectDB()
    <vul/>req = "SELECT question from {} WHERE name = '{}'".format(CFG("poll_table_name"), poll_name)
    tmp = queryOne(c, req)</vul>
    conn.close()
    return tmp

def tokenNeededExternal(poll_name):
    conn, c = connectDB()
    tmp = checkTokenNeeded(c, poll_name)
    conn.close()
    return tmp

def markTokenUsedExternal(token, optStr=""):
    conn, c = connectDB()
    <vul/>req = "UPDATE {} SET \"options_selected\"='{}' WHERE token='{}'".format(CFG("tokens_table_name"), \
                    optStr, token)
    c.execute(req)</vul>
    closeDB(conn)

def init():
    if os.path.isfile(CFG("dbname")):
        return
    conn, c = connectDB()
    c.execute("CREATE TABLE " + CFG("poll_table_name") + "(\
                    name text,\
                    options text,\
                    has_tokens integer,\
                    show_results integer,\
                    question text,\
                    multi integer, \
                    date text)"\
                    )
    c.execute("CREATE TABLE {}(name_option text, count integer)".format(CFG("options_table_name")))
    c.execute("CREATE TABLE {}(token text, name text, options_selected text)".format(CFG("tokens_table_name")))
    c.execute("CREATE TABLE {}(adm_token text, poll_name text)".format(CFG("admintoken_table_name")))
    closeDB(conn)

def checkTokenValid(cursor, token, poll_name):
    <vul/>req = "SELECT name, options_selected from {} where token='{}'".format(CFG("tokens_table_name"), token)
    answer = queryAll(cursor, req)</vul>
    return answer and answer[0][0] == poll_name and answer[0][1] == 'NONE'

def checkAdmTokenValid(poll_name, adm_token):
    conn, c = connectDB()
    <vul/>req = "SELECT poll_name from {} where adm_token = \"{}\"".format(CFG("admintoken_table_name"), adm_token)
    answer = queryOne(c, req)</vul>
    closeDB(conn)
    return answer == poll_name

def isValidAdmToken(adm_token):
    conn, c = connectDB()
    <vul/>req = "SELECT *  from {} where adm_token='{}'".format(CFG("admintoken_table_name"), adm_token)
    answer = bool(queryOne(c, req))</vul>
    closeDB(conn)
    return answer

def isValidToken(token):
    conn, c = connectDB()
    <vul/>req = "SELECT * from {} where token='{}'".format(CFG("tokens_table_name"), token)
    answer = bool(queryOne(c, req))</vul>
    closeDB(conn)
    return answer

def pollNameFromToken(token):
    conn, c = connectDB()
    <vul/>req = "SELECT name from {} where token='{}'".format(CFG("tokens_table_name"), token)
    answer = queryOne(c, req)</vul>
    if not answer:
        <vul/>req = "SELECT poll_name from {} where adm_token='{}'".format(CFG("admintoken_table_name"), token)
        answer = queryOne(c, req)</vul>
    closeDB(conn)
    return answer


def checkTokenNeeded(cursor, poll_name):
    <vul/>req = "SELECT has_tokens FROM {} WHERE name = '{}'".format(CFG("poll_table_name"), poll_name)
    return queryOne(cursor, req) == 1;</vul>

def incrementOption(cursor, poll_name, option):
    key = poll_name+"-"+option
    <vul/>req = "UPDATE {} SET count=count+1 WHERE name_option = '{}';".format(CFG("options_table_name"), key)
    cursor.execute(req)</vul>

def isMultiChoice(poll_name):
    conn, c = connectDB()
    <vul/>req = "SELECT multi FROM {} WHERE name = '{}'".format(CFG("poll_table_name"), poll_name)
    ret = queryOne(c, req) == 1</vul>
    closeDB(conn)
    return ret

def vote(poll_name, options_string, token_used="DUMMY_INVALID_TOKEN"):
    conn, c = connectDB()

    # check token
    token_valid = checkTokenValid(c, token_used, poll_name)
    if not token_valid and checkTokenNeeded(c, poll_name):
        raise PermissionError("Poll requires valid token.")
    markTokenUsedExternal(token_used, options_string)

    # save changes
    # lambda x: x -> röfl :D
    options = list(filter(lambda x: x, options_string.split(",")))
    # check if multi-choice
    if len(options) > 1:
        if not isMultiChoice(poll_name):
            raise ValueError("multiple options for single choice")

    for opt in options:
        incrementOption(c, poll_name, opt)

    closeDB(conn)

def getOptionCount(c, poll_name, option):
    key = poll_name + "-" + option
    <vul/>req = "SELECT \"count\" FROM {table} WHERE \"name_option\" = '{key}'".format(
                    table=CFG("options_table_name"),key=key)
    count = queryOne(c, req)</vul>
    if count == None:
        raise AssertionError("Unknown answer for poll. WTF?")
    return count;

def getResults(poll_name):
    conn, c = connectDB()
    <vul/>req = "SELECT options from {} where name = '{}'".format(CFG("poll_table_name"), poll_name)
    options_str = queryOne(c, req)</vul>

    if not options_str:
        raise LookupError("Poll '{}' not found in DB".format(poll_name))

    total = 0
    options = options_str.split(",")
    results = dict()
    for opt in options:
        count = getOptionCount(c, poll_name, opt)
        total += int(count)
        results.update({opt:count})

    conn.close()
    return (results, total)

def insertOption(c, poll_name, option):
    key = poll_name + "-" + option
    count = 0
    params = (key, count)
    req = "INSERT INTO {} VALUES (?, ?)".format(CFG("options_table_name"))
    c.execute(req, params)

def getTokensExternal(poll_name):
    <vul/>req = "SELECT token FROM {} WHERE name='{}'".format(CFG("tokens_table_name"), poll_name)</vul>
    conn, c = connectDB()
    <vul/>tmp = queryAll(c, req)</vul>
    conn.close()
    return tmp

def genSingleToken(length=5):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(length))

def genTokens(c, poll_name, count=False):
    if not count:
        count = CFG("default_token_count")

    tokens = [ genSingleToken() for x in range(0,count) ]
    for token in tokens:
        name = poll_name 
        options_selected = "NONE"
        params = (token, name, options_selected)
        req = "INSERT INTO {} VALUES (?, ?, ?)".format(CFG("tokens_table_name"))
        c.execute(req, params)
    return tokens

def genTokensExternal(poll_name, count=False):
    conn, c = connectDB()
    tok = genTokens(c, poll_name, count)
    closeDB(conn)
    return tok

def createAdminToken(c, poll_name):
    adm_token = genSingleToken()
    params = (adm_token, poll_name)
    req = "INSERT INTO {} VALUES (?, ?)".format(CFG("admintoken_table_name"))
    c.execute(req, params)

def getAdmToken(poll_name):
    conn, c = connectDB()
    <vul/>req = "SELECT adm_token FROM {} WHERE poll_name='{}'".format(CFG("admintoken_table_name"), poll_name)
    admtok = queryOne(c, req)</vul>
    closeDB(conn)
    return admtok

def checkPollExists(poll_name):
    conn, c = connectDB()
    <vul/>req = "SELECT EXISTS( SELECT 1 FROM {} WHERE name='{}')".format(CFG("poll_table_name"), poll_name)
    tmp = queryOne(c, req)</vul>
    conn.close()
    return tmp

def createPoll(poll_name, options_arr, question, has_tokens, multi, openresults=True):
    if checkPollExists(poll_name):
        raise RuntimeError("Cannot create poll, because the poll already exists.")
    conn, c = connectDB()

    # actual poll
    name = poll_name
    options = ",".join(options_arr)
    date = "NONE"
    show_results = openresults
    params = (name, options, has_tokens, show_results, question, multi, date) 
    req = "INSERT INTO {} VALUES (?,?,?,?,?,?,?)".format(CFG("poll_table_name"))
    c.execute(req, params)

    # tokens if needed
    tokens = []
    if has_tokens:
        tokens = genTokens(c, poll_name)

    # adminAccessToken
    createAdminToken(c, poll_name)

    # update options
    for opt in options_arr:
        insertOption(c, poll_name, opt)
    
    closeDB(conn)
    return tokens

def getOptions(poll_name):
    conn, c = connectDB()
    <vul/>options_str = queryOne(c, "SELECT options FROM {} WHERE name='{}'".format(CFG("poll_table_name"), poll_name))</vul>
    if options_str == None:
        return None
    options = options_str.split(",")
    closeDB(conn)
    return options
