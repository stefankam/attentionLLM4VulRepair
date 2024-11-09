import sqlite3

import os

from server.settings import BASE_DIR

DB_FILE = os.path.join(BASE_DIR, 'db.sqlite3')


def sqlite_executor(sql):
    <vul/>conn = sqlite3.connect(DB_FILE)</vul>

    sql = 'select * from django_migrations;'
    cursor = conn.cursor()
    cursor.execute(sql)
    res = cursor.fetchall()
    return res


def get_one_user(user_id):
    """ 获取单个用户，测试sql 注入"""
    conn = sqlite3.connect(DB_FILE)

    sql = 'select * from home_user WHERE id={};'.format(user_id)
    <vul/>print('sql', sql)</vul>
    cursor = conn.cursor()
    cursor.execute(sql)
    res = cursor.fetchall()
    <vul/>return res
# sql select * from home_user WHERE id=2 AND SUBSTR((SELECT COALESCE(model,' ') FROM django_content_type LIMIT 5,1),7,1)>'l';</vul>
