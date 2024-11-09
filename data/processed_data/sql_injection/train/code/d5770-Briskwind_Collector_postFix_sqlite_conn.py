import sqlite3

import os

from django.db import connection

from server.settings import BASE_DIR

DB_FILE = os.path.join(BASE_DIR, 'db.sqlite3')


def sqlite_executor(sql):
    <fix/>conn = sqlite3.connect(DB_FILE)</fix>

    sql = 'select * from django_migrations;'
    cursor = conn.cursor()
    cursor.execute(sql)
    res = cursor.fetchall()
    return res




def get_one_user(user_id):
    """ 获取单个用户，测试sql 注入"""

    # 防止Sql注入的sql执行方式
    # sql2 = "select * from home_user WHERE id=%s"
    # with connection.cursor() as cursor:
    #     cursor.execute(sql2, [user_id])
    #     res = cursor.fetchall()

    sql = 'select * from home_user WHERE id={};'.format(user_id)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(sql)
    res = cursor.fetchall()
    return res