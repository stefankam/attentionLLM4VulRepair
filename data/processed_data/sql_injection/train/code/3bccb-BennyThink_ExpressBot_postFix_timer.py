#!/usr/bin/python
# coding:utf-8

import db
import main

if __name__ == '__main__':

    <fix/>sql_cmd = 'SELECT track_id,username,chat_id,content FROM job WHERE done=?'
    s = db.select(sql_cmd, (0,))</fix>

    for i in s:
        main.cron(i[0], i[1], i[2])
