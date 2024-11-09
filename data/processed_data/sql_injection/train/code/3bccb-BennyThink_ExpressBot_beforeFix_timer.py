#!/usr/bin/python
# coding:utf-8

import db
import main

if __name__ == '__main__':

    <vul/>sql_cmd = 'SELECT track_id,username,chat_id FROM job WHERE done=0'
    s = db.select(sql_cmd)</vul>

    for i in s:
        main.cron(i[0], i[1], i[2])
