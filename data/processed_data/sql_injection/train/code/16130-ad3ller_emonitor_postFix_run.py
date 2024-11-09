# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 13:43:09 2017

@author: Adam
"""
import sys
import os
import time
import warnings
import getpass
import sqlite3
from importlib import import_module
import pymysql
from cryptography.fernet import Fernet
from .core import (TABLE,
                   DATA_DIRE,
                   KEY_FILE)
from .tools import (db_check,
                    db_insert,
                    sql_insert,
                    parse_settings)


def get_columns(settings):
    """ get columns from sensor names """
    sensors = settings["sensors"]
    if "column_fmt" in settings:
        column_fmt = settings["column_fmt"]
        columns = ("TIMESTAMP",) \
                  + tuple([column_fmt.replace("{sensor}", str(sen).strip()) for sen in sensors])
    else:
        columns = ("TIMESTAMP",) \
                  + tuple([str(sen).strip() for sen in sensors])
    return columns

def get_device(settings, instrum, debug=False):
    """ get instance of device_class """
    if "device_class" in settings:
        device_class = settings["device_class"]
    else:
        if instrum in ["simulate", "fake"]:
            device_class = "fake.Fake"
        else:
            device_class = "generic.Generic"
    # serial connection
    mod, obj = device_class.split(".")
    module = import_module("..devices." + mod, __name__)
    return getattr(module, obj)(settings, debug=debug)

def get_sqlite(settings, columns):
    """ get sqlite connection """
    assert "db" in settings, "`db` not set in config"
    name = settings["db"]
    fname, _ = os.path.splitext(name)
    fname += ".db"
    fil = os.path.join(DATA_DIRE, fname)
    if not os.path.isfile(fil):
        raise OSError(f"{fname} does not exists.  Use generate or create.")
    db = sqlite3.connect(fil)
    db_check(db, TABLE, columns)
    return db

def get_sql(settings):
    """ get connection to sql database """
    assert "sql_host" in settings, "sql_host not set in config"
    assert "sql_port" in settings, "sql_port not set in config"
    assert "sql_db" in settings, "sql_db not set in config"
    assert "sql_table" in settings, "sql_table not set in config"
    if "sql_user" not in settings:
        settings["sql_user"] = input("SQL username: ")
    else:
        print(f"SQL username: {settings['sql_user']}")
    if "sql_passwd" not in settings:
        prompt = f"Enter password: "
        sql_passwd = getpass.getpass(prompt=prompt, stream=sys.stderr)
    else:
        # decrypt password
        assert os.path.isfile(KEY_FILE), f"{KEY_FILE} not found.  Create using passwd."
        with open(KEY_FILE, "rb") as fil:
            key = fil.readline()
        fern = Fernet(key)
        sql_passwd = fern.decrypt(bytes(settings["sql_passwd"], "utf8")).decode("utf8")
    # connect
    sql_conn = pymysql.connect(host=settings["sql_host"],
                               port=int(settings["sql_port"]),
                               user=settings["sql_user"],
                               password=sql_passwd,
                               database=settings["sql_db"])
    return sql_conn


def run(config, instrum, wait,
        output=False, sql=False, header=True, quiet=False, debug=False):
    """ start the emonitor server and output to sqlite database.
    """
    tty = sys.stdout.isatty()
    settings = parse_settings(config, instrum)
    columns = get_columns(settings)
    if debug and tty:
        print("DEBUG enabled")
    try:
        device = get_device(settings, instrum, debug=debug)
        # sqlite output
        db = None
        if output:
            db = get_sqlite(settings, columns)
        # sql output
        sql_conn = None
        if sql:
            sql_conn = get_sql(settings)
        # header
        if tty:
            if not quiet:
                print("Starting emonitor. Use Ctrl-C to stop. \n")
                if header:
                    test = tuple(device.read_data())
                    if debug:
                        print(test)
                    str_width = len(str(test[0]))
                    print(columns[0].rjust(19) + " \t",
                          "\t ".join([col.rjust(str_width) for col in columns[1:]]))
        elif header:
            print(",".join(columns))
        # start server
        while True:
            ## read data
            values = tuple(device.read_data())
            <fix/>is_null = all([v is None for v in values])</fix>
            ## output
            if not is_null:
                values = (time.strftime("%Y-%m-%d %H:%M:%S"), ) + values
                if tty:
                    if not quiet:
                        <fix/>val_str = tuple(str(v).replace("None", "NULL".rjust(str_width)) for v in values)
                        print("\t ".join(val_str))</fix>
                else:
                    <fix/>val_str = tuple(str(v).replace("None", "NULL") for v in values)
                    print(",".join(val_str))</fix>
                if output:
                    db_insert(db, TABLE, columns, values, debug=debug)
                if sql:
                    try:
                        if not sql_conn.open:
                            # attempt to reconnect
                            sql_conn.connect()
                        <fix/>sql_insert(sql_conn, settings["sql_table"], columns, values, debug=debug)</fix>
                    except:
                        warnings.warn("SQL connection failed")
            time.sleep(wait)
    except KeyboardInterrupt:
        if tty and not quiet:
            print("\nStopping emonitor.")
    finally:
        device.close()
        if db is not None:
            db.close()
        if sql_conn is not None:
            sql_conn.close()
