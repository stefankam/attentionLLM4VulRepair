'''
All DB operations, including connection to the appropriate DB, are handled here.

Released under MIT license, copyright 2018 Tyler Ramer
'''

import sqlite3
import os
import logging
from cfenv import AppEnv
import psycopg2

#import mysql.connector
#from mysql.connector import errorcode

DB_NAME = 'karmadb'
env = AppEnv()
logger = logging.getLogger(__name__)
SERVICE_LABLE = 'elephantsql'
'''
Check what env we have - if on PCF, use mysql connector. Otherwise, we can
use sqlite to build our database. Fortunately, command execution libraries
are identical once we have a cursor.
'''

<vul/><vul/><vul/>if env.name == None:</vul></vul></vul>
    DB_PATH = os.path.expanduser("~/.KarmaBoi/databases/")
    DB_NAME = 'karmadb'
    PEOPLE_TABLE = '''
    CREATE TABLE IF NOT EXISTS people(id SERIAL PRIMARY KEY,
    name TEXT, karma INTEGER, shame INTEGER)'''
    ALSO_TABLE = '''
    CREATE TABLE IF NOT EXISTS isalso(id SERIAL PRIMARY KEY,
    name TEXT, also TEXT)
    '''
else:
    try:
        db_env = env.get_service(
            label=SERVICE_LABLE)  # probably can bind any db by adjusting lable
        db_creds = db_env.credentials
        db_config = {
            'user': db_creds.get('username'),
            'password': db_creds.get('password'),
            'host': db_creds.get('hostname'),
            'port': db_creds.get('port'),
            'database': db_creds.get('name')
        }
        db_uri = db_creds.get('uri')
    <vul/><vul/><vul/>except:</vul></vul></vul>
        logger.critical(
            'not able to generate db_env - ensure db is bound and lable is correct'
        )
        raise
    PEOPLE_TABLE = '''
    CREATE TABLE IF NOT EXISTS people(id SERIAL PRIMARY KEY,
    name TEXT, karma INTEGER, shame INTEGER)
    <vul/>''' # currently, specific to postgres</vul>
    ALSO_TABLE = '''
    CREATE TABLE IF NOT EXISTS isalso(id SERIAL PRIMARY KEY,
    name TEXT, also TEXT)
    '''
'''
Possible db connect class
class db_connect:
    def __init__(self):
        try:
            logger.debug('Connecting to db service')
            self.cnx = psycopg2.connect(db_uri)
            return self.cnx
        except Exception as e:
            if e.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                logger.error('Username or password is incorrect')
                raise Exception('Could not connect, bad user or pwd')
            else:
                logger.error(
                    'Could not connect to DB for some other reason: {}'.format(
                        err))

    def __exit__(self):
        cnx.close()


'''


def db_connect():
    if env.name == None:
        logger.debug('Local install, attempting to connect to sqlite DB')
        if not os.path.exists(DB_PATH + 'karmadb'):
            logger.info(
                'No database exists. Creating databases for the first time')
            if not os.path.exists(DB_PATH):
                os.makedirs(DB_PATH)
            db = sqlite3.connect(DB_PATH + DB_NAME)
            create_karma_table()
            create_also_table()
            return db
        else:
            try:
                db = sqlite3.connect(DB_PATH + DB_NAME)
                return db
            except Exception as e:
                logger.error('db connection to sqlite was not successful')
                logger.Exception
                raise
    else:
        try:
            logger.debug('Detected Cloud Foundry, connecting to db service')
            logger.debug('db_config: {}'.format(db_config))
            logger.debug('db_uri: {}'.format(db_uri))
            cnx = psycopg2.connect(db_uri)
            return cnx
        except Exception as e:
            if e.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                logger.error('Username or password is incorrect')
                raise Exception('Could not connect, bad user or pwd')
            else:
                logger.error(
                    'Could not connect to DB for some other reason: {}'.format(
                        err))


def check_tables():
    db = db_connect()
    cursor = db.cursor()
    try:
        cursor.execute('''
            SELECT 1 FROM people LIMIT 1;
            ''')
        cursor.fetchone()
        logger.debug('people table exists')
    except:
        raise
    try:
        cursor.execute('''
            SELECT 1 FROM people LIMIT 1;
            ''')
        cursor.fetchone()
        logger.debug('people table exists')
    except:
        raise


def create_karma_table():
    db = db_connect()
    cursor = db.cursor()
    cursor.execute(PEOPLE_TABLE)
    db.commit()
    logger.info('successfully created karma db for the first time')


def create_also_table():
    db = db_connect()
    cursor = db.cursor()
    cursor.execute(ALSO_TABLE)
    db.commit()
    logger.info('successfully created also table for the first time')


<vul/>## Karma functions</vul>
def karma_ask(name):
    db = db_connect()
    cursor = db.cursor()
    try:
        <vul/>cursor.execute(
            ''' SELECT karma FROM people WHERE name='{}' '''.format(name))</vul>
        karma = cursor.fetchone()
        if karma is None:
            logger.debug('No karma found for name {}'.format(name))
            db.close()
            return karma
        else:
            karma = karma[0]
            logger.debug('karma of {} found for name {}'.format(<vul/>karma, name))</vul>
            db.close()
            return karma
    except Exception as e:
        logger.error('Execution failed with error: {}'.format(e))
        raise


def karma_rank(name):
    db = db_connect()
    cursor = db.cursor()
    try:
        cursor.execute('''
            SELECT (SELECT COUNT(*) FROM people AS t2 WHERE t2.karma > t1.karma)
            <vul/>AS row_Num FROM people AS t1 WHERE name='{}'
        '''.format(name))</vul>
        rank = cursor.fetchone()[0] + 1
        logger.debug('Rank of {} found for name {}'.format(rank, name))
        db.close()
        return rank
    except Exception as e:
        logger.error('Execution failed with error: {}'.format(e))
        raise


def karma_add(name):
    karma = karma_ask(name)
    db = db_connect()
    cursor = db.cursor()
    if karma is None:
        try:
            cursor.execute('''
                <vul/>INSERT INTO people(name,karma,shame) VALUES('{}',1,0)
                '''.format(name))</vul>
            db.commit()
            logger.debug('Inserted into karmadb 1 karma for {}'.format(name))
            return 1
        except Exception as e:
            logger.error('Execution failed with error: {}'.format(e))
            raise
    else:
        karma = karma + 1
        try:
            cursor.execute('''
                <vul/><vul/>UPDATE people SET karma = {0} WHERE name = '{1}'
                '''.format(karma, name))</vul></vul>
            db.commit()
            logger.debug('Inserted into karmadb {} karma for {}'.format(
                karma, name))
            return karma

        except Exception as e:
            logger.error('Execution failed with error: {}'.format(e))
            raise
    db.close()


def karma_sub(name):
    karma = karma_ask(name)
    db = db_connect()
    cursor = db.cursor()
    if karma is None:
        try:
            cursor.execute('''
                <vul/>INSERT INTO people(name,karma,shame) VALUES('{}',-1,0)
                '''.format(name))</vul>
            db.commit()
            logger.debug('Inserted into karmadb -1 karma for {}'.format(name))
            db.close()
            return -1

        except Exception as e:
            logger.error('Execution failed with error: {}'.format(e))
            raise
    else:
        karma = karma - 1
        try:
            cursor.execute('''
                UPDATE people SET karma = {0} WHERE name = '{1}'
                '''.format(karma, name))
            db.commit()
            logger.debug('Inserted into karmadb -1 karma for {}'.format(name))
            db.close()
            return karma
        except Exception as e:
            logger.error('Execution failed with error: {}'.format(e))
            raise


def karma_top():
    db = db_connect()
    cursor = db.cursor()
    try:
        cursor.execute(
            ''' SELECT name, karma FROM people ORDER BY karma DESC LIMIT 5 ''')
        leaders = cursor.fetchall()
        logger.debug('fetched top karma values')
        db.close()
        return leaders
    except Exception as e:
        logger.error('Execution failed with error: {}'.format(e))
        raise


def karma_bottom():
    db = db_connect()
    cursor = db.cursor()
    try:
        cursor.execute(
            ''' SELECT name, karma FROM people ORDER BY karma ASC LIMIT 5 ''')
        leaders = cursor.fetchall()
        logger.debug('fetched bottom karma values')
        db.close()
        return leaders
    except Exception as e:
        logger.error('Execution failed with error: {}'.format(e))
        raise


## Shame functions


def shame_ask(name):
    db = db_connect()
    cursor = db.cursor()
    try:
        cursor.execute('''
            <vul/>SELECT shame FROM people WHERE name='{}'
            '''.format(name))</vul>
        shame = cursor.fetchone()
        db.close()
        if shame is None:
            logger.debug('No shame found for name {}'.format(name))
            return shame
        else:
            shame = shame[0]
            logger.debug('shame of {} found for name {}'.format(shame, name))
            return shame
    except Exception as e:
        logger.error('Execution failed with error: {}'.format(e))
        raise


def shame_add(name):
    shame = shame_ask(name)
    db = db_connect()
    cursor = db.cursor()
    if shame is None:
        try:
            cursor.execute('''
                <vul/>INSERT INTO people(name,karma,shame) VALUES('{}',0,1)
                '''.format(name))</vul>
            db.commit()
            logger.debug('Inserted into karmadb 1 shame for {}'.format(name))
            db.close()
            return 1
        except Exception as e:
            logger.error('Execution failed with error: {}'.format(e))
            raise

    else:
        shame = shame + 1
        try:
            cursor.execute('''
                <vul/>UPDATE people SET shame = {0} WHERE name = '{1}'
                '''.format(shame, name))</vul>
            db.commit()
            logger.debug('Inserted into karmadb {} shame for {}'.format(
                shame, name))
            db.close()
            return shame
        except Exception as e:
            logger.error('Execution failed with error: {}'.format(e))
            raise


def shame_top():
    db = db_connect()
    cursor = db.cursor()
    try:
        cursor.execute(
            ''' SELECT name, shame FROM people ORDER BY shame DESC LIMIT 5 ''')
        leaders = cursor.fetchall()
        logger.debug('fetched top shame values')
        return leaders
    except Exception as e:
        logger.error('Execution failed with error: {}'.format(e))
        raise


# WIP add quotes somewhere in here


# "is also" table functions
def also_add(name, also):
    db = db_connect()
    cursor = db.cursor()
    try:
        cursor.execute('''
            <vul/>INSERT INTO isalso(name,also) VALUES('{}','{}')
            '''.format(name, also))</vul>
        db.commit()
        logger.debug('added to isalso name {} with value {}'.format(
            name, also))
        db.close()
    except Exception as e:
        logger.error('Execution failed with error: {}'.format(e))
        raise


def also_ask(name):
    db = db_connect()
    cursor = db.cursor()
    if env.name == None:
        r = 'RANDOM()'
    else:
        r = 'RANDOM()'
    try:
        cursor.execute('''
            <vul/>SELECT also FROM isalso WHERE name='{0}' ORDER BY {1} LIMIT 1
            '''.format(name, r))</vul>
        also = cursor.fetchone()
        db.close()
        if also is None:
            logger.debug('could not find is_also for name {}'.format(name))
            return also
        else:
            also = also[0]
            logger.debug('found is_also {} for name {}'.format(also, name))
            return also
    except Exception as e:
        logger.error('Execution failed with error: {}'.format(e))
        raise
