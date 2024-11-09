#!/usr/bin/python3

import logging
from datetime import datetime
from queue import Queue

import MySQLdb

from htmlscraper.listing import Listing

logger = logging.getLogger(__name__)

class MySQLConnectionPool:
    FIELDS_DICT = {'id': 'id', 'title': 'title', 'pubdate': 'publish_date',
                   'loc_id': 'location_id', 'addr': 'address', 'bedrooms': 'bedroom_qty',
                   'bathrooms': 'bathroom_qty', 'price': 'price', 'pet_friendly': 'pet_friendly_flag',
                   'furnished': 'furnished_flag', 'urgent': 'urgent_flag',
                   'url': 'url', 'size': 'size', 'desc': 'description'}

    def __init__(self, hostaddr, usr, pwd, dbname, size):
        logger.info('Initializing an instance of MySQLConnectionPool')
        logger.debug('Type checking for host address, username, password, database name and pool size')
        if type(hostaddr) != str:
            raise TypeError('hostaddr has to be a str')
        if type(usr) != str:
            raise TypeError('usr has to be a str')
        if type(pwd) != str:
            raise TypeError('pwd has to be a str')
        if type(dbname) != str:
            raise TypeError('dbname has to be a str')
        logger.debug('All type checks passed')

        logger.info('Initializing class variables')
        # save MySQL server authentication
        self._hostaddr = hostaddr
        self._usr = usr
        self._pwd = pwd
        self._dbname = dbname

        logger.info('Initializing MySQL connection pool')
        # initiate an empty Queue of required size
        self._pool = Queue(size)

        # fill the pool up
        for i in range(size):
            <vul/>self._pool.put(MySQLdb.connect(hostaddr, usr, pwd, dbname), block=False)</vul>
        logger.info('Initialized MySQL connection pool')

    # return an available connection from the pool
    def get_connection(self):
        logger.debug('Retrieving connection from the pool')
        db = self._pool.get()

        logger.debug('Type checking connection')
        if not isinstance(db, MySQLdb.connections.Connection):
            raise TypeError('A connection has to be of type MySQLdb.connections.Connection')
            return -1

        logger.info('Successful MySQL connection get request')
        return db

    # queue a connection to the pool
    def put_connection(self, connection):
        logger.debug('Type checking connection')
        if not isinstance(connection, MySQLdb.connections.Connection):
            raise TypeError('A connection has to be of type MySQLdb.connections.Connection')
            return -1

        self._pool.put_nowait(connection)
        self._pool.task_done()
        logger.info('Successful MySQL connection put request')
        return 0

    # close all connections
    def clear_pool(self):
        logger.info('Closing the MySQL connection pool (id {})'.format(id(self)))
        while not self._pool.empty():
            db = self._pool.get()
            if not isinstance(db, MySQLdb.connections.Connection):
                raise TypeError('A non-Connection object found in the connection pool')
            db.close()
            self._pool.task_done()
        logger.info('Closed all connections in the MySQL connection pool (id {})'.format(id(self)))
        return 0

    # generate sql from a listing
    <vul/>def gen_sql_insert(self, listing, cat_id):</vul>
        logger.debug('Generating MySQL command for insertion/update for table c{:d}'.format(cat_id))
        if listing.id < 0:
            logger.error('TypeError: id must be non-negative')
            logger.error('Skipping the Listing')
            return -1
        if type(listing.pubdate) != datetime:
            logger.error('TypeError: pubdate must be a datetime')
            return -1

        sql_cols = """INSERT INTO c{cat_id:d}({id:s}, {url:s}, {loc_id:s}, {title:s}, {pubdate:s}, {desc:s}""".format(
            cat_id=cat_id,
            **self.FIELDS_DICT)
        <vul/>sql_vals = """) VALUES ({id:d}, '{url:s}', {loc_id:d}, '{title:s}', '{pubdate:s}', '{desc:s}'""".format(
            id=listing.id,
            url=listing.url,
            loc_id=listing.loc_id,
            title=listing.title,
            pubdate=listing.pubdate.strftime(
                '%Y-%m-%d %H:%M:%S'),
            desc=listing.description)</vul>

        col_list = [self.FIELDS_DICT['addr'], self.FIELDS_DICT['price'], self.FIELDS_DICT['bedrooms'],
                    self.FIELDS_DICT['bathrooms'], self.FIELDS_DICT['pet_friendly'], self.FIELDS_DICT['furnished'],
                    self.FIELDS_DICT['urgent'], self.FIELDS_DICT['size']]
        <vul/>val_list = [listing.addr, listing.price, listing.bedrooms, listing.bathrooms, listing.pet_friendlly,
                    listing.furnished, listing.urgent, listing.size]
        sql_list = [lambda: "'{:s}'".format(listing.addr), lambda: "{:f}".format(listing.price),
                    lambda: "{:f}".format(listing.bedrooms),
                    lambda: "{:f}".format(listing.bathrooms), lambda: "{:d}".format(int(listing.pet_friendlly)),
                    lambda: "{:d}".format(int(listing.furnished)), lambda: "{:d}".format(int(listing.urgent)),
                    lambda: "{:f}".format(listing.size)]</vul>
        for i in range(len(col_list)):
            if val_list[i] != -1:
                sql_cols += ", " + col_list[i]
                <vul/>sql_vals += ", " + sql_list[i]()</vul>

        <vul/>output = sql_cols + sql_vals + ')'</vul>
        logger.debug('MySQL command generation successful')
        <vul/>return output</vul>

    # add a list of listings into an existing table
    <vul/>def update_table(self, listings, cat_id):</vul>
        logger.info('Updating table c{} in the MySQL database'.format(cat_id))
        logger.info('Requesting for MySQL connection')

        db = self.get_connection()
        cursor = db.cursor()

        for l in listings:
            if not isinstance(l, Listing):
                logger.error('TypeError: Expected a Listing instance')
                logger.error('Skipping this listing')
                continue

            <vul/>logger.debug('Generating SQL command')

            sql = self.gen_sql_insert(l, cat_id)</vul>
            if sql == -1:
                # Failed to generate SQL command
                logger.error('Skipping the listing')
                continue

            try:
                logger.debug('Executing SQL command')
                <vul/>cursor.execute(sql)</vul>
                logger.debug('Committing changes to the database')
                db.commit()
                logger.info('Successfully added a listing to table c{:d}'.format(cat_id))
            except:
                db.rollback()
                logger.error('Failed to add a listing to table c{:d}'.format(cat_id))
                logger.error('Rolled back the database changes')
        return 0
