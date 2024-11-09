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
            <fix/>connection = MySQLdb.connect(hostaddr, usr, pwd, dbname)

            # use utf8 encoder
            connection.set_character_set('utf8')
            connection.cursor().execute('SET NAMES utf8;')
            connection.cursor().execute('SET CHARACTER SET utf8;')
            connection.cursor().execute('SET character_set_connection=utf8;')

            self._pool.put(connection, block=False)</fix>
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
    <fix/>def _gen_sql_insert(self, listing, cat_id):</fix>
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
        <fix/>sql_vals = """) VALUES (%s, %s, %s, %s, %s, %s"""
        sql_pars = [listing.id, listing.url, listing.loc_id, listing.title,
                    listing.pubdate.strftime('%Y-%m-%d %H:%M:%S'), listing.description]</fix>

        col_list = [self.FIELDS_DICT['addr'], self.FIELDS_DICT['price'], self.FIELDS_DICT['bedrooms'],
                    self.FIELDS_DICT['bathrooms'], self.FIELDS_DICT['pet_friendly'], self.FIELDS_DICT['furnished'],
                    self.FIELDS_DICT['urgent'], self.FIELDS_DICT['size']]
        <fix/>val_list = [listing.addr, listing.price, listing.bedroomqty, listing.bathroomqty, listing.pet_friendly_flag,
                    listing.furnished_flag, listing.urgent_flag, listing.size]
</fix>
        for i in range(len(col_list)):
            if val_list[i] != -1:
                sql_cols += ", " + col_list[i]
                <fix/>sql_vals += ", %s"
                sql_pars.append(val_list[i])</fix>

        <fix/>cmd = sql_cols + sql_vals + ')'</fix>
        logger.debug('MySQL command generation successful')
        <fix/>return (cmd, sql_pars)</fix>

    # add a list of listings into an existing table
    <fix/>def update_table(self, listings):
        if not listings:
            logger.warning('Did not update table. listings is empty.')
            return 0

        cat_id = listings[0].cat_id
</fix>
        logger.info('Updating table c{} in the MySQL database'.format(cat_id))
        logger.info('Requesting for MySQL connection')

        db = self.get_connection()
        cursor = db.cursor()

        for l in listings:
            if not isinstance(l, Listing):
                logger.error('TypeError: Expected a Listing instance')
                logger.error('Skipping this listing')
                continue

            <fix/>(sql, pars) = self._gen_sql_insert(l, cat_id)</fix>
            if sql == -1:
                # Failed to generate SQL command
                logger.error('Skipping the listing')
                continue

            try:
                logger.debug('Executing SQL command')
                <fix/>cursor.execute(sql, pars)</fix>
                logger.debug('Committing changes to the database')
                db.commit()
                logger.info('Successfully added a listing to table c{:d}'.format(cat_id))
            except:
                db.rollback()
                print(sql)
                logger.exception('Exception:')
                logger.error('Failed to add a listing to table c{:d}'.format(cat_id))
                logger.error('Listing url: {:s}'.format(l.url))
                logger.error('Rolled back the database changes')
        return 0
