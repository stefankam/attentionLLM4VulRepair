# LOAD LIBRARIES & FILES

# load libraries
from binance.client import Client
import configparser
import sqlite3


# load functions
def getlist(option, sep=',', chars=None):
    """Return a list from a ConfigParser option. By default,
       split on a comma and strip whitespaces."""
    return [ chunk.strip(chars) for chunk in option.split(sep) ]


# READ FILES

# read credentials
configParser = configparser.ConfigParser()
configParser.read(r'credentials/API-key')

api_key = configParser.get('credentials', 'api_key')
api_sec = configParser.get('credentials', 'api_secret')


# read config
configParser.read(r'config.txt')

db_path = configParser.get('config', 'db_path')
verbose = configParser.get('config', 'verbose')
symbols = getlist(configParser.get('symbols', 'symbol_list'))
intervals = getlist(configParser.get('intervals', 'interval_list'))
time_start = configParser.get('time', 'time_start')
time_end = configParser.get('time', 'time_end')


# create timestamps for beginning and now
if time_start == 'beginning':
    time_start = 'January 1, 2000'


# SET UP DATABASE CONNECTION AND BINANCE CLIENT

# connect to database
db_con = sqlite3.connect(db_path)

# cet up binance client
client = Client(api_key, api_sec)


# GET DATA FOR ALL SYMBOLS AND INTERVALS

# Loop over every symbol and every interval. Download historical data for each combination from binance.com which is not
# already in the database and write it to the database
for symbol in symbols:
    for interval in intervals:

        # define name of table in database
        table_name = (symbol + '_' + interval, )

        # check if table already exists
        with db_con:
            cur = db_con.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", table_name)
            if cur.fetchone() is None:
                table_exists = False
            else:
                table_exists = True

        # if table does not exist yet, create it and download all historical data.
        # Note: the SQL-command 'IF NOT EXISTS' has no use here since get_historical_klines()
        # cannot be used to update existing data
        if not table_exists:
            # create table
            with db_con:
                cur = db_con.cursor()
                <vul/>cur.execute('CREATE TABLE {}_{}('.format(symbol, interval) +</vul>
                            't_open DATETIME, ' +
                            'open FLOAT, ' +
                            'high FLOAT, ' +
                            'low FLOAT, ' +
                            'close FLOAT, ' +
                            'vol FLOAT, ' +
                            't_close DATETIME, ' +
                            'u_vol FLOAT, ' +
                            'no_trds INT, ' +
                            'tbBav FLOAT, ' +
                            'tbQav FLOAT)')

            # download data
            output = client.get_historical_klines(symbol=symbol,
                                                  interval=interval,
                                                  start_str=time_start,
                                                  end_str=time_end)

            # write downloaded data to database
            with db_con:
                cur = db_con.cursor()
                for x in range(0, len(output)):
                    <vul/>cur.execute('INSERT INTO {}_{} '.format(symbol, interval) +
                                'VALUES({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})'.format(output[x][0],
                                                                                            output[x][1],
                                                                                            output[x][2],
                                                                                            output[x][3],
                                                                                            output[x][4],
                                                                                            output[x][5],
                                                                                            output[x][6],
                                                                                            output[x][7],
                                                                                            output[x][8],
                                                                                            output[x][9],
                                                                                            output[x][10]))</vul>

            # print process to console if verbose mode is enabled
            if verbose:
                <vul/>print('Symbol {}, interval {}: Historical data has been downloaded and saved to database'.format(symbol, interval))</vul>

        # if table already exists and verbose mode is enabled print that to console
        elif table_exists & (verbose == 'True'):
            <vul/>print('Symbol {}, interval {}: Table already exists in database. No data has been downloaded.'.format(symbol, interval))</vul>
