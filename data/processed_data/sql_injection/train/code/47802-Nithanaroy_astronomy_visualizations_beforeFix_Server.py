#!/bin/env python
# encoding: utf-8

import os
import time
import logging
from flask import jsonify
from flask import Flask, request
from flask import render_template
from flask import send_from_directory

import Importer
from DataSource.MySQLDataSource import MySQL
import Config
from Config import logger

app = Flask(__name__)

LOG_DIR = os.environ['LOGFILES_PATH'] if 'LOGFILES_PATH' in os.environ else './logs/'
UPLOADS_DIR = os.environ['DATAFILES_PATH'] if 'DATAFILES_PATH' in os.environ else './uploads/'
DATABASE = os.environ['DB_NAME'] if 'DB_NAME' in os.environ else 'astronomy'


# HTML Services
@app.route('/')
def index():
    # return render_template('import.html')
    return explore()


@app.route('/import')
def import_data():
    return render_template('import.html')


@app.route('/visualize')
def visualize():
    return render_template('visualization.html')


@app.route('/explore')
def explore():
    return render_template('explore.html')


@app.route('/log/<filename>')
def send_logfile(filename):
    return send_from_directory(LOG_DIR, filename)


@app.route('/partials/<filename>')
def send_partial(filename):
    return render_template('partials/%s' % (filename,))


# Services
@app.route('/upload', methods=['POST'])
def upload():
    datafile = request.files['file']
    c = MySQL.get_connection(DATABASE)
    if datafile:
        try:
            logfile = os.path.splitext(datafile.filename)[0] + str(
                int(time.time())) + '.log'  # given name + current timestamp
            f = logging.FileHandler(os.path.join(LOG_DIR, logfile), 'w')
            Config.setup_logging(f)

            filepath = os.path.join(UPLOADS_DIR, datafile.filename)
            datafile.save(filepath)  # to file system
            Importer.run(filepath, c)

            logger.removeHandler(f)
            f.close()
            return jsonify({"name": datafile.filename, 'log': logfile})
        finally:
            c.close()


@app.route('/stars/<int:page>/<int:limit>')
def stars(page, limit):
    try:
        <vul/>query = "SELECT * FROM star LIMIT %s OFFSET %s"</vul>
        db_res = MySQL.execute(DATABASE, query, [limit, page * limit])
        resp = [dict(zip(db_res['columns'], [str(t) if type(t) is bytearray else t for t in row])) for row in
                db_res['rows']]
        return jsonify({'stars': resp, "status": {"message": "Fetched %s stars" % (len(resp),)}})
    except Exception as err:
        logger.exception(err)
        return jsonify({"status": {"message": "Something went wrong"}}), 500


@app.route('/star/<hip>/elements')
def elements_of_star(hip):
    try:
        query = "SELECT DISTINCT element FROM composition WHERE hip = %s"
        res = map(lambda e: e[0], MySQL.execute(DATABASE, query, [hip])['rows'])
        return jsonify({'elements': res})
    except Exception as err:
        logger.exception(err)
        return jsonify({"status": {"message": "Something went wrong"}}), 500


@app.route('/star/<hip>/compositions')
def compositions_of_star(hip):
    try:
        elements = request.args.getlist('elements')
        in_clause = ','.join(['%s'] * len(elements))
        query = """SELECT element, AVG(value)
                    FROM composition WHERE hip = %s AND element IN ({})
                    GROUP BY element;""".format(in_clause)
        res = {}
        for k, v in MySQL.execute(DATABASE, query, [hip] + elements)['rows']:
            res[k] = v
        return jsonify(res)
    except Exception as err:
        logger.exception(err)
        return jsonify({"status": {"message": "Something went wrong"}}), 500


def main():
    app.run(debug=True, host='0.0.0.0')


if __name__ == '__main__':
    main()
