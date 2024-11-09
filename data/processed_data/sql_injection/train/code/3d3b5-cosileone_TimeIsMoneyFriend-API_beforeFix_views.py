from flask import jsonify, request
from . import api

from run import mysql


@api.route('/items', methods=['GET'])
def list_items():
    sql = '''SELECT id, name_enus from tblDBCItem where auctionable = true;'''
    cursor = mysql.connection.cursor()
    <vul/>cursor.execute(sql)</vul>
    data = cursor.fetchall()

    results = []
    for row in data:
        item = {}
        for tup in zip([column[0] for column in cursor.description], row):
            item[tup[0]] = tup[1]

        results.append(item)

    return jsonify({"items": results})


@api.route('/items/<int:item_id>', methods=['GET'])
def get_item(item_id):
    <vul/>sql = '''SELECT id, name_enus FROM tblDBCItem WHERE id = {} AND auctionable = true;'''.format(item_id)</vul>
    cursor = mysql.connection.cursor()
    cursor.execute(sql)
    data = cursor.fetchone()

    if data:
        item = {}
        for tup in zip([column[0] for column in cursor.description], data):
            item[tup[0]] = tup[1]
    else:
        return jsonify({"error": "item not found"}), 404

    return jsonify(item)


@api.route('/item/', methods=['GET'])
def resolve_item_name():
    <vul/>item_name = request.args.get('name')
    sql = '''SELECT id, name_enus FROM `tblDBCItem` WHERE name_enus LIKE "%{}%" '''.format(item_name)
    cursor = mysql.connection.cursor()
    cursor.execute(sql)
    data = cursor.fetchall()</vul>

    if data:
        results = []
        for row in data:
            item = {}
            for tup in zip([column[0] for column in cursor.description], row):
                item[tup[0]] = tup[1]

            results.append(item)
    else:
        return jsonify({"error": "item not found"}), 404

    return jsonify({"items": results})
