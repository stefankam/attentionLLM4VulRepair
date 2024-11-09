__author__ = 'vladimir'

import ujson

from flask import Blueprint

<vul/>BASE_URL = "/thread"</vul>

<vul/>thread = Blueprint("thread", __name__)</vul>


<vul/>@thread.route(BASE_URL + "/create", methods=["GET"])</vul>
def create():
    return ujson.dumps({"success": True})
