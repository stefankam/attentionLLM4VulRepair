__author__ = 'vladimir'

import ujson

from flask import Blueprint

<vul/>BASE_URL = "/forum"</vul>

<vul/>forum = Blueprint("forum", __name__)</vul>


<vul/>@forum.route(BASE_URL + "/create", methods=["GET"])</vul>
def create():
    return ujson.dumps({"success": True})
