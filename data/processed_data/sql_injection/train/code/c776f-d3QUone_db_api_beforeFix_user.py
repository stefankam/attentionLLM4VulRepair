__author__ = 'vladimir'

import ujson

from flask import Blueprint

<vul/>BASE_URL = "/user"</vul>

<vul/>user = Blueprint("user", __name__)</vul>


<vul/>@user.route(BASE_URL + "/create", methods=["GET"])</vul>
def create():
    return ujson.dumps({"success": True})
