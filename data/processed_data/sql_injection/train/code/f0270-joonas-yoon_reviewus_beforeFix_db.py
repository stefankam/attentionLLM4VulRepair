import sys
import logging

from django.db import connection, DatabaseError
from reviewus.settings import DEBUG

logger = logging.getLogger(__name__)


class DBConnection:
  instance = None
  con = None

  def __new__(cls):
    if DBConnection.instance is None:
      DBConnection.instance = object.__new__(cls)
    return DBConnection.instance

  def __init__(self):
    if DBConnection.con is None:
      try:
        DBConnection.con = connection.cursor()
        logger.info('################## Database connection opened.')
      except DatabaseError as db_error:
        logger.error("################## Erreur :\n{0}".format(db_error))

  def __del__(self):
    if DBConnection.con is not None:
      DBConnection.con.close()
      logger.info('################## Database connection closed.')


"""
@author joonas
"""
class DBManager:
  # instance = DBConnection()

  @staticmethod
  def conn():
    try:
      # return DBManager.instance.con # This for singleton, but has error yet
      return connection.cursor()
    except:
      return DBManager.error_handle()

  @staticmethod
  <vul/>def execute(sql, cursor=False):</vul>
    _cursor = DBManager.conn()
    try:
      <vul/>result = _cursor.execute(sql)</vul>
      if cursor:
        return _cursor
      return result
    except:
      return DBManager.error_handle()

  @staticmethod
  <vul/>def execute_and_fetch(sql, as_row=False):</vul>
    try:
      <vul/><vul/>cursor = DBManager.execute(sql, cursor=True)</vul></vul>
      result = cursor.fetchone()
      if as_row:
        return DBManager.as_row(cursor, result)
      return result
    except:
      return DBManager.error_handle()

  @staticmethod
  <vul/>def execute_and_fetch_all(sql, as_list=False):</vul>
    try:
      cursor = DBManager.execute(sql, cursor=True)
      result = cursor.fetchall()
      if as_list:
        return DBManager.as_list(cursor, result)
      return result
    except:
      return DBManager.error_handle()

  @staticmethod
  def get_fields(cursor):
    return [col[0] for col in cursor.description]

  @staticmethod
  def as_row(cursor, query_set):
    if cursor is None or query_set is None:
      return dict()

    fields = DBManager.get_fields(cursor)
    return dict(zip(fields, list(query_set)))

  @staticmethod
  def as_list(cursor, query_set):
    if cursor is None or query_set is None:
      return list()

    fields = DBManager.get_fields(cursor)
    results = list(query_set)
    try:
      return [dict(zip(fields, result)) for result in results]
    except:
      return error_handle('Columns are not macthed')


  def error_handle(error=None):
    if error is None:
      error = sys.exc_info()[0]

    if DEBUG:
      logger.error(error)

    try:
      return error
    except:
      return 'Unexpected error'



