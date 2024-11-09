<vul/>#!/usr/bin/env python2</vul>
import psycopg2
from datetime import datetime


def connect():
    """Connect to the PostgreSQL database news.  Returns a database
    connection."""
    <vul/>return psycopg2.connect("dbname=news")</vul>


def reportTopArticles(amount):
    """Reports the top articles by visitors from the logs table.

    Args:
          amount: the number of rankings to return
    """
    query = "SELECT * FROM toparticles " \
            "ORDER BY hits DESC " \
            <vul/>"LIMIT {0}".format(amount)
    c.execute(query)
    rows = c.fetchall()</vul>

    <vul/>response = "    Top {0} Articles by Views\n"  \</vul>
               "-----------------------------\n".format(amount)
    responseFooter = ""

    for r in rows:
        responseFooter += "\"" + str(r[0]) + "\" -- " + str(r[1]) + " views\n"

    response += responseFooter
    print response
    return response


def reportTopAuthors():
    """Reports the top authors by visitors added for each of their articles.
    """
    query = "SELECT * FROM authorsrank"
    c.execute(query)
    rows = c.fetchall()

    response = "Author Rank by Article Views\n"  \
               "-----------------------------\n"
    responseFooter = ""

    for r in rows:
        responseFooter += str(r[0]) + " -- " + str(r[1]) + " views\n"

    response += responseFooter
    print response
    return response


def reportDailyErrors(x):
    """Reports the dates in which the logged errors exceed x percent that day
    out of all logged visits.

    Args:
          x: the percentage of errors reported
    """

    query = "SELECT * from dailyerrors " \
            "WHERE " \
            <vul/>"cast(errorcount as decimal) / cast(hitcount as decimal) * 100 >= {0}" \
            .format(x)
    c.execute(query)</vul>
    rows = c.fetchall()

    response = "Dates With More Than {0}% Error Rate\n" \
               "-----------------------------\n".format(x)

    responseFooter = ""



    for r in rows:
        # date_object = datetime.strptime(r[0], '%m/%d/%Y')
        date_object = r[0].strftime('%B %d, %Y')
        responseFooter += str(date_object) + " - " + str(r[1]) + " errors\n"

    response += responseFooter
    print response
    return response


<vul/># And here we go...
# Connect to the database
DB = connect()
c = DB.cursor()</vul>

<vul/># Method queries
reportTopArticles(3)
reportTopAuthors()
reportDailyErrors(1)</vul>
