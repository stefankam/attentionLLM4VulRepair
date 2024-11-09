<fix/>#!/usr/bin/env python</fix>
import psycopg2
from datetime import datetime
import sys


def connect():
    """Connect to the PostgreSQL database news.  Returns a database
    connection."""
    <fix/>try:
        return psycopg2.connect("dbname=news")
    except psycopg2.Error, e:
        print 'Error in connecting. Make sure the database exists. ' \
            'Now exiting application.'
        sys.exit()</fix>


def reportTopArticles(amount):
    """Reports the top articles by visitors from the logs table.

    Args:
          amount: the number of rankings to return
    """
    query = "SELECT * FROM toparticles " \
            "ORDER BY hits DESC " \
            <fix/>"LIMIT %s"</fix>

    <fix/>c.execute(query, (amount,))
    rows = c.fetchall()
    response = "    Top {0} Articles by Views\n" \</fix>
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
            <fix/>"cast(errorcount as decimal) / cast(hitcount as decimal) " \
            "* 100 > %s"
    c.execute(query, (x,))</fix>
    rows = c.fetchall()

    response = "Dates With More Than {0}% Error Rate\n" \
               "-----------------------------\n".format(x)
    responseFooter = ""

    for r in rows:
        date_object = r[0].strftime('%B %d, %Y')
        responseFooter += str(date_object) + " - " + str(r[1]) + " errors\n"

    response += responseFooter
    print response
    return response


<fix/>def main():
    # Method queries
    reportTopArticles(3)
    reportTopAuthors()
    reportDailyErrors(1)
</fix>

<fix/>if __name__ == '__main__':
    # And here we go...
    # Connect to the database
    DB = connect()
    c = DB.cursor()
    main()</fix>
