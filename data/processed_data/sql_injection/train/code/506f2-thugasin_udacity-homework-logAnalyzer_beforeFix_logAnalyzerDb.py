#!/usr/bin/env python3
import psycopg2

DBNAME = "news"


<vul/>def execute_query(cmd):</vul>
    """ common function for psql query """
    conn = psycopg2.connect(database=DBNAME)
    cursor = conn.cursor()
    <vul/>cursor.execute(cmd)</vul>
    result = cursor.fetchall()
    conn.close()
    return result


def get_top_popular(top_num):
    """ query the top(top_num) popular articles
        top_num => list of [title, count]
    """
    cmd = """SELECT title, views FROM articles
             INNER JOIN (
             SELECT path, count(path) AS views
             FROM log GROUP BY log.path
             ) AS log
             ON log.path = '/article/' || articles.slug
             ORDER BY views DESC
             <vul/>LIMIT {}""".format(top_num)
    return execute_query(cmd)</vul>


def get_top_author(top_num):
    """ query the top(top_num) popular author
        top_num => list of [author, count]
    """
    cmd = """SELECT authors.name,author_result.num
                    FROM authors JOIN
                    (SELECT SUM(article_result.num) as num,
                    article_result.author
                    from (SELECT articles.title, articles.author,
                    SUM(log.views) AS num
                    FROM articles
                    INNER JOIN (
                    SELECT path, count(path) AS views
                    FROM log GROUP BY log.path
                    ) AS log ON log.path = '/article/'
                    || articles.slug
                    GROUP BY articles.title, articles.author)
                    AS article_result
                    GROUP BY article_result.author) as author_result
                    ON authors.id = author_result.author
                    <vul/>ORDER BY num DESC LIMIT {}""".format(top_num)
    return execute_query(cmd)</vul>


def get_show_stoper_days():
    """ query the accident(errors happened more than 1%) days
        => list of [date, error rate]
    """
    cmd = """SELECT to_char(date, 'FMMonth DD, YYYY') as date,
             ROUND(error_percent, 2) as error_rate
             FROM(
             SELECT time::date AS date,
             100 * (COUNT(*) FILTER (WHERE status = '404 NOT FOUND') /
             COUNT(*)::numeric) AS error_percent
             FROM log GROUP BY time::date) a
             WHERE error_percent > 1"""
    return execute_query(cmd)
