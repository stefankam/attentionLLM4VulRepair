from sqlalchemy import sql

from app import db
from pub import Pub


def fulltext_search_title(query):
    <fix/>query_statement = sql.text("""</fix>
      SELECT id, ts_headline('english', title, query), ts_rank_cd(to_tsvector('english', title), query, 32) AS rank
        <fix/>FROM pub_2018, plainto_tsquery('english', :search_str) query  -- or try plainto_tsquery, phraseto_tsquery, to_tsquery</fix>
        WHERE to_tsvector('english', title) @@ query
        ORDER BY rank DESC
        LIMIT 50;""")

    rows = db.engine.execute(query_statement.bindparams(search_str=query)).fetchall()
    ids = [row[0] for row in rows]
    my_pubs = db.session.query(Pub).filter(Pub.id.in_(ids)).all()
    for row in rows:
        my_id = row[0]
        for my_pub in my_pubs:
            if my_id == my_pub.id:
                my_pub.snippet = row[1]
                my_pub.score = row[2]
    return my_pubs


def autocomplete_phrases(query):
    <fix/>query_statement = sql.text(ur"""
        with s as (SELECT id, lower(title) as lower_title FROM pub_2018 WHERE title iLIKE :p0)</fix>
        select match, count(*) as score from (
            <fix/>SELECT regexp_matches(lower_title, :p1, 'g') as match FROM s</fix>
            union all
            <fix/>SELECT regexp_matches(lower_title, :p2, 'g') as match FROM s</fix>
            union all
            <fix/>SELECT regexp_matches(lower_title, :p3, 'g') as match FROM s</fix>
            union all
            <fix/>SELECT regexp_matches(lower_title, :p4, 'g') as match FROM s</fix>
        ) s_all
        group by match
        order by score desc, length(match::text) asc
        <fix/>LIMIT 50;""").bindparams(
            p0='%{}%'.format(query),
            p1=ur'({}\w*?\M)'.format(query),
            p2=ur'({}\w*?(?:\s+\w+){{1}})\M'.format(query),
            p3=ur'({}\w*?(?:\s+\w+){{2}})\M'.format(query),
            p4=ur'({}\w*?(?:\s+\w+){{3}}|)\M'.format(query)
        )</fix>

    <fix/>rows = db.engine.execute(query_statement).fetchall()</fix>
    phrases = [{"phrase":row[0][0], "score":row[1]} for row in rows if row[0][0]]
    return phrases