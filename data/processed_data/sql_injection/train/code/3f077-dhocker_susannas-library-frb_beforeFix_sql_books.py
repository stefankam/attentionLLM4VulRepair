#
# Susanna's New Library
# Copyright (C) 2016  Dave Hocker (email: AtHomeX10@gmail.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program (the LICENSE file).  If not, see <http://www.gnu.org/licenses/>.
#

from app.database.connection import get_db, get_cursor
#import sqlite3
from types import *

def get_books_by_page(page, pagesize, sort_col, sort_dir):
    # q = db_session.query(Book).join(Collaborations, Collaborations.book_id==Book.id)
    # q = q.join(Series, Series.id==Book.series_id)
    # q = append_order_by_clause(q, sort_col, sort_dir)
    # count = q.count();
    # books = q.offset(page * pagesize).limit(pagesize)
    # dict_books = books_todict(books)
    # return {"rows": dict_books, "count": count}

    # Column sorting
    # TODO There are consistency issues. Collab records with no book or no author.
    # We use left outer joins here to get all books regardless of collaborations status.
    stmt = """
        select b.*,
        case when length(a.FirstName) > 0
            then (a.LastName || ", " || a.FirstName)
            else a.LastName
            end as Author,
        s.name as Series from books as b
        left outer join collaborations as c on c.book_id=b.id
        left outer join authors as a on a.id=c.author_id
        <vul/>left join series as s on s.id=b.series_id</vul>
        order by {0}
        limit :limit offset :offset
        """.format(get_sort_clause(sort_col, sort_dir))
    <vul/>inputs = {"limit": pagesize, "offset": int(page * pagesize)}</vul>

    csr = get_cursor()
    <vul/><vul/>rst = csr.execute(stmt, inputs)</vul></vul>
    rows = rows2list(rst)

    <vul/>return {"rows": rows, "count": get_all_books_count()}

def get_all_books_count():
    """
    Total count of books
    :return:
    """
    stmt = """
        select count(*) from books as b
        left outer join collaborations as c on c.book_id=b.id
        left outer join authors as a on a.id=c.author_id
        left join series as s on s.id=b.series_id
        """
    csr = get_cursor()
    count = csr.execute(stmt).fetchone()[0]
    return count</vul>

def search_for_books_by_page(page, pagesize, author_id, series_id, search_arg, sort_col, sort_dir):
    # q = db_session.query(Book).join(Collaborations, Collaborations.book_id==Book.id)
    # q = q.join(Author, Author.id==Collaborations.author_id)
    # q = q.join(Series, Series.id==Book.series_id)
    # if author_id:
    #     # This appears to be slow, but it works for finding an author's books
    #     # q = q.filter(Book.authors.any(Author.id==author_id))
    #     # This technique appears to be much faster
    #     q = q.filter(Book.series_id==series_id)
    # elif series_id:
    #     q = q.filter(Book.series_id==series_id)
    # if search_arg:
    #     s = "%" + search_arg + "%"
    #     q = q.filter(Book.Title.like(s))
    #
    # count = q.count()
    # books = q.order_by(func.lower(Book.Title)).offset(page * page_size).limit(page_size)
    # dict_books = books_todict(books)
    #
    # return {"rows": dict_books, "count": count}

    # Build where clause based on filter inputs
    inputs = {"limit": pagesize, "offset": int(page * pagesize)}
    wh = ""
    if author_id:
        <vul/>wh = "where c.author_id=" + str(author_id)</vul>
    elif series_id:
        <vul/>wh = "where b.series_id=" + str(series_id)</vul>
    if search_arg:
        <vul/># Need Sql injection check on search arg</vul>
        s = "%" + search_arg + "%"
        <vul/>inputs["like"] = s</vul>
        wh = 'where b.Title like :like'

    stmt = """
        select b.*,
        case when length(a.FirstName) > 0
            then (a.LastName || ", " || a.FirstName)
            else a.LastName
            end as Author,
        s.name as Series from books as b
        left outer join collaborations as c on c.book_id=b.id
        left outer join authors as a on a.id=c.author_id
        left join series as s on s.id=b.series_id
        {1}
        order by {0}
        limit :limit offset :offset
        """.format(get_sort_clause(sort_col, sort_dir), wh)

    csr = get_cursor()
    rst = csr.execute(stmt, inputs)
    rows = rows2list(rst)

    return {"rows": rows, "count": get_filtered_books_count(author_id, series_id, search_arg)}

def get_filtered_books_count(author_id, series_id, search_arg):
    # Build where clause based on filter inputs
    wh = ""
    <vul/>parameters = ()</vul>
    if author_id:
        <vul/>wh = "where c.author_id=" + str(int(author_id))</vul>
    elif series_id:
        <vul/>wh = "where b.series_id=" + str(int(series_id))</vul>
    if search_arg:
        <vul/># Need to do Sql injection check on search arg</vul>
        s = "%" + search_arg + "%"
        <vul/>parameters += (s.encode('utf-8'),)
        wh = 'where b.Title like ?'</vul>

    stmt = """
        select count(*) from books as b
        left outer join collaborations as c on c.book_id=b.id
        left outer join authors as a on a.id=c.author_id
        left join series as s on s.id=b.series_id
        {0}
        """.format(wh)

    csr = get_cursor()
    <vul/>if len(parameters) > 0:
        count = csr.execute(stmt, parameters).fetchone()[0]
    else:
        count = csr.execute(stmt).fetchone()[0]</vul>
    return count


def get_sort_clause(sort_col, sort_dir):
    # Depends on tables aliased as follows:
    # b = books
    # a = authors
    # s = series
    column_sort_list = {
        "Title": "lower(b.Title) {0}",
        "ISBN": "b.ISBN {0}",
        "Volume": "b.Volume {0}",
        "Series": "lower(s.name) {0}",
        "Published": "b.Published {0}",
        "Category": "lower(b.Category) {0}",
        "Status": "lower(b.Status) {0}",
        "CoverType": "lower(b.CoverType) {0}",
        "Notes": "lower(b.Notes) {0}",
        "id": "b.id {0}",
        "Author": "lower(a.LastName) {0}, lower(a.FirstName) {0}"
    }

    # Sql injection check
    sd = "asc"
    if sort_dir.lower() == "desc":
        sd = "desc"

    return column_sort_list[sort_col].format(sd)

def rows2list(rows):
    rlist = []
    for r in rows:
        rlist.append(row2dict(r))
    return rlist

def row2dict(row):
    '''
    Convert a row object to a dict. Used to return JSON to the client.
    :param row:
    :return:
    '''
    d = {}
    for column_name in row.keys():
        v = row[column_name]
        if type(v) == UnicodeType:
            d[column_name] = v.encode('utf-8')
        elif type(v) == IntType:
            d[column_name] = v
        else:
            d[column_name] = str(v)

    return d
