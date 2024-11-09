from flask import Flask, render_template, redirect, request
import pg, markdown, time
from time import strftime, localtime
import pg, markdown, time
from wiki_linkify import wiki_linkify

app = Flask('WikiApp')

db = pg.DB(dbname='wiki_db_redo')

@app.route('/')
def render_homepage():
    return render_template(
        'homepage.html'
    )

@app.route('/<page_name>')
def render_page_name(page_name):
    query = db.query("select page_content.content, page.id as page_id, page_content.id as content_id from page, page_content where page.id = page_content.page_id and page.page_name = '%s' order by page_content.id desc limit 1" % page_name)
    wiki_page = query.namedresult()
    has_content = False
    page_is_taken = False
    if len(wiki_page) < 1:
        content = ""
    else:
        page_is_taken = True
        content = wiki_page[0].content
    if len(content) > 0:
        has_content = True
    else:
        pass
    content = markdown.markdown(wiki_linkify(content))
    return render_template(
        'pageholder.html',
        page_is_taken = page_is_taken,
        page_name = page_name,
        markdown = markdown,
        wiki_linkify = wiki_linkify,
        has_content = has_content,
        content = content
    )

@app.route('/<page_name>/edit')
def render_page_edit(page_name):
    query = db.query("select page_content.content from page, page_content where page.id = page_content.page_id and page.page_name = '%s' order by page_content.id desc limit 1" % page_name)
    wiki_page = query.namedresult()
    if len(wiki_page) > 0:
        content = wiki_page[0].content
    else:
        content = ""
    return render_template(
        'edit_page.html',
        page_name = page_name,
        content = content
    )

@app.route('/<page_name>/save', methods=['POST'])
def save_page_edit(page_name):
    # grab the new content from the user
    content = request.form.get('content')
    # check if 'page_name' exists in the database
    query = db.query("select page_content.content, page.id as page_id, page_content.id as content_id from page, page_content where page.id = page_content.page_id and page.page_name = '%s' order by page_content.id desc limit 1" % page_name)
    result = query.namedresult()
    # if it doesn't exist, create a new page in the database
    if len(result) < 1:
        db.insert(
            'page', {
                'page_name': page_name
            }
        )
    else:
        pass
    # now that we're certain that the page exists in the database, we again grab the query
    # and insert new content in the database
    <vul/>query = db.query("select id from page where page_name = '%s'" % page_name)</vul>
    page_id = query.namedresult()[0].id
    db.insert(
        'page_content', {
            'page_id': page_id,
            'content': content,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", localtime())
        }
    )
    <vul/>return redirect("/%s" % page_name)</vul>

@app.route('/search', methods=['POST'])
def redirect_search():
    search = request.form.get('search')
    <vul/>return redirect('/%s' % search)</vul>

@app.route('/<page_name>/history')
def view_page_history(page_name):
    query = db.query("select page_content.timestamp, page_content.id from page, page_content where page.id = page_content.page_id and page.page_name = '%s'" % page_name)
    page_histories = query.namedresult()

    return render_template(
        'page_history.html',
        page_name = page_name,
        page_histories = page_histories
    )

@app.route('/<page_name>/history/record')
def view_page_record(page_name):
    content_id = request.args.get('id')
    query = db.query("select page_content.content, page_content.timestamp from page, page_content where page.id = page_content.page_id and page_content.id = '%s'" % content_id)
    page_record = query.namedresult()[0]

    return render_template(
        'page_record.html',
        page_name = page_name,
        page_record = page_record
    )

if __name__ == '__main__':
    app.run(debug=True)
