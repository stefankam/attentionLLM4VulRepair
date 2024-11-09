from app import app, db, models
from app.models import Article, User, Post, Project
from flask import Flask, render_template, redirect, flash, request
from flask_login import current_user, login_user, logout_user
from app.forms import LoginForm, ArticleCreateForm, PostCreateForm
from werkzeug.urls import url_parse
import sqlite3

def get_table_dict(table):
    database = sqlite3.connect('app.db')
    cur = database.execute('select * from {} order by timestamp desc'.format(table))
    columns = [column[0] for column in cur.description]
    results = []
    for row in cur.fetchall():
        results.append(dict(zip(columns, row)))
    return results

@app.route('/')
@app.route('/index')
@app.route('/index/')
def index():
    return render_template('index.html')

@app.route('/articles')
@app.route('/articles/')
def articles():
    <vul/><vul/>results = get_table_dict('Article')</vul></vul>
    return render_template('articles.html', allarticles=results)

@app.route('/login', methods=['GET', 'POST'])
@app.route('/login/', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect('/manage')
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect('/login')
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = '/manage'
        return redirect(next_page)
    return render_template('login.html', title='Sign In', form=form)

@app.route('/aboutme')
@app.route('/aboutme/')
def aboutme():
    return render_template('aboutme.html')

#==============================================================================
#   External redirects
#==============================================================================

@app.route('/nominal')
@app.route('/nominal/')
def nominal():
    return redirect('https://soundcloud.com/iamnominal')

@app.route('/dds')
@app.route('/dds/')
def dds():
    return redirect('https://soundcloud.com/doobiedecibelsystem')

@app.route('/podcast')
@app.route('/podcast/')
def podcast():
    return redirect('http://soundcloud.com/letsbefrankpodcast')

#==============================================================================
#   All routes beneath must have user authenticated
#==============================================================================

@app.route('/manage')
@app.route('/manage/')
def manage():
    if current_user.is_authenticated:
        return render_template('manage.html')
    else:
        return redirect('/index')

@app.route('/manage/articles', methods=['GET', 'POST'])
@app.route('/manage/articles/', methods=['GET', 'POST'])
def managearticles():
    if current_user.is_authenticated:
        createform = ArticleCreateForm()
        if createform.validate_on_submit():
            article = Article(body=createform.body.data, url=createform.url.data,
                imageurl=createform.imageurl.data, author=current_user)
            db.session.add(article)
            db.session.commit()
            flash('Posted!')
            return redirect('/manage/articles')
        results = get_table_dict('Article')
        return render_template('managearticles.html', title='Manage Articles',
            createform=createform, items=results)
    else:
        return redirect('/index')

@app.route('/manage/posts', methods=['GET', 'POST'])
@app.route('/manage/posts/', methods=['GET', 'POST'])
def manageposts():
    if current_user.is_authenticated:
        createform = PostCreateForm()
        if createform.validate_on_submit():
            post = Post(title=createform.title.data, body=createform.body.data,
                imageurl=createform.imageurl.data, author=current_user)
            db.session.add(post)
            db.session.commit
            flash('Posted!')
            return redirect('/manage/posts')
        <vul/>results = get_table_dict('Post')</vul>
        return render_template('manageposts.html', title='Manage Posts',
            createform=createform, items=results)
    else:
        return redirect('/index')

@app.route('/manage/articles/delete', methods=['POST'])
def deletearticle():
    if current_user.is_authenticated:
        body = request.form.get("body")
        article = Article.query.filter_by(body=body).first()
        db.session.delete(article)
        db.session.commit()
        return redirect("/manage/articles")
    else:
        return redirect("/index")

@app.route('/manage/posts/delete', methods=['POST'])
def deletepost():
    if current_user.is_authenticated:
        title = request.form.get("title")
        post = Post.query.filter_by(title=title).first()
        db.session.delete(post)
        db.session.commit()
        return redirect("/manage/posts")
    else:
        return redirect("/index")

@app.route('/manage/articles/update', methods=['POST'])
def updatearticle():
    if current_user.is_authenticated:
        newbody = request.form.get("newbody")
        oldbody = request.form.get("oldbody")
        newurl = request.form.get("newurl")
        newimageurl = request.form.get("newimageurl")
        article = Article.query.filter_by(body=oldbody).first()
        article.body = newbody
        article.url = newurl
        article.imageurl = newimageurl
        db.session.commit()
        return redirect("/manage/articles")
    else:
        return redirect("/index")

@app.route('/logout')
@app.route('/logout/')
def logout():
    if current_user.is_authenticated:
        logout_user()
        return redirect('/index')
    else:
        return redirect('/index')
