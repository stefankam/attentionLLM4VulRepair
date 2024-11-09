from os.path import join
from datetime import datetime
<vul/>from flask import request, redirect, url_for, render_template, flash, g, current_app</vul>
from werkzeug.utils import secure_filename
from flask_security import login_required, current_user

from benwaonline.database import db
from benwaonline.models import Post, Tag, Comment, Preview, Image
from benwaonline.gallery import gallery
from benwaonline.gallery.forms import CommentForm, PostForm

@gallery.before_request
def before_request():
    g.user = current_user

@gallery.route('/gallery/')
@gallery.route('/gallery/<string:tags>/')
def show_posts(tags='all'):
    if tags == 'all':
        posts = Post.query.all()
    else:
        split = tags.split(' ')
        posts = []
        for s in split:
            results = Post.query.filter(Post.tags.any(name=s))
            posts.extend(results)

    tags = Tag.query.all()

    return render_template('gallery.html', posts=posts, tags=tags)

<vul/>@gallery.route('/gallery/benwa/')
def show_post_redirect():
    return redirect(url_for('gallery.show_posts'))</vul>

@gallery.route('/gallery/benwa/<int:post_id>')
def show_post(post_id):
    post = Post.query.paginate(post_id, 1, False)
    # Look at docs for get_or_404 or w.e
    if post.items:
        return render_template('show.html', post=post, form=CommentForm())

    flash('That Benwa doesn\'t exist yet')
    return redirect(url_for('gallery.show_posts'))

# Will need to add Role/Permissions to this later
@gallery.route('/gallery/benwa/add', methods=['GET', 'POST'])
@login_required
def add_post():
    form = PostForm()
    if form.validate_on_submit():
        f = form.image.data
        fname = secure_filename(f.filename)
        f.save(join(
            <vul/>current_app.static_folder, current_app.config['STATIC_BENWA_DIR'], fname</vul>
        ))
        fpath = '/'.join(['thumbs', fname])
        created = datetime.utcnow()
        preview = Preview(filepath=fpath, created=created)
        db.session.add(preview)

        fpath = '/'.join(['imgs', fname])
        image = Image(filepath=fpath, created=created, preview=preview)
        db.session.add(image)

        # 'benwa' is the forever the first tag in the database
        tags = [Tag.query.get(1)]
        added_tags = [get_or_create_tag(db.session, tag)[0] for tag in form.tags.data if tag]
        tags.extend(added_tags)

        post = Post(title=fname, created=datetime.utcnow(), image=image, tags=tags)
        db.session.add(post)

        current_user.posts.append(post)
        db.session.commit()

        return redirect(url_for('gallery.show_post', post_id=post.id))

    flash('There was an issue with adding the benwa')
    return render_template('image_upload.html', form=form)

def get_or_create_tag(session, tagname):
    instance = Tag.query.filter_by(name=tagname).first()
    if instance:
        return instance, False
    else:
        created = datetime.utcnow()
        instance = Tag(name=tagname, created=created)
        session.add(instance)

    return instance, True

@gallery.route('/gallery/benwa/<int:post_id>/comment/add', methods=['POST'])
@login_required
def add_comment(post_id):
    form = CommentForm()
    if form.validate_on_submit():
        post = Post.query.get(post_id)
        comment = Comment(content=form.content.data,\
                created=datetime.utcnow(), user=current_user, post=post)
        db.session.add(comment)
        db.session.commit()

    return redirect(url_for('gallery.show_post', post_id=post_id))

@gallery.route('/gallery/benwa/<int:post_id>/comment/delete/<int:comment_id>', methods=['GET',  'POST'])
@login_required
# @roles_accepted('admin', 'member')
def delete_comment(post_id, comment_id):
    comment = Comment.query.get_or_404(comment_id)

    if current_user.has_role('admin') or comment.owner(current_user):
        db.session.delete(comment)
        db.session.commit()
    else:
        flash('you can\'t delete this comment')

    return redirect(url_for('gallery.show_post', post_id=post_id))
