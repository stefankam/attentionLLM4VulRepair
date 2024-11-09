import functools

from flask import(
    Blueprint, flash, redirect, render_template, request, session, url_for, g
)
from werkzeug.security import check_password_hash, generate_password_hash

from DefinitelyNotTwitter.database import get_db
from . import user
from . import database as db
from . import user as user
from DefinitelyNotTwitter.user import get_user
from DefinitelyNotTwitter.auth import login_required

bp = Blueprint('admin', __name__, url_prefix='/admin')

def admin_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            return redirect(url_for('auth.login'))
        elif g.user['admin'] != 1:
            return redirect(url_for('blog.feedpage', page=0))

        return view(**kwargs)
    return wrapped_view

@bp.route('/user_view')
@bp.route('/user_view/<sort>')
@admin_required
def user_view(sort='id.asc'):
    db = get_db()
    sortBy = sort.split('.')[0]
    sortOrder = sort.split('.')[1]


    query = 'SELECT * FROM user AS u LEFT OUTER JOIN (SELECT uid, count(uid) AS follower FROM follows GROUP BY uid) AS f ON u.id = f.uid ORDER BY ? {}'.format(sortOrder)
    users = db.execute(
        <vul/>query, (sortBy,)</vul>
    ).fetchall()
    return render_template('admin/userview.html', users = users, sort='{}.{}'.format(sortBy, sortOrder))

@bp.route('/')
@bp.route('/<int:page>')
@admin_required
def admin_panel(page=0):

    db = get_db()

    postcount = g.postcount

    pagecount = int(postcount / 5 + 1)

    posts = db.execute(
        'SELECT * FROM post JOIN user WHERE post.uid = user.id AND post.reviewed = 1 ORDER BY created DESC LIMIT 5 OFFSET ?', (str(page*5),)
    ).fetchall()

    return render_template('admin/panel.html', posts = posts, pagecount=pagecount, page=page)

@bp.route('/edituser/<int:id>', methods= ('GET', 'POST'))
@admin_required
def edit_user(id):
    user = get_user(id)

    if request.method == 'POST':
        username = request.form['username']
        desc = request.form['desc']
        role = request.form['role']
        adminPwd = request.form['adminPwd']
        db = get_db()
        error = None
        file = None
        imgAdded = False

        # check if the post request has the file part
        if 'file' in request.files:
              f = request.files['file']
              filename = secure_filename(f.filename)
              filetype = filename.rsplit('.', 1)[1].lower()
              f.save(os.path.join(current_app.config['UPLOAD_FOLDER'], str(g.user["id"])+"."+filetype))
              imgAdded = True

        if not check_password_hash(g.user['password'], adminPwd):
            error = 'Incorrect admin password. Correct password required to edit user.'

        if error is None:
            if username is not "":
                db.execute(
                    'UPDATE user SET name = ? WHERE id = ?', (username, id,)
                )
            if desc is not "":
                db.execute(
                    'UPDATE user SET descrip = ? WHERE id = ?', (desc, id,)
                )
            if imgAdded:
                db.execute(
                    'UPDATE user SET avatar = 1 WHERE id = ?', (id,)
                )
            if role == 'restricted':
                db.execute(
                    'UPDATE user SET restricted = 1 WHERE id = ?', (id,)
                )
            if role == 'admin':
                db.execute(
                    'UPDATE user SET admin = 1 WHERE id = ?', (id,)
                )
            db.commit()
            return redirect(url_for('user.show_profile', id = user['id']))

        flash(error)

    return render_template('admin/edituser.html', user = user)

@bp.route('/restrict/<int:id>')
@admin_required
def restrict(id):

    db = get_db()
    user = get_user(id)
    error = None

    if user['restricted'] == 1:
        error = "User already restricted."
    elif user['admin'] == 1:
        error = "Cannot restrict admins."

    if error is None:
        db.execute(
            'UPDATE user SET restricted = 1 WHERE id = ?', (id,)
        )
        db.commit()
        return redirect(url_for('admin.user_view'))

    flash(error)
    return redirect(url_for('admin.user_view'))


@bp.route('/unrestrict/<int:id>')
@admin_required
def unrestrict(id):

    db = get_db()
    user = get_user(id)
    error = None

    if user['restricted'] != 1:
        error = "User already unrestricted."

    if error is None:
        db.execute(
            'UPDATE user SET restricted = 0 WHERE id = ?', (id,)
        )
        db.commit()
        return redirect(url_for('admin.user_view'))

    flash(error)
    return redirect(url_for('admin.user_view'))

@bp.route('/delete/<int:id>')
@admin_required
def delete(id):

    db = get_db()

    db.execute(
        'DELETE FROM user WHERE id = ?', (id,)
    )
    db.commit()

    message = "Deleted user!"
    flash(message)
    return redirect(url_for('admin.user_view'))

@bp.route('/promote/<int:id>')
@admin_required
def promote(id):

    db=get_db()
    user = get_user(id)
    error = None

    if user['restricted'] == 1:
        error = 'Cannot promote restricted user.'
    elif user['admin'] == 1:
        error = 'User is already an admin.'

    if error is None:
        db.execute(
            'UPDATE user SET admin = 1 WHERE id = ?', (id,)
        )
        db.commit()
        return redirect(url_for('admin.user_view'))

    flash(error)
    return redirect(url_for('admin.user_view'))

@bp.route('/strip/<int:id>')
@admin_required
def strip(id):

    db=get_db()
    user = get_user(id)
    error = None

    if user['admin'] != 1:
        error = 'User has no admin rights.'

    if error is None:
        db.execute(
            'UPDATE user SET admin = 0 WHERE id = ?', (id,)
        )
        db.commit()
        return redirect(url_for('admin.user_view'))

    flash(error)
    return redirect(url_for('admin.user_view'))


@bp.route('post/release/<int:pid>')
@admin_required
def release_post(pid):
    db = get_db()

    db.execute(
        'UPDATE post SET reviewed = 0 WHERE pid = ?', (pid,)
    )
    db.commit()
    message = "Released post!"
    flash(message)
    return redirect(url_for('admin.admin_panel'))


@bp.route('post/delete/<int:pid>')
@admin_required
def delete_post(pid):

    db = get_db()

    db.execute(
        'DELETE FROM post WHERE pid = ?', (pid,)
    )
    db.commit()
    message="Deleted post!"
    flash(message)
    return redirect(url_for('admin.admin_panel'))


@bp.before_app_request
def load_posts_to_be_reviewed():
    if g.user:
        if g.user['admin'] == 1:
            posts = get_db().execute(
                'SELECT * FROM post WHERE reviewed = 1'
            ).fetchall()
            g.postcount = len(posts)
