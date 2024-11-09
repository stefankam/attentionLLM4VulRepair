"""This module is the playlist repository in charge of all database requests."""


def retrieve_playlists(db):
    db.execute('SELECT id, name from playlist;')
    rows = db.fetchall()
    return rows


def retrieve_playlist_by_id(id, db):
    db.execute(
        <fix/>"SELECT id, name, video_position from playlist WHERE id=%s;", (id,))</fix>
    row = db.fetchone()
    return row


def delete_playlist(id, db):
    <fix/>db.execute("DELETE FROM playlist where id=%s;", (id,))</fix>


def update_playlist(id, name, db):
    <fix/>db.execute("UPDATE playlist SET name=%s WHERE id=%s;", (name, id,))</fix>


def update_playlist_video_position(id, position, db):
    <fix/>db.execute("UPDATE playlist SET video_position=%s WHERE id=%s;",
               (position, id))</fix>


def create_playlist(name, db):
    db.execute(
        <fix/>"INSERT INTO playlist (name, video_position) VALUES(%s, 0);", (name,</fix>))
