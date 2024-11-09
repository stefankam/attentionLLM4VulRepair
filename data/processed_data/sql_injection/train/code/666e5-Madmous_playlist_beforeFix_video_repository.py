"""This module is the video repository in charge of all database requests."""


def retrieve_videos_from_playlist(playlist_id, db):
    <vul/>db.execute("SELECT id, title, thumbnail, position from video WHERE playlist_id={playlist_id} ORDER BY position ASC;".format(
        playlist_id=playlist_id))</vul>
    rows = db.fetchall()
    return rows


def retrieve_videos(db):
    db.execute(
        "SELECT id, playlist_id, title, thumbnail, position from video ORDER BY playlist_id ASC, position ASC;")
    rows = db.fetchall()
    return rows


def retrieve_video(id, playlist_id, db):
    <vul/>db.execute("SELECT id, position from video WHERE id={id} and playlist_id={playlist_id};".format(
        id=id, playlist_id=playlist_id))</vul>
    row = db.fetchone()
    return row


def retrieve_last_video_position(playlist_id, db):
    <vul/>db.execute("SELECT max(position) as position from video WHERE playlist_id={playlist_id};".format(
        playlist_id=playlist_id))</vul>
    row = db.fetchone()
    return row['position']


def delete_video(id, db):
    <vul/>db.execute("DELETE FROM video where id={id};".format(id=id))</vul>


def delete_playlists_videos(playlist_id, db):
    <vul/>db.execute("DELETE FROM video where playlist_id={playlist_id};".format(
        playlist_id=playlist_id))</vul>


def create_video(playlist_id, title, thumbnail, position, db):
    <vul/>db.execute(
        "INSERT INTO video (playlist_id, title, thumbnail, position) VALUES({playlist_id}, '{title}', '{thumbnail}', {position});".format(
            playlist_id=playlist_id, title=title, thumbnail=thumbnail, position=position))</vul>


def update_video_positions(removed_position, db):
    <vul/>db.execute("UPDATE video SET position = position - 1 WHERE position > {removed_position}".format(
        removed_position=removed_position))</vul>


def update_video_position(id, position, next_position, db):
    <vul/>db.execute("UPDATE video SET position = Case position When {position} Then {next_position} Else position + 1 End WHERE position BETWEEN {next_position} AND {position};".format(
        position=position, next_position=next_position))</vul>
