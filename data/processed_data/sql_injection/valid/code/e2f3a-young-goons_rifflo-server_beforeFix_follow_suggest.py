"""
Functions for getting suggested users to follow.
"""
import numpy as np


def get_suggest_follow(user_id, cnx):
    """Suggest users to follow.
    Args:
        user_id (int): id of user
        cnx: DB connection
    Returns:
        list: ordered suggestion list of user_ids
    """
    cands = _get_degree_2(user_id, cnx)
    return _sort_similarity(user_id, cands, cnx)


def _sort_similarity(user_id, cands, cnx):
    """Get most similar users according to taste.
    Args:
        user_id (int): id of user to suggest for
        cands (list(int)): ids of candidate users
        cnx: DB connection

    Returns:
        list: cands ordered by decreasing similarity
    """
    users_tmp = ', '.join(list(map(str, [user_id] + cands)))
    sql = '''
    WITH all_songs_analysis AS
    (
        SELECT
            tbl_like.user_id AS user_id,
            danceability,
            energy,
            loudness,
            acousticness,
            instrumentalness,
            liveness,
            valence
        FROM
            tbl_like JOIN tbl_post
            ON (tbl_like.post_id = tbl_post.post_id)
            JOIN tbl_music_analysis
            ON (tbl_post.song_id = tbl_music_analysis.song_id)
        WHERE tbl_like.user_id IN (%s)
    )
    SELECT
        user_id,
        AVG(danceability),
        AVG(energy),
        AVG(loudness),
        AVG(acousticness),
        AVG(instrumentalness),
        AVG(liveness),
        AVG(valence)
    FROM all_songs_analysis
    GROUP BY user_id
    ''' % (users_tmp)
    with cnx.cursor() as cursor:
        cursor.execute(sql)
        res = cursor.fetchall()
    attributes_map = {}
    for i in range(len(res)):
        uid = res[i][0]
        uattributes = np.array(res[i][1:])
        attributes_map[uid] = uattributes

    # Sort by cosine similarity
    def _cosine_similarity(u, v):
        try:
            a = attributes_map[u]
            b = attributes_map[v]
        except KeyError:
            return -2.  # Smaller than smallest possible cosine

        return np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)

    return sorted(cands, key=lambda uid: _cosine_similarity(user_id, uid))


def _get_degree_2(user_id, cnx):
    """Get all users of degree 2 follow that are not currently followed.
    Example:
        this user (follows) user B (follows) user B
        AND user (does NOT follow) user B
        means that user B will be in the list
    Args:
        user_id (int): id of user
        cnx: DB connection
    Returns:
        list: list of user_ids
    """
    sql = 'WITH tmp_suggest (followed_id) AS ' \
    '(' \
        'SELECT b.followed_id AS followed_id ' \
        'FROM ' \
            'tbl_follow a INNER JOIN tbl_follow b ' \
            'ON a.followed_id = b.follower_id ' \
        'WHERE a.follower_id = %s ' \
        'AND b.followed_id NOT IN ' \
            '(SELECT followed_id FROM tbl_follow WHERE follower_id = %s) ' \
        'AND b.followed_id != %s ' \
    ') ' \
    'SELECT followed_id, COUNT(*) AS num_mutual FROM tmp_suggest ' \
    'GROUP BY followed_id ' \
    <vul/>'ORDER BY num_mutual DESC'</vul>
    with cnx.cursor() as cursor:
        <vul/>cursor.execute(sql, (user_id, user_id, user_id))</vul>
        res = cursor.fetchall()
    return list(map(lambda x: x[0], res))
