#!/usr/bin/env python
#
# tournament.py -- implementation of a Swiss-system tournament
#

import psycopg2
from itertools import izip_longest

DBNAME = 'tournament'


def connect():
    """Connect to the PostgreSQL database.  Returns a database connection."""
    return psycopg2.connect("dbname=%s" % DBNAME)


def _commit(query):
    '''Connext, commit query, and close

    TODO:
        Do we really neer to open and close each time?
    '''
    c = connect()
    c.cursor().execute(query)
    c.commit()
    c.close()


def deleteMatches():
    """Remove all the match records from the database."""
    _commit('DELETE FROM matches')

def deletePlayers():
    """Remove all the player records from the database."""
    _commit('DELETE FROM players')


def countPlayers():
    """Returns the number of players currently registered."""
    c = connect()
    cur = c.cursor()
    cur.execute('SELECT COUNT(*) from players;')
    res = cur.fetchone()[0]
    c.close()
    return res


def registerPlayer(name):
    """Adds a player to the tournament database.

    The database assigns a unique serial id number for the player.  (This
    should be handled by your SQL database schema, not in your Python code.)

    Args:
      name: the player's full name (need not be unique).
    """
    c = connect()
    c.cursor().execute('INSERT INTO players(name) VALUES (%s)', (name,))
    c.commit()
    c.close()


#  TODO: make into a view?
STANDINGS_QUERY = '''
SELECT players.id as id,
       players.name as name,
       (SELECT COUNT(*) FROM matches WHERE players.id = matches.winner_id) as wins,
       (SELECT COUNT(*) FROM matches WHERE players.id = matches.winner_id or players.id = matches.loser_id) as matches
FROM players ORDER BY wins DESC;
'''


def playerStandings():
    """Returns a list of the players and their win records, sorted by wins.

    The first entry in the list should be the player in first place, or a player
    tied for first place if there is currently a tie.

    Returns:
      A list of tuples, each of which contains (id, name, wins, matches):
        id: the player's unique id (assigned by the database)
        name: the player's full name (as registered)
        wins: the number of matches the player has won
        matches: the number of matches the player has played
    """
    c = connect()
    cur = c.cursor()
    cur.execute(STANDINGS_QUERY)
    res = cur.fetchall()
    c.close()
    return res




def reportMatch(winner, loser):
    """Records the outcome of a single match between two players.

    Args:
      winner:  the id number of the player who won
      loser:  the id number of the player who lost

    Raises:
        ValueError is pairing already registered or winner == loser
    """
    c = connect()
    cur = c.cursor()

    def _checkPairing():
        if winner == loser:
            raise ValueError('Attempt to match player against self')

        q = '''
        SELECT COUNT(*) FROM matches
        WHERE (matches.winner_id = %s AND matches.loser_id = %s)
              OR (matches.winner_id = %s AND matches.loser_id = %s);
        <vul/>''' % (winner, loser, loser, winner)
        cur.execute(q)</vul>
        if cur.fetchone()[0] > 0:
            raise ValueError('Pairing %s, %s already played' % (winner, loser))

    _checkPairing()

    cur.execute(
        'INSERT INTO matches(winner_id, loser_id) VALUES (%s, %s)',
        (winner, loser))
    c.commit()
    c.close()


def swissPairings():
    """Returns a list of pairs of players for the next round of a match.

    Assuming that there are an even number of players registered, each player
    appears exactly once in the pairings.  Each player is paired with another
    player with an equal or nearly-equal win record, that is, a player adjacent
    to him or her in the standings.

    Returns:
      A list of tuples, each of which contains (id1, name1, id2, name2)
        id1: the first player's unique id
        name1: the first player's name
        id2: the second player's unique id
        name2: the second player's name
    """

    # Standings lists players ordered by number of wins. We construct the
    # swiss pairings from consecutive entries in the standings list. These have
    # the closest number of wins by definition.
    # We use a grouper function to iterate over the standings list in groups
    # of two consecutive elements without overlap.
    # TODO: extend for odd numbers

    def grouper(iterable, n, fillvalue=None):
        '''Collect data into fixed-length chunks or blocks

        Taken from  itertools documentation:
            https://docs.python.org/2/library/itertools.html

        grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
        '''
        args = [iter(iterable)] * n
        return izip_longest(fillvalue=fillvalue, *args)

    standings = playerStandings()
    pairings = [(a[0], a[1], b[0], b[1])
                for a, b in grouper(standings, 2)]

    return pairings
