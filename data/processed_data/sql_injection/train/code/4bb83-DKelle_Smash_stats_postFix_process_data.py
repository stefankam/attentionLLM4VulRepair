import logger
import datetime
import constants
import get_results
import time
import copy
import player_web
import bracket_utils
from get_ranks import get_ranks
from get_results import get_coalesced_tag, sanitize_tag
import re
from tweet import tweet

LOG = logger.logger(__name__)

class processData(object):
    def __init__(self, db):
        LOG.info('loading constants for process')
        self.db = db

    def process(self, bracket, scene, display_name, new_bracket=False):
        # There are some brackets that have been blacklisted. Skip these brackets
        if bracket in constants.BLACKLIST:
            LOG.info('Skipping bracket due to blacklisting: {}'.format(bracket))
            return

        # Before we do anything, check if this url has been analyzed already, and bomb out
        <fix/>sql = "SELECT * FROM analyzed WHERE base_url = '{bracket}';"
        args = {'bracket': bracket}
        result = <fix/>self.db.exec(sql, args)</fix></fix>
        if len(result) > 0:
            LOG.info('tried to analyze {}, but has already been done.'.format(bracket))
            return

        # Send this bracket to get_results
        # We know the bracket is valid if it is from smashgg
        if 'smash.gg' in bracket:
            <fix/>success = <fix/>get_results.process(bracket, scene, self.db, display_name, new_bracket)</fix></fix>
            if success:
                self.insert_placing_data(bracket, new_bracket)
            else:
                #TODO add this URL to a table called 'failed_smashgg_brackets' or something
                LOG.exc('Analyzing smashgg tournament {} was not successful'.format(bracket))

        else:
            html, status = bracket_utils.hit_url(bracket)
            if status == 200 and bracket_utils.is_valid(html):
                get_results.process(bracket, scene, self.db, display_name, new_bracket)

    def insert_placing_data(self, bracket, new_bracket):
        # Get the html from the 'standings' of this tournament
        tournament_placings = bracket_utils.get_tournament_placings(bracket)

        for player, placing in tournament_placings.items():
            player = sanitize_tag(player)

            # Coalesce tag
            player = get_coalesced_tag(player)
            <fix/>sql = "INSERT INTO placings (url, player, place) VALUES ('{url}', '{player}', '{place}')"
            args = {'url': bracket, 'player': player, 'place': placing}</fix>

            self.db.exec(sql, args)

            if 'christmasmike' == player and new_bracket:
                if placing < 10:
                    msg = "Congrats on making {} dude! You're the best.".format(placing)
                    tweet(msg)

        LOG.info("tournament placings for {} are {}".format(bracket, tournament_placings))

    def check_and_update_ranks(self, scene):
        # There are 2 cases here:
        #   1) Ranks have never been calculated for this scene before
        #       - This means we need to calculate what the ranks were every month of this scenes history
        #       - We should only do this if ranks don't already exist for this scene
        #   2) Ranks have been calculated for this scene before
        #       - We already have bulk ranks. We should check if it has been more than 1 month since we last
        #           calculated ranks. If so, calculate again with the brackets that have come out this month

        LOG.info('About to check if ranks need updating for {}'.format(scene))
        # First, do we have any ranks for this scene already?
        <fix/>sql = 'select count(*) from ranks where scene="{scene}";'
        args = {'scene': scene}
        res = self.db.exec(sql, args)</fix>
        count = res[0][0]

        n = 5 if (scene == 'pro' or scene == 'pro_wiiu') else constants.TOURNAMENTS_PER_RANK
        if count == 0:
            LOG.info('Detected that we need to bulk update ranks for {}'.format(scene))
            # Alright, we have nothing. Bulk update ranks
            first_month = bracket_utils.get_first_month(self.db, scene)
            last_month = bracket_utils.get_last_month(self.db, scene)
            
            # Iterate through all tournaments going month by month, and calculate ranks
            months = bracket_utils.iter_months(first_month, last_month, include_first=False, include_last=True)
            for month in months:
                urls, _ = bracket_utils.get_n_tournaments_before_date(self.db, scene, month, n)
                self.process_ranks(scene, urls, month)
        else:

            # Get the date of the last time we calculated ranks
            <fix/>sql = "select date from ranks where scene='{scene}' order by date desc limit 1;"
            args = {'scene': scene}
            res = self.db.exec(sql, args)</fix>
            last_rankings_date = res[0][0]

            # Check to see if it's been more than 1 month since we last calculated ranks
            more_than_one_month = bracket_utils.has_month_passed(last_rankings_date)
            if more_than_one_month:
                # Get only the last n tournaments, so it doesn't take too long to process
                today = datetime.datetime.today().strftime('%Y-%m-%d')
                msg = 'Detected that we need up update monthly ranks for {}, on {}'.format(scene, today)
                LOG.info(msg)

                # We should only ever calculate ranks on the 1st. If today is not the first, log error
                if not today.split('-')[-1] == '1':
                    LOG.exc('We are calculating ranks today, {}, but it isnt the first'.format(today))

                months = bracket_utils.iter_months(last_rankings_date, today, include_first=False, include_last=True)
                for month in months:
                    # Make sure that we actually have matches during this month
                    # Say we are trying to calculate ranks for 2018-05-01, the player would need to have matches during 2018-04-01, 2018-04-30
                    prev_date = bracket_utils.get_previous_month(month)
                    brackets_during_month = bracket_utils.get_tournaments_during_month(self.db, scene, prev_date)

                    if len(brackets_during_month) > 0:
                        tweet('Calculating {} ranks for {}'.format(month, scene))
                        urls, _ = bracket_utils.get_n_tournaments_before_date(self.db, scene, month, n)
                        self.process_ranks(scene, urls, month)

            else:
                LOG.info('It has not yet been 1 month since we calculated ranks for {}. Skipping'.format(scene))


    def process_ranks(self, scene, urls, recent_date):
        PLAYER1 = 0
        PLAYER2 = 1
        WINNER = 2
        DATE = 3
        SCENE = 4

        # make sure if we already have calculated ranks for these players at this time, we do not do it again
        <fix/>sql = "SELECT * FROM ranks WHERE scene = '{scene}' AND date='{date}';"
        args = {'scene': scene, 'date': recent_date}
        res = self.db.exec(sql, args)</fix>
        if len(res) > 0:
            LOG.info('We have already calculated ranks for {} on date {}. SKipping'.format(scene, recent_date))
            return

        matches = bracket_utils.get_matches_from_urls(self.db, urls)
        LOG.info('About to start processing ranks for scene {} on {}'.format(scene, recent_date))

        # Iterate through each match, and build up our dict
        win_loss_dict = {}
        for match in matches:
            p1 = match[PLAYER1]
            p2 = match[PLAYER2]
            winner = match[WINNER]
            date = match[DATE]

            #Add p1 to the dict
            if p1 not in win_loss_dict:
                win_loss_dict[p1] = {}

            if p2 not in win_loss_dict[p1]:
                win_loss_dict[p1][p2] = []

            # Add an entry to represent this match to p1
            win_loss_dict[p1][p2].append((date, winner == p1))

            # add p2 to the dict
            if p2 not in win_loss_dict:
                win_loss_dict[p2] = {}

            if p1 not in win_loss_dict[p2]:
                win_loss_dict[p2][p1] = []

            win_loss_dict[p2][p1].append((date, winner == p2))

        ranks = get_ranks(win_loss_dict)

        tag_rank_map = {}
        for i, x in enumerate(ranks):
            points, player = x
            rank = len(ranks) - i

            <fix/>sql = "INSERT INTO ranks (scene, player, rank, points, date) VALUES ('{scene}', '{player}', '{rank}', '{points}', '{recent_date}');"
            args = {'scene': scene, 'player': player, 'rank': rank, 'points': points, 'recent_date': recent_date}
            self.db.exec(sql, args)</fix>

            # Only count this player if this is the scene he/she belongs to
            <fix/>sql = "SELECT scene FROM players WHERE tag='{player}';"
            args = {'player': player}
            res = self.db.exec(sql, args)</fix>

            if len(res) == 0 or res[0][0] == scene:
                # Also create a list to update the player web
                map = {'rank':rank, 'total_ranked':len(ranks)}
                tag_rank_map[player] = map

        player_web.update_ranks(tag_rank_map)
