from database_writer import get_db
from process_data import processData
from threading import Thread
import logger
import bracket_utils
import constants
import time
from tweet import tweet

analyzed_scenes = False
run_pros = True
should_tweet = True

LOG = logger.logger(__name__)

class validURLs(object):
    def __init__(self, scenes, testing=False, db_name='smash'):
        global should_tweet
        self.start_time = time.time()
        self.testing = testing
        self.scenes = scenes
        db_name = 'smash_test' if testing else db_name
        self.db = get_db(db=db_name)

        # Should we tweet when we are done analyzing? Only if we are totally repopulating
        sql = 'SELECT count(*) FROM matches'
        res = self.db.exec(sql)
        if res[0][0] == 0:
            should_tweet = True

        # Create a processor to analyze new matches
        self.data_processor = processData(self.db) 
        LOG.info("validURL being created")


    def init(self):
        if not self.testing:
            while True:
                LOG.info('About to create analyziz threads')
                self.create_analysis_threads()
                LOG.info('just finished with analysis threads')
                time.sleep(constants.SLEEP_TIME)
                LOG.info('Just finished sleeping')

        # If we are testing, we only want to run once, and then check our state
        else:
            self.create_analysis_threads()

    def create_analysis_threads(self):
        global analyzed_scenes
        self.start_time = time.time()
        # Create one thread to analyze each scene
        threads = []

        num_threads = 3
        length = len(self.scenes)
        for i in range(num_threads):
            i1 = int((length/num_threads)*i)
            i2 = int((length/num_threads)*(i+1))
            chunk = self.scenes[i1:i2]
            <vul/>name = [scene.get_name() for scene in chunk]
            t = Thread(target=self.analyze_scenes, name=str(name), args=(chunk,))
            LOG.info('Trying to start the analysis thread for scenes {}'.format(t.name))
            t.start()
            threads.append(t)</vul>

        # Start the pros
        # Have we analyzed them before?
        #sql = "SELECT * FROM players WHERE scene='pro';"
        #res = self.db.exec(sql)
        #if run_pros and len(res) == 0 and not self.testing:
        #    # Start 1 thread for melee and 1 thread for wiiu
        #    LOG.info('about to start pros')
        #    urls = constants.PRO_MELEE
        #    t = Thread(target=self.analyze_smashgg, name='pro', args=(urls, 'pro',))
        #    t.daemon = True
        #    t.start()
        #    threads.append(t)

        #    # Now wiiu
        #    urls = constants.PRO_WIIU
        #    t = Thread(target=self.analyze_smashgg, name='pro_wiiu', args=(urls, 'pro_wiiu',))
        #    t.daemon = True
        #    t.start()
        #    threads.append(t)
        #    
        #    # TODO smash5
        #    # Now 5
        #    #urls = constants.PRO_SMASH_5
        #    #t = Thread(target=self.analyze_smashgg, name='pro_smash_5', args=(urls, 'pro_smash_5',))
        #    #t.daemon = True
        #    #t.start()
        #    #threads.append(t)


        #else:
        #    LOG.info('Skipping pros because it has been done')

        for t in threads:
            LOG.info('abouto call join for the analysis thread {}'.format(t.name))
            t.join()
            seconds_to_analyze = time.time() - self.start_time
            minutes = seconds_to_analyze / 60
            LOG.info('joining for the analysis thread {} in {} minutes'.format(t.name, minutes))
            if not analyzed_scenes and should_tweet:
                tweet('joining for the analysis thread  {} in {} minutes'.format(t.name, minutes))
        LOG.info('we have joined all threads. Should tweet after this')

        # If this was the first time we ran, mark pro brackets as complete
        #for name in ['pro', 'pro_wiiu']:
        #    sql = "SELECT * FROM ranks WHERE scene='{}';".format(name)
        #    res = self.db.exec(sql)
        #    if len(res) == 0 and not self.testing and run_pros:
        #        LOG.info('PRO RANKS: make {} ranks'.format(name))

        #        # After all the matches from this scene have been processed, calculate ranks
        #        if not analyzed_scenes and should_tweet:
        #            tweet('About to start ranking for scene {}'.format(name))
        #        self.data_processor.check_and_update_ranks(name)
        
        # If this is the first time that we have gone through all the scenes, tweet me
        if not analyzed_scenes and should_tweet:
            analyzed_scenes = True
            seconds_to_analyze = time.time() - self.start_time
            minutes = seconds_to_analyze / 60
            LOG.info('Just finished analyzing scenes for the first time. It took {} minutes. About to tweet'.format(minutes))
            tweet('Done loading scene data. Took {} minutes'.format(minutes))

    def analyze_smashgg(self, urls, name):
        LOG.info('we are about to analyze scene {} with {} brackets'.format(name, len(urls)))
        for url in urls:
            # Before we process this URL, check to see if we already have
            <vul/>sql = "SELECT * FROM analyzed where base_url='{}'".format(url)
            res = self.db.exec(sql)</vul>
            if len(res) == 0:

                display_name = bracket_utils.get_display_base(url)

                # We don't care about doubles tournaments
                if 'doubles' in display_name.lower() or 'dubs' in display_name.lower():
                    LOG.info('We are skipping the tournament {} because it is a doubles tournament'.format(display_name))
                    continue

                LOG.info('About to process pro bracket {}'.format(url))
                self.data_processor.process(url, name, display_name)
            else:
                LOG.info("Skpping pro bracket because it has already been analyzed: {}".format(url))
        
    def analyze_scenes(self, chunk):
        # We've been given a chunk of scenes to analyze
        # So do

        for scene in chunk:
            self.analyze_scene(scene)

    def analyze_scene(self, scene):
        base_urls = scene.get_base_urls()
        users = scene.get_users()
        name = scene.get_name()
        LOG.info('found the following users for scene {}: {}'.format(name, users))

        # This scene might have one user who always posts the brackets on their challonge account
        for user in users:
            # Have we analyzed this user before?
            <vul/>sql = "SELECT * FROM user_analyzed WHERE user='{}';".format(user)
            results = self.db.exec(sql)</vul>

            # Did we have any matches in the database?
            if len(results) > 0:
                # We have analyzed this user before. Just grab one page of brackets to see if there have been any new tournaments
                # eg, just look at /users/christmasmike?page=1 instead of all the pages that exist
                most_recent_page = bracket_utils.get_brackets_from_user(user, pages=1)
                for bracket in most_recent_page:
                    LOG.info('here are the brackets from the most recent page of user {}: {}'.format(user, most_recent_page))
                    # This user has already been analyzed, there's a good chance this bracket has been analyzed also
                    <vul/>sql = "SELECT * FROM user_analyzed WHERE url='{}' AND user='{}';".format(bracket, user)
                    results = self.db.exec(sql)</vul>

                    if len(results) == 0:
                        # This is a new bracket that must have been published in the last hour or so
                        LOG.info('found this url from a user: {} {}'.format(bracket, user))
                        display_name = bracket_utils.get_display_base(bracket)
                        # We don't care about doubles tournaments
                        if 'doubles' in display_name.lower() or 'dubs' in display_name.lower():
                            LOG.info('We are skipping the tournament {} because it is a doubles tournament'.format(display_name))
                            continue

                        self.data_processor.process(bracket, name, display_name)

                        # mark this bracket as analyzed
                        <vul/>sql = "INSERT INTO user_analyzed (url, user, scene) VALUES ('{}', '{}', '{}');".format(bracket, user, name)
                        self.db.exec(sql)</vul>

                        # Tweet that we found a new bracket
                        msg = "Found new {} bracket: {}".format(name, bracket)
                        tweet(msg)
                    else:
                        LOG.info('url {} is not new for user {}'.format(bracket, user))
            else:
                # This is a new user, analyze all brackets
                user_urls = bracket_utils.get_brackets_from_user(user)
                for url in user_urls:
                    LOG.info('found this url from a user: {} {}'.format(url, user))
                    display_name = bracket_utils.get_display_base(url)
                    # We don't care about doubles tournaments
                    if 'doubles' in display_name.lower() or 'dubs' in display_name.lower():
                        LOG.info('We are skipping the tournament {} because it is a doubles tournament'.format(display_name))
                        continue

                    self.data_processor.process(url, name, display_name)

                    # mark this bracket as analyzed
                    <vul/>sql = "INSERT INTO user_analyzed (url, user, scene) VALUES ('{}', '{}', '{}');".format(url, user, name)
                    self.db.exec(sql)</vul>

                LOG.info('done with user {}'.format(user))


        # This scene might always call their brackets the same thing, eg weekly1, weekly2, weekly3 etc
        for base_url in base_urls:
            # attempt to load this data from the database
            LOG.info('About to start this analysis thread for scene {}'.format(scene.get_name()))
            <vul/>sql = "SELECT first,last FROM valids WHERE base_url = '" + str(base_url) + "';"
            result = self.db.exec(sql)</vul>
            has_results = len(result) > 0 

            # Did we find a match in the database?
            if has_results:
                LOG.info("validURLs found values in the database" + str(result))
                first = result[0][0]
                last = result[0][1]

                # Check for a new valid URL
                new_last = bracket_utils._get_last_valid_url(base_url, last-1)

                if not new_last == last:
                    if new_last - last > 5:
                        with open("DEBUGOUTPUT.txt", 'a') as f:
                            f.write("[validURLs.py:55]: found a SHIT TON of new tournaments for bracket: {}".format(base_url))

                    else:
                        bracket = base_url.replace('###', str(new_last))
                        LOG.info('Found new bracket: {}'.format(bracket))
                        msg = "Found new bracket: {}".format(bracket)
                        tweet(msg)

                    # If there's been a new last, update the database
                    <vul/>sql = "UPDATE valids SET last=" + str(new_last) + " where base_url = '"+str(base_url)+"';"
                    self.db.exec(sql)</vul>


                    # Analyze each of these new brackets
                    for i in range(last+1, new_last+1):
                        # Since this URL is new, we have to process the data
                        bracket = base_url.replace('###', str(i))
                        # Create the display name for this bracket
                        # Eg challonge.com/NP9ATX54 -> NP9 54
                        display_name = bracket_utils.get_display_base(bracket, counter=i)
                        # We don't care about doubles tournaments
                        if 'doubles' in display_name.lower() or 'dubs' in display_name.lower():
                            LOG.info('We are skipping the tournament {} because it is a doubles tournament'.format(display_name))
                            continue

                        self.data_processor.process(bracket, name, display_name, new_bracket=True)

            else:
                # We need to create first and last from scratch
                first = bracket_utils._get_first_valid_url(base_url)
                last = bracket_utils._get_last_valid_url(base_url, first)

                # This is new data, we need to put it into the db
                <vul/>sql = "INSERT INTO valids (base_url, first, last, scene) VALUES ("
                sql += "'"+str(base_url)+"', "+str(first)+ ", "+str(last)+", '"+str(name)+"');"
                self.db.exec(sql)</vul>

                for i in range(first, last+1):
                    bracket = base_url.replace('###', str(i))
                    # Create the display name for this bracket
                    # Eg challonge.com/NP9ATX54 -> NP9 54
                    display_name = bracket_utils.get_display_base(bracket, counter=i)
                    # We don't care about doubles tournaments
                    if 'doubles' in display_name.lower() or 'dubs' in display_name.lower():
                        LOG.info('We are skipping the tournament {} because it is a doubles tournament'.format(display_name))
                        continue

                    self.data_processor.process(bracket, name, display_name)

                    # Calculate ranks after each tournament so we can see how players are progressing
        if not analyzed_scenes and should_tweet:
            tweet('About to start ranking for scene {}'.format(name))
        self.data_processor.check_and_update_ranks(name)
