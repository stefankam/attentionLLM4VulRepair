from time import sleep
import operator
from bs4 import BeautifulSoup
from requests import get
import constants
import logger
import re
import os
import pickle
import pysmash
from get_results import get_coalesced_tag
from urllib.parse import urlparse
import datetime

DEFAULT_BASE_URLS = ['https://challonge.com/NP9ATX###', 'http://challonge.com/heatwave###', 'https://austinsmash4.challonge.com/atx###',\
        'http://challonge.com/RAA_###']

debug = False
LOG = logger.logger(__name__)

def _get_first_valid_url(base_url):

    #Start from 1, and increment the number at the end or URL until we find a valid URL
    valid = False
    index = 1
    <fix/>while(index < 100):</fix>
        url = base_url.replace('###', str(index))
        data, status = hit_url(url)

        if status < 300 and is_valid(data, url=base_url):
            if debug: print('url ' + url + ' is valid')
            return index
            valid = True
        else:
            if debug: print('url ' + url + ' is not valid')
            index = index + 1

    <fix/>return -1</fix> 

def _get_last_valid_url(base_url, start=1):

    # If there was no start, there won't be an end
    if start == -1:
        return -1

    #We know that URL number 'start' is valid. What is the next invalid URL?
    invalid_count = 0
    end = start #Use this to keep track of the last valid URL

    #Sometimes a week is skipped -- Make sure we see 100 invalid URLs in a row before calling it quits
    while(invalid_count <= 30):
        #if base_url == "https://austinsmash4.challonge.com/atx145":
        #    print
        url = base_url.replace('###', str(start))
        print('about to check url {}'.format(url))
        if debug: print('start is ' + str(start))

        data, status = hit_url(url)

        if status < 300  and is_valid(data, url=base_url):
            if debug: print('url ' + str(url) + ' is valid')
            invalid_count = 0
            end = start
        else:
            invalid_count = invalid_count + 1

        start = start + 1
    return end

def get_valid_url_range(base_url):
    # Try to get this data form pickle
    start_end = load_pickle_data(base_url)
    if start_end:
        start, end = start_end

        # See if there have been new brackets since we pickled this data
        end = _get_last_valid_url(base_url, end)

    else:
        start = _get_first_valid_url(base_url)
        end = _get_last_valid_url(base_url, start)

    dump_pickle_data(base_url, (start,end))

    return start, end

def dump_pickle_data(base_fname, data):
    cwd = os.getcwd()

    # Go from https://ausin_melee_bracket -> austin_melee_bracket
    bracket_name = base_fname.replace('/', '_')
    fname = cwd+'/pickle/'+str(bracket_name)+'.p'

    with open(fname, "wb") as p:
        pickle.dump(data, p)

def load_pickle_data(base_fname):
    if debug: print('attempting to get pickle data for ', base_fname)
    # Attempt to get data from pickle
    cwd = os.getcwd()

    # Go from https://ausin_melee_bracket -> austin_melee_bracket
    bracket_name = base_fname.replace('/', '_')
    fname = cwd+'/pickle/'+str(bracket_name)+'.p'
    LOG.info('attempting to load pickle data for {}'.format(fname))

    try:
        with open(fname, 'rb') as p:
            data = pickle.load(p)
            return data

    except FileNotFoundError:
        LOG.info('could not load pickle data for {}'.format(fname))
        if debug: print('failed to get pickle data for ', base_fname)
        return None

def hit_url(url, load_from_cache=True):
    # Before we try to hit this URL, see if we have pickle data for it

    if load_from_cache:
        data =  load_pickle_data(url)
        if data:
            return data, 200

    #sleep, to make sure we don't go over our rate-limit
    sleep(.02)

    #Get the html page
    r = get(url)
    data = r.text

    if(is_valid(data, url=url) and load_from_cache):
        # Make sure we pickle this data, so we can get it next time
        dump_pickle_data(url, data)

    return data, r.status_code

def get_brackets_from_user(scene_url, total=None, pages=None):
    # Given the url for a given scene (https://austinsmash4.challonge.com)
    # Return all of the brackets hosted by said scene

    # 'total' is number of brackets to get. If None, get all. Usually either None or 1

    def get_bracket_urls_from_scene(scene_url, load_from_cache=True):
        # Given a specific page of a scene, parse out the urls for all brackets
        # eg inputhttps://austinsmash4.challonge.com?page=4
        # The above URL contains a list of brackets. Find those bracket URLs
        scene_brackets_html, status = hit_url(scene_url, load_from_cache=load_from_cache)
        scene_name = scene_url.split('https://')[-1].split('.')[0]
        soup = BeautifulSoup(scene_brackets_html, "html.parser")

        links = soup.find_all('a')
        bracket_links = []
        for link in links:
            if link.has_attr('href') and scene_name in link['href']:
                # Make sure this is a real bracket
                html = get_bracket(link['href'])
                if html and is_valid(html, url = link['href']):
                    bracket_links.append(link['href'])

                    # If we have more than 'total' links, we can return them now
                    if total and len(bracket_links) >= total:
                        return bracket_links
        return bracket_links

    # This scene may have multiple pages.
    # eg, https://austinsmash4.challonge.com?page=###
    # Find all the pages
    # Then find all the URLs for each page
    scene_url_with_pages = scene_url + '?page=###'
    start, end = get_valid_url_range(scene_url_with_pages)
    brackets = []
    for i in range(start, end+1):
        # It is possible that page 1 has changed since last time we checked. Don't load this page from cache
        cache = i > 1
        scene_url = scene_url_with_pages.replace('###', str(i))
        page_brackets = get_bracket_urls_from_scene(scene_url, cache)
        brackets.extend(page_brackets)

        # If we have more than 'total' links, we can return them now
        if total and len(brackets) >= total:
            return brackets

        # If we have already gotten urls from 'pages' pages, we can return now
        iterations = (start - i) + 1
        if pages and iterations >= pages:
            return brackets

    # Reverse this list so list[0] is the oldest bracket, and list[-1] is the newest bracket
    return brackets[::-1]

def is_valid(html, url=None):

    #Check to see if this tournament page exists
    errors= ['The page you\'re looking for isn\'t here', 'No tournaments found',\
            "Internal Server Error",
            "Not Implemented",
            "Bad Gateway",
            "Gateway Time-out",
            "Gateway Timeout",
            "Service Unavailable",
            "Gateway Timeout",
            "HTTP Version Not Supported",
            "Variant Also Negotiates",
            "Insufficient Storage",
            "Loop Detected",
            "Not Extended",
            "Network Authentication Required"]
    for error in errors:
        if error.lower() in str(html).lower():
            if debug:
                print('page invalid, found error string {}'.format(error))
            return False

    # If we are on a bracket, we need to make sure it is complete.
    # But, this might be a users page, eg. https://challonge.com/users/kuya_mark96
    # If that is the case, we shouldn't check for completeness
    if 'member since' in str(html).lower():
        return True

    # It may also be a page like this... http://smashco.challonge.com
    # Which is similar to a users page. Also don't check for completeness
    if 'organizations' in str(html).lower():
        return True

    # This might be a 'standings' page, like https://challongw.com/RAA_1/standings
    if url and 'standings' in url:
        return True

    return bracket_complete(html)


def bracket_complete(data):
    # Are there any matches that haven't been played yet?
    if "player1" not in data.lower() and "player2" not in data.lower():
        if debug:
            print('didnt find any players, must be invalid')
        return False
    if '"player1":null' in data.lower() or '"player2":null' in data.lower():
        if debug:
            print('found a null player, must be invalid')
        return False

    return True
    
def get_bracket(url):
    if debug:
        print('about to get bracket for url {}'.format(url))

    data, status = hit_url(url)

    # Create the Python Object from HTML
    soup = BeautifulSoup(data, "html.parser")

    # the bracket is inside a 'script' tag
    script = soup.find_all('script')
    bracket = None
    for s in script:
        if 'matches_by_round' in str(s):
            #We found the actual bracket. S contains all data about matches
            index = str(s).index('matches_by_round')
            s = str(s)[index:]
            bracket = (s)

    if debug: print('got bracket: \n', bracket)

    return bracket

def get_sanitized_bracket(url, symbol="{}"):
    bracket = get_bracket(url)
    sanitized = sanitize_bracket(bracket, symbol) if bracket else None
    return sanitized

def sanitize_bracket(bracket, symbol="{}"):
    #Which symbol should we be trying to match on? It will be either () or {}
    opn = symbol[0]
    close = symbol[-1]

    index = bracket.index(opn)

    #Cut off everything up until the first open bracket
    bracket = bracket[index:]

    #use a queue to cut off everything after the aligning close bracket
    count = 0
    for i, letter in enumerate(bracket):
        if letter == opn:
            count = count + 1
        if letter == close:
            count = count - 1

            #Also check to see if this is the final closing bracket
            if count == 0:
                index = i
                break

    bracket = bracket[:index+1]
    return bracket

def get_tournament_placings(bracket_url):
    # Map tags to their respective placings in this bracket
    placings_map = {}

    if 'challonge' in bracket_url:
        LOG.info('just entering "get tournament palcings')
        standings_html, status = hit_url(bracket_url+'/standings')
        soup = BeautifulSoup(standings_html, "html.parser")
        tds = soup.find_all('td')

        # Cycle thorugh these tds, and find the ones that represent different placings
        current_placing = 1
        for td in tds:
            if td.has_attr('class') and td['class'][0] == 'rank':
                current_placing = int(td.getText())
            span = td.find('span')
            # Player tags are kept in <span> elements
            if span:
                player = span.getText()

                # Coalesce tags
                player = get_coalesced_tag(player)
                placings_map[player.lower()] = current_placing
                LOG.info('just got placing {} for player {} in bracket {}'.format(current_placing, player, bracket_url))

    # This bracket is from smashgg
    else:
        smash = pysmash.SmashGG()
        url_parts = bracket_url.split('/')

        if 'tournament' in url_parts and 'events' in url_parts:
            t = url_parts[url_parts.index('tournament')+1]
            e = url_parts[url_parts.index('events')+1]
            players = smash.tournament_show_players(t, e)
            for player_dict in players:
                tag = player_dict['tag']
                # sanitize the tag
                tag = ''.join([i if ord(i) < 128 else ' ' for i in tag])
                place = player_dict['final_placement']
                placings_map[tag.lower()] = place

    return placings_map

<fix/>def player_in_urls(db, player, urls):
    """
    This method doesn't make use of the sql injection protection, because it would be more complicated to adapt.
    This is probably fine, because the URLs aren't entirely user input. They still must match the hardcoded base_urls
    provided by the smash api
    """</fix>

    sql = "SELECT * FROM matches WHERE (player1='{}' or player2='{}')".format(player, player, urls)
    if len(urls) > 0:
        sql = sql + " and (url='{}'".format(urls[0])
        for url in urls[1:]:
            sql = sql + " or url='{}'".format(url)
        sql = sql + ");"
    res = db.exec(sql)

    if len(res) > 0:
        return True
    LOG.info('player {} is not in {}'.format(player, urls))
    return False

def player_in_bracket(player, bracket=None):
    # Make sure to add quotations around the tag
    # this way, we ony match on actual tags, and not *tag*
    #player = '<title>'+player+'</title>'

    # This player may have multiple tags
    # Check if any of them are in the bracket
    tags = get_coalesce_tags(player)
    for tag in tags:
        if re.search(tag, bracket, re.IGNORECASE):
            return True
    return False

def get_coalesce_tags(player):
    for tags in constants.TAGS_TO_COALESCE:
        if player in tags:
            return tags
    # If this tag does not need to be coalesced, just return a list of this
    return [player]

def get_urls_with_players(players=["Christmas Mike", "christmasmike"], base_urls=DEFAULT_BASE_URLS):
    urls = []
    for base in base_urls:
        start, end = get_valid_url_range(base)
        for i in range(start, end+1):
            bracket_url = base.replace('###', str(i))
            bracket = get_sanitized_bracket(bracket_url)
            for player in players:
                if bracket and player_in_bracket(player, bracket=bracket):
                    urls.append(bracket_url)
                    break
    return urls

def get_list_of_scenes():
    <fix/>return constants.BRACKETS_TO_ANALYZE</fix>

def get_list_of_named_scenes():
    austin = constants.AUSTIN_URLS
    smashbrews = constants.SMASHBREWS_RULS
    colorado_singles = constants.COLORADO_SINGLES_URLS
    colorado_doubles = constants.COLORADO_DOUBLES_URLS
    sms = constants.SMS_URLS
    base_urls = [['sms', sms], ['smashbrews', smashbrews], ['austin', austin], ['colorado', colorado_singles], ['colorado_doubles', colorado_doubles]]
    return base_urls

def get_list_of_scene_names():
    # TODO This needs to change if you want to analyze another scene besides only ult
    #return ['Smash Ultimate']
    return ['sms', 'austin', 'smashbrews', 'colorado', 'colorado_doubles', 'pro', 'pro_wiiu', 'test1', 'test2']

def get_last_n_tournaments(db, n, scene):
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    return get_n_tournaments_before_date(db, scene, today, n)

def get_first_month(db, scene):
    <fix/>sql = "select date from matches where scene='{scene}' order by date limit 1;"
    args = {'scene': scene}
    res = db.exec(sql, args)</fix>
    date = res[0][0]
    return date

def get_next_month(date):
    y, m, d = date.split('-')
    m = '01' if m == '12' else str(int(m)+1).zfill(2)
    y = str(int(y)+1).zfill(2) if m == '01' else y
    date = '{}-{}-{}'.format(y, m, d)
    return date

def get_previous_month(date):
    y, m, d = date.split('-')
    m = '12' if m == '01' else str(int(m) - 1).zfill(2)
    y = str(int(y) - 1).zfill(2) if m == '12' else y
    date = '{}-{}-{}'.format(y, m, d)
    return date

def get_last_month(db, scene):
    <fix/>sql = "select date from matches where scene='{scene}' order by date desc limit 1;"
    args = {'scene': scene}
    res = db.exec(sql, args)</fix>
    date = res[0][0]

    # If it has been more than 1 month since this last tournament,
    # go ahead and round this date up by a 1 month
    # eg, if the last tournament was 2015-01-15 (a long time ago)
    # we can assume the scene won't have more tournaments
    # So just round to 2015-02-01
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    y, m, d = today.split('-')
    cy, cm, cd = date.split('-')
    if y > cy or m > cm:
        # Add 1 to the month before we return
        # eg 2018-03-01 -> 2018-04-01
        date = get_next_month(date)

    return date

def get_first_ranked_month(db, scene, player):
    <fix/>sql = "select date from ranks where scene='{scene}' and player='{player}' order by date limit 1;"
    args = {'scene': scene, 'player': player}
    res = db.exec(sql, args)</fix>
    date = res[0][0]
    return date

def get_last_ranked_month(db, scene, player):
    <fix/>sql = "select date from ranks where scene='{scene}' and player='{player}' order by date desc limit 1;"
    args = {'scene': scene, 'player': player}
    res = db.exec(sql, args)</fix>
    date = res[0][0]
    return date

def iter_months(first, last, include_first=True, include_last=False):
    # Both first and last are date strings in the format yyyy-mm-dd

    y, m, d = first.split('-')
    last_y, last_m, last_d = last.split('-')
    cur = '{}-{}'.format(y, m)
    last = '{}-{}'.format(last_y, last_m)

    # Calculate ranks on the first of every month between first and last
    months = []
    if include_first:
        months.append('{}-01'.format(cur))


    op = operator.ge if include_last else operator.gt
    while op(last, cur):
        m = str(int(m) + 1)

        if m == '13':
            m = '01'
            y = str(int(y) + 1)

        # Make sure to pad the month with 0s
        m = m.zfill(2)
        cur = '{}-{}'.format(y, m)
        months.append('{}-01'.format(cur))

    # We don't actually want to include this last month.
    # Eg. if the last tournament was played on 2018-02-04, we don't want to calculate the ranks
    # For Feb. until March starts

    return months[:len(months)-1]

def has_month_passed(date):
    y, m, d = date.split('-')
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    today_y, today_m, today_d = today.split('-')

    # Are these two in the same month?
    if m == today_m:
        return False

    # Otherwise, we know that 'date' is in the past, and 'today' is current.
    # We must be in a new month now. Always rank on the 1st
    if today_d == '01':
        return True

    return False

def get_monthly_ranks_for_scene(db, scene, tag):

    <fix/>sql = "SELECT date, rank FROM ranks WHERE scene='{scene}' AND player='{tag}'"
    args = {'scene': scene, 'tag': tag}
    res = db.exec(sql, args)</fix>

    res = [r for r in res if played_during_month(db, scene, tag, get_previous_month(r[0]))]

    # Build up a dict of {date: rank}
    ranks = {}
    for r in res:
        ranks[r[0]] = r[1]

    return ranks

def get_ranking_graph_data(db, tag):
    # First, we have to find out which scenes this player is ranked in
    <fix/><fix/>sql = "SELECT DISTINCT scene FROM ranks WHERE player='{tag}'"
    args = {'tag': tag}
    scenes = db.exec(sql, args)</fix></fix>
    scenes = [s[0] for s in scenes]

    # Get the first time we were ranked in each of these scenes
    first_months = [get_first_ranked_month(db, s, tag) for s in scenes]
    last_months = [get_last_ranked_month(db, s, tag) for s in scenes]

    first_month = min(first_months)
    last_month = max(last_months)

    # Get a list of each month that we want to know the ranks for
    iterated_months = iter_months(first_month, last_month, include_last=True)

    monthly_ranks_per_scene = {s:get_monthly_ranks_for_scene(db, s, tag) for s in scenes}

    ranks_per_scene = {s:[] for s in scenes}
    # Reformat this data to use with Zing
    for month in iterated_months:
        for s in scenes:
            scene_ranks = monthly_ranks_per_scene[s]
            if month in scene_ranks:
                ranks_per_scene[s].append([month, scene_ranks[month]])

    

    return ranks_per_scene, iterated_months

def get_bracket_placings_in_scene(db, scene, tag):
    sql = "select distinct matches.date, placings.place from placings join matches on \
            <fix/>matches.url=placings.url where scene='{scene}' and ((player1='{tag_one}' and placings.player=player1) or \
            (player2='{tag_two}' and placings.player=player2));"
    args = {'scene': scene, 'tag_one': tag, 'tag_two': tag}
    res = db.exec(sql, args)</fix>

    # Convert all placings to ints
    res = [[r[0], int(r[1])] for r in res]
    return res

def get_bracket_graph_data(db, tag):
    # First, we have to find out which scenes this player has brackets in
    sql = "SELECT DISTINCT scene FROM ranks WHERE player='{tag}'"
    args = {'tag': tag}
    scenes = db.exec(sql, args)
    scenes = [s[0] for s in scenes]

    bracket_placings_by_scene = {s: get_bracket_placings_in_scene(db, s, tag) for s in scenes}

    return bracket_placings_by_scene


def get_tournaments_during_month(db, scene, date):
    y, m, d = date.split('-')
    ym_date = '{}-{}'.format(y, m)
    <fix/>sql = "select url, date from matches where scene='{scene}' and date like '%{date}%' group by url, date order by date"
    args = {'scene': scene, 'date': ym_date}
    res = db.exec(sql, args)</fix>
    urls = [r[0] for r in res]
    return urls

def played_during_month(db, scene, tag, date):
    # First, which tournaments were hosted during this month?
    tournaments = get_tournaments_during_month(db, scene, date)

    <fix/>if player_in_urls(db, tag, urls=tournaments):</fix>
        return True

    return False

def get_n_tournaments_before_date(db, scene, date, limit):
    <fix/>sql = "select url, date from matches where scene='{scene}' and date<='{date}' group by url, date order by date desc limit {limit};"
    args = {'scene': scene, 'date': date, 'limit': limit}
    res = db.exec(sql, args)</fix>
    urls = [r[0] for r in res]
    return urls, date

def get_n_tournaments_after_date(db, scene, date, limit):
    <fix/>sql = "select url, date from matches where scene='{scene}' and date>='{date}' group by url, date order by date desc limit {limit};"
    args = {'scene': scene, 'date': date, 'limit': limit}
    res = db.exec(sql, args)</fix>
    urls = [r[0] for r in res]
    return urls, date

def get_date(url):
    url = url + "/log"
    bracket, status = hit_url(url)

    # TODO figure out what to do if this string is not in
    s2 = '2015-03-07'
    if 'created_at' not in bracket:
        return s2

    first_occurance = str(bracket).index('created_at')
    bracket = bracket[first_occurance:]

    #TODO if one day this code randomly stop working, it's probably this
    s = 'created_at":"'
    i = len(s)
    i2 = len(s2) + i
    date = bracket[i:i2]
    y = date.split('-')[0]
    m = date.split('-')[1]
    d = date.split('-')[2]

    return date

def get_matches_from_urls(db, urls):
    matches = set()
    for url in urls:
        <fix/>sql = "SELECT * FROM matches WHERE url='{url}';"
        args = {'url': url}
        res = set(db.exec(sql, args))</fix>
        matches |= set(res)

    return matches

def get_display_base(url, counter=None):
    # Try to get the title of this challonge page, maybe the creator gave it a good display name
    if 'challonge' in url:
        html, _ = hit_url(url)
        soup = BeautifulSoup(html, "html.parser")

        display_name = soup.find('div', {'id' :'title'})
        if display_name and hasattr(display_name, 'title'):
            title = display_name.text.rstrip().lstrip()
            name = re.sub("[^a-z A-Z 0-9 # / \ .]",'', title)
            return name
        else:
            LOG.info('url {} has no title'.format(url))

        # We couldn't find the title in a div. It may be in an h1
        display_name = soup.find('h1', {'class': 'title'})
        if display_name:
            name = display_name.find(text=True).lstrip().rstrip()
            LOG.info('just found new title for url: {} - {}'.format(url, name))

            return name

    # We couldn't find a title in the HTML. See if we have a hard-coded one
    d_map = constants.DISPLAY_MAP
    for k in d_map:
        if  k.lower() in url.lower():
            base = d_map[k]
            if counter:
                name = '{} {}'.format(base, counter)
                return name
            return base
    
    # If this is a pro bracket, just pull the name out of the URL
    if 'smash.gg' in url:
        parts = url.split('event')[0].split('/')[-2].split('-')
        display_list = [s.title() for s in parts]
        return ' '.join(display_list)

    # None of the above methods worked. Just call this by its URL
    return url

def get_smashgg_brackets(pages=None, all_brackets=True, singles=True, scene='pro'):
    results = 0
    per_page = 5
    page = 1 if pages == None else pages[0]
    brackets = {}
    smash = pysmash.SmashGG()

    def iterate():
        print('PAGE {}'.format(page))
        # melee
        #results_url = 'https://smash.gg/results?per_page=5&filter=%7B%22completed%22%3Atrue%2C%22videogameIds%22%3A%221%22%7D&page={}'.format(page)
        results_url = "https://smash.gg/tournaments?per_page=30&filter=%7B%22upcoming%22%3Afalse%2C%22videogameIds%22%3A4%2C%22past%22%3Atrue%7D&page={}".format(page)
        
        #wiiu
        #results_url = 'https://smash.gg/results?per_page=5&filter=%7B%22completed%22%3Atrue%2C%22videogameIds%22%3A3%7D&page={}'.format(page)

        #Get the html page
        r = get(results_url)
        data = r.text
        soup = BeautifulSoup(data, "html.parser")
        grep = 'singles' if singles else 'doubles'
        #print(data)

        links = soup.find_all('a')
        for link in links:
            try:
                if link.has_attr('href') and 'tournament' in link['href']:
                    url_parts = link['href'].split('/')

                    t = url_parts[url_parts.index('tournament')+1]
                    if t in brackets:
                        continue

                    events = smash.tournament_show_events(t)
                    def get_event(events, matches):
                        # Do we have a melee singles event?
                        for e in events['events']:
                            if all([match in e for match in matches]):
                                return e
                                
                        return None

                    if scene=='pro_wiiu':
                        e = get_event(events, ['wii', 'single'])
                        if e == None:
                            e = get_event(events, ['single'])
                        if e == None:
                            e = get_event(events, ['wii'])
                        if e == None:
                            e = get_event(events, ['smash-4'])
                        if e == None:
                            e = get_event(events, ['smash4'])
                        if e == None:
                            e = get_event(events, ['smash'])
                        if e == None:
                            continue

                    elif scene=='pro':
                        e = get_event(events, ['melee', 'single'])
                        if e == None:
                            e = get_event(events, ['single'])
                        if e == None:
                            e = get_event(events, ['gamecube'])
                        if e == None:
                            e = get_event(events, ['melee'])
                        if e == None:
                            e = get_event(events, ['smash'])
                        if e == None:
                            continue

                    url = 'https://smash.gg/tournament/{}/events/{}'.format(t, e)
                    brackets[t] = url
                    with open('threaded_smash_gg_brackets.txt', 'a') as f:
                        f.write('PAGE{}[[{}]]\n'.format(page, url))

                    
            except Exception as e:
                continue

    if pages:
        for page in pages:
            iterate()
    else:
        while results < 7730:
            iterate()
            results = results + per_page
            page = page + 1

    return brackets

def translate_url_to_api_name(url):
    # eg. https://austinsmash4.challonge.com/atx122 -> austinsmash4-atx122
    # eg. https://challonge.com/atx122 -> atx122
    parse = urlparse(url)
    net = parse.netloc
    # Do we have a username? eg. austinsmash4
    index = net.index('challonge')
    user = None
    if index > 0:
        user = net[:index-1]
    tourney_id = parse.path.replace('/', '')
    if user:
        return '{}-{}'.format(user, tourney_id)
    return tourney_id
