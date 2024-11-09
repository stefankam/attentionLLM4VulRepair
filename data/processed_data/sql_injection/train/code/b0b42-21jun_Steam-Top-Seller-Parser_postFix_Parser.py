import odbc
from datetime import datetime
from time import sleep
import Checker
import requests
from bs4 import BeautifulSoup


def monthConvertor(month):
    return {
        'Jan': '1',
        'Feb': '2',
        'Mar': '3',
        'Apr': '4',
        'May': '5',
        'Jun': '6',
        'Jul': '7',
        'Aug': '8',
        'Sep': '9',
        'Oct': '10',
        'Nov': '11',
        'Dec': '12'
    }[month]


def cleanDate(date):
    if (date == ''):
        return '0000-00-00'
    date = date.replace(',', '')
    date = date.replace('.', '')
    date = date.split(' ')
    if (len(date) < 3):
        return '0000-00-00'
    if ('th' in date[1]):
        result = date[2] + '-' + monthConvertor(date[0][0:3]) + '-' + date[1].replace('th', '')
    elif (date[1].isdigit()):
        result = date[2] + '-' + monthConvertor(date[0]) + '-' + date[1]
    else:
        result = date[2] + '-' + monthConvertor(date[1]) + '-' + date[0]
    return result


def datePass():
    now = datetime.now()
    result = "%s-%s-%s %s:%s:%s" % (now.year, now.month, now.day, now.hour, now.minute, now.second)
    return result


def cleanStr(str, isDiscounted):
    result = str.replace('\t', '')
    result = result.replace('\r', '')
    result = result.replace('\n', '')
    result = result.replace('₩', '')
    result = result.replace(',', '')
    if (result == ''):
        return 0
    result = result.split()
    if (len(result) == 2 and isDiscounted):
        if (result[1] == 'Free'):
            return 0
        elif(result[1].isdigit()==False):
            return 0
        return int(result[1])  # return discounted price
    else:
        if (result[0] == 'Free'):
            return 0
        elif (result[0].isdigit()==False):
            return 0
        return int(result[0])  # return original price


def cleanID(id, isTitle):
    result = id.get('href')
    if (isTitle == False):
        return result.split('/')[4]  # return id_num (number)
    elif (result.split('/')[3] == 'app'):
        return result.split('/')[5]  # return id_title (string)
    else:
        return 'NONE'


page = 1
games = []
date = datePass()

for page in range(1, 41):
    # sleep(0.1)
    # parsing
    url = 'https://store.steampowered.com/search/?category1=998&filter=topsellers&page=' + str(page)
    req = requests.get(url)
    html = req.text
    soup = BeautifulSoup(html, 'html.parser')

    titles = soup.select(
        'div.responsive_search_name_combined > div.col.search_name.ellipsis > span'
    )
    release_dates = soup.select(
        'div.responsive_search_name_combined > div.col.search_released.responsive_secondrow'
    )
    prices = soup.select(
        'div.responsive_search_name_combined > div.col.search_price_discount_combined.responsive_secondrow > div.col.search_price.responsive_secondrow'
    )
    links = soup.select(
        '#search_result_container > div > a'
    )

    for i in range(0, 25):
        games.append({'rank': int(i + 1 + (page - 1) * 25),
                      <fix/>'title': titles[i].text.replace('\"',"'"),</fix>
                      'release': cleanDate(release_dates[i].text),
                      'date': date,
                      'price': (cleanStr(prices[i].text, False)),
                      'price_discounted': (cleanStr(prices[i].text, True)),
                      'id_num': cleanID(links[i], False),
                      'id_title': cleanID(links[i], True),
                      'type': links[i].get('href').split('/')[3]})

for i in range(0, 1000):
    print(
        games[i]['rank'],
        games[i]['title'],
        games[i]['release'],
        games[i]['date'],
        games[i]['price'],
        games[i]['price_discounted'],
        games[i]['id_num'],
        games[i]['id_title'],
        games[i]['type'])

# db connection
connect = odbc.odbc('oasis')
db = connect.cursor()
sql = '''
    INSERT INTO oasis.games(title, ranking, price, price_discounted, date, release_date, type, id_title, id_num) VALUES ("%s","%d","%d","%d","%s","%s","%s","%s","%s")
    '''
for i in range(0, 1000):
    db.execute(sql % (
        games[i]['title'], games[i]['rank'], games[i]['price'], games[i]['price_discounted'], games[i]['date'],
        games[i]['release'], games[i]['type'], games[i]['id_title'], games[i]['id_num']))

# check db
Checker.check('games')