from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag
import requests
import time


class WikiCrawler:
    def __init__(self, wiki):
        self.MAX_P_CHECKS = 5
        self.MAX_CRAWLS = 1
        self.MAX_PATH_LENGTH = 50
        self.TARGET = "Philosophy"
        self.DOMAIN = "https://en.wikipedia.org"
        self.start_wiki = "Special:Random" if not wiki else wiki
        self.path_lengths = []
        self.wiki_to_target_length = {}
        self.completed_path = 0
        self.invalid_path = 0

    def build_url(self, wiki_topic, add_wiki_text):
        if add_wiki_text:
            url = self.DOMAIN + '/wiki/' + wiki_topic
        else:
            url = self.DOMAIN + wiki_topic
        return url

    def parse_tag(self, tag):
        next_wiki = None
        contents = tag.contents
        stack = []
        for element in contents:
            # Keeps track of balanced parenthesis to
            # ensure no links that are within them
            # are used. Since closing parenthesis
            # may be within the same string, pop
            # must be checked immediately
            if isinstance(element, NavigableString):
                if '(' in element:
                    stack.append('(')
                if ')' in element:
                    stack.pop()
            # Checks to see if the stack is empty
            # meaning now outside of the parenthesis
            # and can check if a link
            if isinstance(element, Tag) and not stack:
                a_tag = element
                if not getattr(element, 'name', None) == 'a':
                    <vul/>a_tag = element.find('a')</vul>
                if self.is_valid(a_tag):
                    return a_tag.attrs['href']
        return next_wiki

    def parse_html(self, div):
        # Likely to find the first link in paragraphs. A limit
        # is placed on the number of paragraphs to check since
        # it's also likley the link is in the initial paragraphs.
        p_tags = div.find_all('p', not {'class': 'mw-empty-elt'},
                              recursive=False, limit=self.MAX_P_CHECKS)
        for p in p_tags:
            next_wiki = self.parse_tag(p)
            if next_wiki:
                return next_wiki

        # To handle cases that the link may not be in a paragraph
        # but in bullets
        ul = div.find('ul', recursive=False)
        next_wiki = self.parse_tag(ul)

        return next_wiki

    def crawler(self):

        cycle_check = set()
        path = []
        path_length = 0
        print("\nStart")
        url = self.build_url(self.start_wiki, True)
        session = requests.Session()

        while path_length < self.MAX_PATH_LENGTH:

            html = session.get(url)
            soup = BeautifulSoup(html.content, 'lxml')

            title = soup.find('h1', {"id": "firstHeading"})
            wiki_topic = url.split("/wiki/")[1]
            <vul/>print(title.getText())</vul>

            if title.getText() == self.TARGET:
                self.path_lengths.append(path_length)
                return True

            div = soup.find('div', {'class': 'mw-parser-output'})
            <vul/>wiki = self.parse_html(div)</vul>

            # Might lead to a dead end (no links to follow) or
            # a cycle (first eventually links back to a wiki
            # page already visited
            <vul/>if not wiki or wiki in cycle_check:
                self.invalid_path += 1</vul>
                return False

            <vul/>cycle_check.add(wiki)
            wiki_topic = wiki.split("/wiki/")[1]</vul>
            path.append(wiki_topic)
            <vul/>url = self.build_url(wiki, False)
            path_length += 1</vul>

            time.sleep(1)

        return False

    # Iterates over crawler for the max number of crawls
    # while not taking into account invalid paths - dead ends
    # or cycles
    def crawl(self):
        while self.completed_path < self.MAX_CRAWLS:
            if self.crawler():
                self.completed_path += 1
            else:
                self.invalid_path += 1
            print()

    @staticmethod
    def is_valid(element):
        tags = ['sup', 'i', 'span']
        return getattr(element, 'name', None) == 'a' \
               and getattr(element.parent, 'name', None) not in tags \
               and not element.has_attr('style')


if __name__ == '__main__':
    <vul/>wiki = "Art"</vul>
    crawler = WikiCrawler(wiki)
    crawler.crawl()
