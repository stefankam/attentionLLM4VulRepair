<vul/></vul>import requests
from bs4 import BeautifulSoup
import difflib
import random
import string
import re


# source: https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits
def generate_random_value():
    return "".join(random.choices(string.ascii_letters + string.digits, k=12))


def get_page_text(url):
    page = requests.get(url, verify=False)
    soup = BeautifulSoup(page.content, "html.parser")
    return str(soup)


def post_page_text(url, data):
    page = requests.post(url, data=data, verify=False)
    soup = BeautifulSoup(page.content, "html.parser")
    return str(soup)


def get_page_htmli_header(url, data):
    page = requests.get(url, headers=data, verify=False)
    soup = BeautifulSoup(page.content, "html.parser")
    return str(soup)

def get_page_htmli_cookie(url, data):
    page = requests.get(url, cookies=data, verify=False)
    soup = BeautifulSoup(page.content, "html.parser")
    return str(soup)

def compare_page_htmli(url, inj, cookie):
    page_text_normal = get_page_text(url)
    if cookie == False:
        <vul/>header = {
        "User-Agent": inj
        }</vul>
        page = get_page_htmli_header(url, header)
    else:
        get_header = requests.get(url)
        get_cookie = get_header.cookies
        <vul/>get_cookie_name = list(get_cookie.keys())[0] #ostatni</vul>
        get_cookie.update({get_cookie_name: get_cookie.get(get_cookie_name) + inj})
        page = get_page_htmli_cookie(url, get_cookie)
   
    diff = difflib.unified_diff(
        page_text_normal.splitlines(), page.splitlines(), lineterm=""
    )

    output = "\n".join(diff)
    regex = r"<img([\w\W]+?)/>"

    return re.findall(regex, output)


def compare_pages_post(url1, data, regex=r">([^<]+)<"):
    page_text1 = get_page_text(url1)
    page_text2 = post_page_text(url1, data)

    # source https://docs.python.org/3/library/difflib.html
    diff = difflib.unified_diff(
        page_text1.splitlines(), page_text2.splitlines(), lineterm=""
    )

    output = "\n".join(diff)

    return re.findall(regex, output)


def compare_pages(url1, url2, regex=r">([^<]+)<"):
    page_text1 = get_page_text(url1)
    page_text2 = get_page_text(url2)

    # source https://docs.python.org/3/library/difflib.html
    diff = difflib.unified_diff(
        page_text1.splitlines(), page_text2.splitlines(), lineterm=""
    )

    output = "\n".join(diff)

    return re.findall(regex, output)


def list_to_dict(list):
    result = {}

    for item in list:
        key, value = item.split(":", 1)
        key = key.strip().strip("'")
        value = value.strip().strip("'")
        result[key] = value

    return result


def find_forms(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    forms = soup.find_all("form")
    form_data = []

    for form in forms:
        form_details = {}
        inputs = form.find_all(["input", "textarea", "select"])

        for input_element in inputs:
            name = input_element.get("name")
            value = input_element.get("value", "")

            if name:
                form_details[name] = value

        form_data.append(form_details)

    return form_data


def generate_data_forms(form_data, custom_value):
    full_forms = []
    random_value = generate_random_value()
    for form_details in form_data:
        filled_form = {}
        keys = list(form_details.keys())
        if "page" in keys:
            page_index = keys.index("page")
        else:
            page_index = -1
        for idx, key in enumerate(keys):
            if key == "page":
                filled_form[key] = form_details[key]
            elif idx == page_index + 1:
                filled_form[key] = custom_value
            else:
                filled_form[key] = random_value

            full_forms.append(filled_form)

    return full_forms
