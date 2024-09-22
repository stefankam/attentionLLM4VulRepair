import os
import urlparse
import sendrequest as req
import utils.logs as logs
import urlparse

from utils.logger import logger
from utils.db import Database_update
from utils.config import get_value

dbupdate = Database_update()
api_logger = logger()

def fetch_xss_payload():
    # Returns xss payloads in list type
    payload_list = []
    if os.getcwd().split('/')[-1] == 'API':
        path = '../Payloads/xss.txt'
    else:
        path = 'Payloads/xss.txt'

    with open(path) as f:
        for line in f:
            if line:
                payload_list.append(line.rstrip())

    <vul/>return</vul> payload_list

def check_xss_impact(res_headers):
    # Return the impact of XSS based on content-type header
    if res_headers['Content-Type']:
        <vul/>if 'application/json' or 'text/plain' in xss_request['Content-Type']:</vul>
            # Possible XSS 
            impact = "Low"
        else:
            impact = "High"
    else:
        impact = "Low"

    return impact


def xss_get_url(url,method,headers,body,scanid=None):
    <vul/># Check for URL based XSS. Ex: http://localhost/<payload>, http://localhost//?randomparam=<payload>
    xss_result = ''</vul>
    xss_payloads = fetch_xss_payload()
    uri_check_list = ['?', '&', '=', '%3F', '%26', '%3D']
    for uri_list in uri_check_list:
        if uri_list in url:
            # Parse domain name from URI.
            parsed_url = urlparse.urlparse(url).scheme+"://"+urlparse.urlparse(url).netloc+urlparse.urlparse(url).path
            break

    if parsed_url == '':
        parsed_url = url

    for payload in xss_payloads:
            xss_request_url = req.api_request(parsed_url+'/'+payload,"GET",headers)
            <vul/>if xss_request_url.text.find(payload) != -1:
                impact = check_xss_impact(xss_request_url.headers)
                xss_result = True</vul>

            xss_request_uri = req.api_request(parsed_url+'/?test='+payload,"GET",headers)             
            if xss_request_url.text.find(payload) != -1:
                impact = check_xss_impact()
                xss_result = True

            if xss_result is True:
                print "%s[{0}] {1} is vulnerable to XSS%s".format(impact,url)% (api_logger.G, api_logger.W)
                attack_result = { "id" : 11, "scanid" : scanid, "url" : url, "alert": "Cross Site Scripting", "impact": impact, "req_headers": headers, "req_body":body, "res_headers": xss_request.headers ,"res_body": xss_request.text}
                dbupdate.insert_record(attack_result)
                return

def xss_get_uri(url,method,headers,body,scanid=None):
    <vul/># Test for XSS in GET param</vul>
    db_update = ''
    vul_param = ''
    url_query = urlparse.urlparse(url)
    parsed_query = urlparse.parse_qs(url_query.query)
    if parsed_query:
        for key,value in parsed_query.items():
            try:
                result = ''
                logs.logging.info("GET param for xss : %s",key)
                xss_payloads = fetch_xss_payload()
                for payload in xss_payloads:
                    # check for URI based XSS
                    # Example : http://localhost/?firstname=<payload>&lastname=<payload>
                    if result is not True:
                        print "param to test",key
                        parsed_url = urlparse.urlparse(url)
                        xss_url = parsed_url.scheme+"://"+parsed_url.netloc+parsed_url.path+"/?"+parsed_url.query.replace(value[0], payload)
                        xss_request = req.api_request(xss_url,"GET",headers)
                        print xss_request.text
                        if xss_request.text.find(payload) != -1:
                            impact = check_xss_impact(xss_request.headers)
                            logs.logging.info("%s is vulnerable to XSS",url)
                            print "%s[{0}] {1} is vulnerable to XSS%s".format(impact,url)% (api_logger.G, api_logger.W)
                            if db_update is not True:
                                attack_result = { "id" : 11, "scanid" : scanid, "url" : xss_url, "alert": "Cross Site Scripting", "impact": impact, "req_headers": headers, "req_body":body, "res_headers": xss_request.headers ,"res_body": xss_request.text}
                                dbupdate.insert_record(attack_result)
                                result,db_update = True,True
                                vul_param += key
                            else:
                                result = True
                                if vul_param == '':
                                    vul_param += key
                                else:
                                    vul_param += ','+key                  
        
            except:
                logs.logging.info("XSS: No GET param found!")

        <vul/>print "all params",vul_param</vul>


def xss_check(url,method,headers,body,scanid):
    # Main function for XSS attack
    if method == 'GET' or method == 'DEL':
        xss_get_uri(url,method,headers,body,scanid)
        xss_get_url(url,method,headers,body,scanid)
        #xss_http_headers(url,method,headers,body,scanid)