<fix/></fix>import os
import urlparse
import sendrequest as req
import utils.logs as logs
import urlparse
import time

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

    return payload_list

def check_xss_impact(res_headers):
    # Return the impact of XSS based on content-type header
    if res_headers['Content-Type']:
        <fix/>if 'application/json' or 'text/plain' in res_headers['Content-Type']:</fix>
            # Possible XSS 
            impact = "Low"
        else:
            impact = "High"
    else:
        impact = "Low"

    return impact


def xss_http_headers(url,method,headers,body,scanid=None):
    # This function checks different header based XSS.
    # XSS via Host header (Limited to IE)
    # Reference : http://sagarpopat.in/2017/03/06/yahooxss/
    temp_headers = {}
    temp_headers.update(headers)
    xss_payloads = fetch_xss_payload()
    for payload in xss_payloads:
        parse_domain = urlparse.urlparse(url).netloc
        host_header = {"Host" : parse_domain + '/' + payload}
        headers.update(host_header)
        host_header_xss = req.api_request(url, "GET", headers)
        if host_header_xss.text.find(payload) != -1:
            impact = "Low"
            attack_result = { "id" : 11, "scanid" : scanid, "url" : url, "alert": "Cross Site Scripting", "impact": impact, "req_headers": headers, "req_body":body, "res_headers": host_header_xss.headers ,"res_body": xss_request.text}
            print "%s[{0}] {1} is vulnerable to XSS%s".format(impact,url)% (api_logger.G, api_logger.W)
            dbupdate.insert_record(xss_http_headers)
            break

    # Test for Referer based XSS 
    for payload in xss_payloads:
        referer_header_value = 'http://attackersite.com?test='+payload
        referer_header = {"Referer" : referer_header_value}
        temp_headers.update(referer_header)
        ref_header_xss = req.api_request(url, "GET", temp_headers)
        if ref_header_xss.text.find(payload) != -1:
            impact = check_xss_impact(temp_headers)
            print "%s[{0}] {1} is vulnerable to XSS via referer header%s".format(impact,url)% (api_logger.G, api_logger.W)
            attack_result = { "id" : 11, "scanid" : scanid, "url" : url, "alert": "Cross Site Scripting via referer header", "impact": impact, "req_headers": temp_headers, "req_body":body, "res_headers": ref_header_xss.headers ,"res_body": ref_header_xss.text}
            dbupdate.insert_record(attack_result)
            break


def xss_get_url(url,method,headers,body,scanid=None):
    <fix/># Check for URL based XSS. 
    # Ex: http://localhost/<payload>, http://localhost//?randomparam=<payload>
    result = ''</fix>
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
            <fix/>if result is not True:
                if xss_request_url.text.find(payload) != -1:
                    impact = check_xss_impact(xss_request_url.headers)
                    <fix/>attack_result = { "id" : 11, "scanid" : scanid, "url" : url, "alert": "Cross Site Scripting", "impact": impact, "req_headers": headers, "req_body":body, "res_headers": xss_request_url.headers ,"res_body": xss_request_url.text}</fix>
                    dbupdate.insert_record(attack_result)
                    result = True</fix>

            xss_request_uri = req.api_request(parsed_url+'/?test='+payload,"GET",headers)             
            if xss_request_url.text.find(payload) != -1:
                impact = check_xss_impact()
                print "%s[{0}] {1} is vulnerable to XSS%s".format(impact,url)% (api_logger.G, api_logger.W)
                attack_result = { "id" : 11, "scanid" : scanid, "url" : url, "alert": "Cross Site Scripting", "impact": impact, "req_headers": headers, "req_body":body, "res_headers": xss_request_url.headers ,"res_body": xss_request_url.text}
                dbupdate.insert_record(attack_result)
                

def xss_get_uri(url,method,headers,body,scanid=None):
    <fix/># This function checks for URI based XSS. 
    # http://localhost/?firstname=<payload>&lastname=<payload></fix>
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
                        parsed_url = urlparse.urlparse(url)
                        xss_url = parsed_url.scheme+"://"+parsed_url.netloc+parsed_url.path+"/?"+parsed_url.query.replace(value[0], payload)
                        xss_request = req.api_request(xss_url,"GET",headers)
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

        <fix/>if vul_param:
            # Update all vulnerable params to db.
            dbupdate.update_record({"scanid": scanid}, {"$set" : {"scan_data" : vul_param+" are vulnerable to XSS"}})</fix>


def xss_check(url,method,headers,body,scanid):
    # Main function for XSS attack
    if method == 'GET' or method == 'DEL':
        xss_get_uri(url,method,headers,body,scanid)
        xss_get_url(url,method,headers,body,scanid)
    
    xss_http_headers(url,method,headers,body,scanid)