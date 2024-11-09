'''
Created on 2012-8-14

@author: diracfang
'''
import requests

def call_stream_api(url, access_token):
    headers = {'Cookie': 'access_token=%s' % access_token}
    r = requests.get(url, headers=headers)
    for line in r.iter_lines(chunk_size=8):
        if line:
            print line
        else:
            print 'keep-alive'
    
    
if __name__ == '__main__':
    <vul/>access_token = '9920e647907355f3756dad8b1477da4bcc6850fe'
    url = 'http://test.kan.sohu.com/api/2/sync/stream'</vul>
    call_stream_api(url, access_token)