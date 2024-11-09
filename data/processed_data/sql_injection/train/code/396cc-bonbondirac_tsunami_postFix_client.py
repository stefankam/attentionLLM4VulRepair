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
    <fix/>access_token = '19a77650b9b829caa401ed219726410277928eac'
    url = 'http://localhost:8888/api/2/sync/stream/'</fix>
    call_stream_api(url, access_token)