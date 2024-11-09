from utility import *
import json

def createGetScript(endpoint, params):
    script = 'curl '+start_url+endpoint+'?'
    keys = params.keys()
    values = params.values()
    pair = [keys[i]+'='+values[i] for i in range(len(keys))]
    evil_param = '&'.join(pair)
    script+=evil_param
    return script

def createPostScript(endpoint, params):
    keys = params.keys()
    values = params.values()
    pair = [keys[i]+'='+values[i] for i in range(len(keys))]
    evil_param = '&'.join(pair)
    script = 'curl -d ' + '"'+ evil_param  +'" '+'-X POST '+start_url+endpoint
    return script

def genDT(endpoint,params,method):
    scope = {
        'class':DT,
        'results':{
            start_url: [
                {
                    'endpoint': endpoint,
                    'params': params,
                    'method': method
                }
            ]
        }
    }    

    script = ''
    if method == 'GET':
        script = createGetScript(endpoint, params)
        
    return scope, script

def genSI(endpoint, params, method):
    scope = {
        'class':SI,
        'results':{
            start_url: [
                {
                    'endpoint': endpoint,
                    'params': params,
                    'method': method
                }
            ]
        }
    }

    if method == 'POST':
        script = createPostScript(endpoint,params)
    
    return scope, script

def genSCI():
    pass

def genSSCI():
    pass

def genCSRF():
    pass

<fix/>def genOR(endpoint, params, method):
    scope = {
        'class':OR,
        'results':{
            start_url: [
                {
                    'endpoint': endpoint,
                    'params': params,
                    'method': method
                }
            ]
        }
    }    

    script = ''
    if method == 'GET':
        script = createGetScript(endpoint, params)
        
    return scope, script</fix>  

render = {
    DT: genDT,
    SI: genSI,
    CSRF: genCSRF,
    OR: genOR,
    SSCI: genSSCI,
    SCI: genSCI
}

class generator(object):
    def __init__(self,category):
        self.scope = {}
        self.category = category
        self.cate_str = '_'.join(category.split(' '))
        self.count = 0
        
    def updateScope(self,scope):
        if(self.count):
            self.scope['results'][start_url]+=scope['results'][start_url]
        else:
            self.scope=scope
        self.count += 1
        
    def saveScript(self,script):
        script_name = 'result/'+self.cate_str+'_attack'+str(self.count)+'.sh'
        with open(script_name, 'w') as f:
            f.write(script)

    def saveScope(self):
        file_name = 'result/'+self.cate_str+'_scope.json'
        with open(file_name,'w+') as f:
            json.dump(self.scope,f)

