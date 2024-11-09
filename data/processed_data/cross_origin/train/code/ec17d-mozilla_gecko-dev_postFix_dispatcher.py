# A server used to store and retrieve arbitrary data.
# This is used by: ./dispatcher.js
def main(request, response):
    <fix/>response.headers.set(b'Access-Control-Allow-Origin', b'*')
    response.headers.set(b'Access-Control-Allow-Methods', b'OPTIONS, GET, POST')
    response.headers.set(b'Access-Control-Allow-Headers', b'Content-Type')
    response.headers.set(b'Cache-Control', b'no-cache, no-store, must-revalidate')
    if request.method == u'OPTIONS': # CORS preflight
        return b''</fix>

    <fix/>uuid = request.GET[b'uuid']</fix>

    <fix/>if request.method == u'POST':</fix>
        return request.server.stash.put(uuid, request.body)
    else:
        body = request.server.stash.take(uuid)
        if body is None:
            <fix/>return b'not ready'</fix>
        else:
            return body
