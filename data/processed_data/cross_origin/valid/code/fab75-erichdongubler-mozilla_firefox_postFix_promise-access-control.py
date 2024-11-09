def main(request, response):
    <fix/>allow = request.GET.first(b"allow", b"false")</fix>

    <fix/>headers = [(b"Content-Type", b"application/javascript")]
    if allow != b"false":
        headers.append((b"Access-Control-Allow-Origin", b"*"))</fix>

    <fix/>body = b"""</fix>
    	function handleRejectedPromise(promise) {
    		promise.catch(() => {});
    	}

    	(function() {
    		new Promise(function(resolve, reject) { reject(42); });
    	})();
    """

    return headers, body
