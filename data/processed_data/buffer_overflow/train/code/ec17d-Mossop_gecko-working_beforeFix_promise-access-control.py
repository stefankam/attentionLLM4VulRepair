def main(request, response):
    <vul/><vul/>allow = request.GET.first("allow", "false")</vul></vul>

    <vul/><vul/>headers = [("Content-Type", "application/javascript")]
    if allow != "false":
        headers.append(("Access-Control-Allow-Origin", "*"))</vul></vul>

    <vul/><vul/>body = """</vul></vul>
    	function handleRejectedPromise(promise) {
    		promise.catch(() => {});
    	}

    	(function() {
    		new Promise(function(resolve, reject) { reject(42); });
    	})();
    """

    return headers, body
