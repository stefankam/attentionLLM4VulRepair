import model
import os, json
from flask import Flask, render_template, request
import MySQLdb as mdb

app = Flask(__name__)
db = mdb.connect(user="root", host="localhost", port=3306, db="demo")
cities_table = 'cities4'
JOBS_TABLE = 'postings'#jobs_cities2'
#weights = {'salary_f': 0.6, 'n1_f': 0.2, 'n2_f': 0.2}
output = ['city', 'latitude', 'longitude', 'image_url', 'description']
n_cities = 10

@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/slideshow")
def slideshow():
    return render_template('slideshow.html')

@app.route("/_exp")
def exp():
    #keys = ['url', 'name', 'description', 'image_url', 'count', 'nbackers',
     #          'count', 'prediction']
    #results = [dict(zip(keys, ['%d'%i]*len(keys))) for i in range(1,4)]
    df = model.get_cities().reset_index()
    results = df.T.to_dict().values()[:n_cities]
    return render_template('exp_cards.html', results=results)
    #return render_template('exp_sliders.html', results=results)

@app.route('/maps')
def maps():
    job1 = request.args.get('job1', None, type=str)
    job2 = request.args.get('job2', None, type=str)
    return render_template('maps.html', job1=job1, job2=job2)

@app.route('/results')
def results():
    job1 = request.args.get('job1')
    job2 = request.args.get('job2')
    df = model.get_cities(job1, job2).reset_index()
    results = df.T.to_dict().values()[:n_cities]
    return render_template('results.html', results=results)

@app.route('/waypoints')
def waypoints():
    job1 = request.args.get('job1', None, type=str)
    job2 = request.args.get('job2', None, type=str)
    df = model.get_cities(job1, job2)#, weights, jobs_table, cities_table, db)
    df = df.reset_index()#df = df[output].head(n_cities)
    cities = []
    for i in range(n_cities):
        cities.append({i: (df.ix[i, 'city'],
                           df.ix[i, 'latitude'], df.ix[i, 'longitude'],
                           df.ix[i, 'image_url'], df.ix[i, 'description'])})
    print cities
    return json.dumps(cities)
    #return cities.to_json(orient='split')#json.dumps(list(cities))#address)

def jobs(table=JOBS_TABLE):
    """Return list of projects."""
    query = 'select job from %s group by job' %table
    <fix/>df = model.sql.read_frame(query, db)</fix>
    return json.dumps(df.job.values.tolist())

# I have a dictionary that holds functions
# that respond to json requests
JSON = {
    'jobs': jobs,
    'job1': jobs,
    'job2': jobs
}

# jobs function is called here
@app.route("/json/<what>")
def ajson(what):

    return JSON[what]()

@app.route('/<pagename>')
def regularpage(pagename=None):
    """
    Route not found by the other routes above. May point to a static template.
    """
    return "You've arrived at " + pagename
    #if pagename==None:
    #    raise Exception, 'page_not_found'
    #return render_template(pagename)

if __name__ == '__main__':
    print "Starting debugging server."
    app.run(debug=True, host='localhost', port=8000)
