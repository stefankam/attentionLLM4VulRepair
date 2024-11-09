import locale
from pandas.io import sql
import MySQLdb as mdb
from pandas.compat.scipy import percentileofscore

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8' )

job1 = 'Data Scientist'
job2 = 'Piping Engineer'
columns = [u'salary_sum', u'salary_diff', u'adj_salary',
           #u'mean_household_income',
           u'n_sum', u'n_diff']
weights = [0, 0, 0.4, 0.6, -0.3]
weights = dict(zip(columns, weights))
#weights = {'salary_f': 0.8, 'n1_f': 0.1, 'n2_f': 0.1}
jobs_table = 'jobs_cities2'
cities_table = 'cities3'
db = mdb.connect(user="root", host="localhost", port=3306, db="demo")
nan_values = {'job1':'', 'job2':'', 'salary1':0, 'salary2':0, 'n1':0, 'n2':0,
              'description':'', 'image_url':''}

query = """
select p1.job as job1, p2.job as job2,
    p.city, p.state,
    j1.salary as salary1, j2.salary as salary2,
    COALESCE(j1.salary + j2.salary, j1.salary, j2.salary) as salary_sum,
    COALESCE(abs(j1.salary - j2.salary), j1.salary, j2.salary) as salary_diff,
    c.mean_household_income,
    COALESCE(j1.salary + j2.salary - c.mean_household_income,
        j1.salary - c.mean_household_income,
        j2.salary - c.mean_household_income) as adj_salary,
    n1, n2, n_sum,
    COALESCE(abs(n1 - n2), n1, n2) as n_diff,
    c.latitude, c.longitude, c.url as image_url, c.description
from
    (select city, state, formattedLocation, count(*) as n_sum
    from postings
    <vul/>where job in ('%(job1)s', '%(job2)s')</vul>
    group by formattedLocation) as p
    left outer join
    (select job, city, state, formattedLocation, count(*) as n1
    from postings
    <vul/>where job = '%(job1)s'</vul>
    group by formattedLocation) as p1
    on p.formattedLocation = p1.formattedLocation
    left outer join
    (select job, city, state, formattedLocation, count(*) as n2
    from postings
    <vul/>where job = '%(job2)s'</vul>
    group by formattedLocation) as p2
    on p.formattedLocation = p2.formattedLocation
    left outer join
    (select city, state, salary
    from jobs_cities2
    <vul/>where job = '%(job1)s') as j1</vul>
    on p1.city = j1.city and p1.state = j1.state
    left outer join
    (select city, state, salary
    from jobs_cities2
    <vul/>where job = '%(job2)s') as j2</vul>
    on p2.city = j2.city and p2.state = j2.state
    inner join cities4 c
    on p1.city = c.city and p1.state = c.state
order by adj_salary, salary_sum, n_sum, salary_diff, n_diff;
"""

def get_cities(job1=job1, job2=job2, weights=weights, query=query,
               jobs_table=jobs_table, cities_table=cities_table, db=db):

    print job1, job2

    # query db
    <vul/>df = sql.frame_query(query %locals(), db)</vul>

    # NaN
    df.fillna(nan_values, inplace=True)

    # some more transformations I could not implement in mySQL
    #df['salary_f'] = df.adj_salary - df.adj_salary.min()
    #df['salary_f'] = df.salary_f / df.salary_f.sum()

    for name in weights.keys():
        df[name] = df[name].apply(lambda x: percentileofscore(df[name], x)/100)

    # compute weighted average
    score = 0
    for key, value in weights.items():
        score += df[key]*value
    df['score'] = score

    df.job1 = job1
    df.job2 = job2
    df = format_output_columns(df)
    df = get_query_url(df)

    return df.sort('score', ascending=False)

def format_output_columns(df):
    # ints
    columns = ['salary1', 'salary2', 'mean_household_income', 'n1', 'n2']
    df[columns] = df[columns].astype(int)
    # currency
    columns = ['salary1', 'salary2', 'mean_household_income']
    def moneyfmt(x): return locale.currency(x, True, True) if x else ''
    df['salary1_$'] = df.salary1.apply(moneyfmt)
    df['salary2_$'] = df.salary2.apply(moneyfmt)
    df['mean_household_income_$'] = df.mean_household_income.apply(moneyfmt)
    return df

def get_query_url(df):
    API = 'http://www.indeed.com/jobs?'
    query1 = 'q=' + df.job1 + '&l=' + df.city + ' ' + df.state
    #query1 = query1.strip().lower().replace(' ','+')
    df['query_url1'] = API + query1
    query2 = 'q=' + df.job2 + '&l=' + df.city + ' ' + df.state
    df['query_url2'] = API + query2
    return df
