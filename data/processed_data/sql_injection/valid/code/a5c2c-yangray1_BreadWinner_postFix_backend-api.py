from flask import Flask
from flask import request
import simplejson as json
import psycopg2

""" Macros for relation and column names """
client_table_name = "\"Client\""
client_client_id_col = "\"ClientID\""
client_client_rating_col = "\"Client Rating\""

client_ratings_table_name = "\"Client Ratings\""
client_ratings_client_id_col = "\"ClientID\""
client_ratings_reviewer_id_col = "\"ReviewerID\""
client_ratings_comments_col = "\"Comments\""
client_ratings_rating_col = "\"Rating\""

cook_table_name = "\"Cook\""
cook_cook_id_col = "\"CookID\""
cook_cook_rating_col = "\"Cook Rating\""

cook_ratings_table_name = "\"Cook Rating\""
cook_ratings_cook_id_col = "\"CookID\""
cook_ratings_reviewer_id_col = "\"ReviewerID\""
cook_ratings_comments_col = "\"Comments\""
cook_ratings_rating_col = "\"Rating\""

listing_table_name = "\"Listing\""
listing_listing_id_col = "\"ListingID\""
listing_cook_id_col = "\"CookID\""
listing_food_name_col = "\"Food Name\""
listing_price_col = "\"Price\""
listing_location_col = "\"Location\""
listing_image_col = "\"Image\""

listing_tags_table_name = "\"Listing Tags\""
listing_tags_listing_id_col = "\"ListingID\""
listing_tags_tag_col = "\"Tag\""

order_table_name = "\"Order\""
order_client_id_col = "\"ClientID\""
order_listing_id_col = "\"ListingID\""
order_status_col = "\"Status\""
order_time_of_order_col = "\"Time of Order\""

user_table_name = "\"User\""
user_user_id_col = "\"UserID\""
user_password_col = "\"Password\""
user_fname_col = "\"FName\""
user_lname_col = "\"LName\""

""" Database login details """
db_host = "mydbinstance.cqzm55sjgiup.us-east-1.rds.amazonaws.com"
db_name = "csc301breadwiener"
db_user = "csc301breadwiener"
db_password = "team7ithink"

conn = psycopg2.connect(host=db_host, database=db_name, user=db_user, password=db_password)
app = Flask(__name__)

##################################################
def removeQuotes(stringy):
    """ Removes the first and last characters (double quotes) from a string, and then return it """
    return stringy[1:-1]


#--------------------------------------------------- GET ALL LISTINGS ---------------------------------------------------#
@app.route('/api/getAllListings', methods=['GET'])
def getAllListings():
    all_rows = []

    search_all = conn.cursor()
    search_all.execute("SELECT {}, {}, {}, {},"
                         " {}, {} FROM public.{}".format(listing_listing_id_col,
                                                                          listing_cook_id_col,
                                                                          listing_food_name_col,
                                                                          listing_price_col,
                                                                          listing_location_col,
                                                                          listing_image_col,
                                                                          listing_table_name))

    single_row = search_all.fetchone()

    while single_row is not None:
        all_rows.append(single_row)
        single_row = search_all.fetchone()

    search_all.close()

    rows_to_json(all_rows)  # want to convert each row into a JSON string

    return json.dumps({'data': all_rows})  # convert to string before returning


#--------------------------------------------------- ADD LISTING ---------------------------------------------------#

@app.route('/api/add', methods=['GET', 'POST'])
def addReq():
    if request.method == "GET":
        return printTables()
    elif request.method == "POST":
        addToDB(request.get_json())
        conn.commit()
        return "Success"

def encase_in_quotes(stringy):
    return "\"" + stringy + "\""


"""
Adds the Listing entry to the PSQL database with the given JSONdata
JSON format is a dictionary where the keys are the column names of the listing, along with
a key "tagList" which is a list of tags:

"""


def addToDB(json_data):
    cur = conn.cursor()
    json_dict = json_data

    list_id = getListId()
    cook_id = json_dict[removeQuotes(listing_cook_id_col)]
    food_name = json_dict[removeQuotes(listing_food_name_col)]
    price = json_dict[removeQuotes(listing_price_col)]
    loc = json_dict[removeQuotes(listing_location_col)]
    image = json_dict[removeQuotes(listing_image_col)]
    tags = json_dict["tags"]

    <fix/>sql = "INSERT INTO %s VALUES (%s, %s, %s, %s, %s, %s)"
	cur.execute(sql, (listing_table_name, list_id, cook_id, food_name, price, loc, image))</fix>

    addTags(tags, list_id)


def addTags(tag_list, listing_id):
    """
    Adds a list of tags tag_list for a given listing with listing_id to the database
    """
    cur = conn.cursor()
    for x in tag_list:
        sql = "INSERT INTO {} VALUES {}".format(listing_tags_table_name, str((listing_id, x)))
        cur.execute(sql)


def getListId():
    """ Returns an unused listing_id """
    cur = conn.cursor()
    sql = "SELECT max({}) FROM {}".format(listing_listing_id_col,
                                          listing_table_name)
    cur.execute(sql)
    maxID = cur.fetchone()
    if maxID[0] == None:
        return 1
    else:
        return maxID[0] + 1


def printTables():
    cur = conn.cursor()
    strout = "--------------------------ListingTable---------------------------<br>"
    sql = "SELECT * FROM {}".format(listing_table_name)
    cur.execute(sql)
    listings = cur.fetchall()
    for x in listings:
        for y in x:
            strout = strout + str(y) + "||	"
        strout = strout + "<br>"
    sql = "SELECT * FROM {}".format(listing_tags_table_name)
    cur.execute(sql)
    listings = cur.fetchall()
    strout += "<br><br><br>--------------------------TagTable-------------------------<br>"
    for x in listings:
        for y in x:
            strout = strout + str(y) + "	"

        strout = strout + "<br>"
    return strout


#--------------------------------------------------- CANCEL ---------------------------------------------------#


@app.route('/api/cancel/<int:clientId>/<int:listingId>', methods=['GET'])
def cancel(clientId, listingId):
    """
    Cancels the order with specified client id and listing id and returns it.
    returns 'order not found' if the client id and listing id do not exist as a key or if the listing has already
    been canceled or fulfilled.
    """

    in_progress = get_in_progress_order(clientId, listingId)

    if in_progress:
        cancel_order(clientId, listingId)
        output = order_to_json(in_progress)  # want to convert each row into a JSON string

        return output  # convert to string before returning
    else:
        return 'order not found'


def get_in_progress_order(clientId, listingId):
    """
    Return the in progress order that corresponds with ClientId and ListingID
    """
    matched_rows = []

    order = conn.cursor()
    order.execute("SELECT t1.\"ClientID\", t1.\"ListingID\", t1.\"Status\", t1.\"Time of Order\" from public.\"Order\""
                  " as t1 WHERE t1.\"ClientID\" = " + str(clientId) + " AND \"ListingID\" = " + str(listingId) +
                  " AND t1.\"Status\" = \'In progress\'")

    order_row = order.fetchone()

    while order_row is not None:
        matched_rows.append(order_row)
        order_row = order.fetchone()

    order.close()

    return matched_rows


def cancel_order(clientId, listingId):
    """
    given a clientId and listingId cancel the order in progress associated with them
    """
    order = conn.cursor()
    order.execute(
        "UPDATE public.\"Order\" SET \"Status\" = 'Canceled' WHERE \"ClientID\" = " + str(clientId) +
        " AND \"ListingID\" = " + str(listingId) + " AND \"Status\" = \'In progress\'")
    conn.commit()

    order.close()


def order_to_json(rows):
    """
    Takes in a list of tupples for the Orders schema and returns a json formated representation of the data.
    """
    string = ""
    for i in range(len(rows)):
        string += json.dumps({'ClientID': rows[i][0],
                              'ListingID': rows[i][1],
                              'Status': rows[i][2],
                              'DateTime': rows[i][3].__str__()})
        if i != len(rows) - 1:
            string += ","

    return string


#--------------------------------------------------- getUserOrders ---------------------------------------------------#


@app.route('/api/getUserOrders/<int:clientId>', methods=['GET'])
def getUserOrders(clientId):
    """
    Retruns a list of jsons representing tupples in the Orders table for the given client
    """

    in_progress = queryOrderUsingClientID(clientId)

    output = order_to_json(in_progress)  # want to convert each row into a JSON string

    return "[" + output + "]"  # convert to string before returning


def queryOrderUsingClientID(clientId):
    """
    Return a list of Order tuples belonging to the client with the given id.
    """
    matched_rows = []

    orders = conn.cursor()
    orders.execute("SELECT t1.\"ClientID\", t1.\"ListingID\", t1.\"Status\", t1.\"Time of Order\" from public.\"Order\""
                   " as t1 WHERE t1.\"ClientID\" = " + str(clientId))

    order_row = orders.fetchone()

    while order_row is not None:
        matched_rows.append(order_row)
        order_row = orders.fetchone()

    orders.close()

    return matched_rows


#--------------------------------------------------- MARK AS COMPLETE ---------------------------------------------------#


completed = "\'Completed\'"


@app.route("/api/markComplete/<int:clientID>/<int:listingID>", methods=['GET'])
def mark_as_complete(clientID, listingID):
    """ A function that changes the status of the order with listing id listing_id to complete.
        Returns "Success" on a sucessful change of the listing id's order to complete.

        @param clientID: the client id number to change the status.
        @param listingID: the listing id number to change the status.
        @rtype: str
    """

    sql = \
        """
            UPDATE public.{}
            SET {} = {}
            WHERE {} = {} AND {} = {}
        """.format(order_table_name, order_status_col, completed, order_listing_id_col, str(listingID),
                   order_client_id_col, str(clientID))

    cur = conn.cursor()
    try:
        cur.execute(sql)
        conn.commit()
    except Exception as e:
        raise Exception(e)

    # Check to see if a row in the database has been updated.
    if cur.rowcount == 0:
        raise Exception("The status of listing id's order was not changed. ClientID or ListingID may be out of range.")
    return "Success"


#--------------------------------------------------- SEARCH ---------------------------------------------------#


@app.route('/api/search/<string:search_query>', methods=['GET'])
def search(search_query):
    """
    Return a string representation of a list of JSON objects. This list contains
    objects that correspond to listings that match names or tags in the search query.
    """
    # separate words in search_query with '+' in place of spaces
    search_terms = search_query.split('+')

    # want to remove whitespace and empty elements from the list
    search_terms_filtered = []

    for search_term in search_terms:
        if not search_term.isspace() and not search_term == '':
            search_terms_filtered.append(search_term)

    matched_rows_by_name = get_rows_from_name(search_terms_filtered)

    matched_rows_by_tag = get_rows_from_tag(search_terms_filtered)

    matched_rows = matched_rows_by_name + matched_rows_by_tag

    unique_matched_rows = list(set(matched_rows))  # remove duplicate rows

    rows_to_json(unique_matched_rows)  # want to convert each row into a JSON string

    return json.dumps({'data': unique_matched_rows})  # convert to string before returning


def get_rows_from_name(search_terms):
    """
    Return a list of listing tuples whose Food Names correspond to words in search_terms.
    """
    matched_rows = []

    for search_term in search_terms:
        search_names = conn.cursor()
        search_names.execute("SELECT t1.{}, t1.{}, t1.{}, t1.{},"
                             " t1.{}, t1.{} FROM public.{} as t1"
                             " FULL OUTER JOIN public.{} as t2 ON t1.{} = t2.{} "
                             "WHERE UPPER(t1.{}) LIKE UPPER(\'%{}%\')".format(listing_listing_id_col,
                                                                              listing_cook_id_col,
                                                                              listing_food_name_col,
                                                                              listing_price_col,
                                                                              listing_location_col,
                                                                              listing_image_col,
                                                                              listing_table_name,
                                                                              listing_tags_table_name,
                                                                              listing_listing_id_col,
                                                                              listing_tags_listing_id_col,
                                                                              listing_food_name_col,
                                                                              search_term))

        search_names_row = search_names.fetchone()

        while search_names_row is not None:
            matched_rows.append(search_names_row)
            search_names_row = search_names.fetchone()

        search_names.close()

    return matched_rows


def get_rows_from_tag(search_terms):
    """
    Return a list of listing tuples whose tags correspond to words in search_terms.
    """
    matched_rows = []

    for search_term in search_terms:
        search_tags = conn.cursor()
        search_tags.execute("SELECT t1.{}, t1.{}, t1.{}, t1.{},"
                             " t1.{}, t1.{} FROM public.{} as t1"
                             " FULL OUTER JOIN public.{} as t2 ON t1.{} = t2.{} "
                             "WHERE UPPER(t2.{}) LIKE UPPER(\'%{}%\')".format(listing_listing_id_col,
                                                                              listing_cook_id_col,
                                                                              listing_food_name_col,
                                                                              listing_price_col,
                                                                              listing_location_col,
                                                                              listing_image_col,
                                                                              listing_table_name,
                                                                              listing_tags_table_name,
                                                                              listing_listing_id_col,
                                                                              listing_tags_listing_id_col,
                                                                              listing_tags_tag_col,
                                                                              search_term))

        search_tags_row = search_tags.fetchone()

        while search_tags_row is not None:
            matched_rows.append(search_tags_row)
            search_tags_row = search_tags.fetchone()

        search_tags.close()

    return matched_rows


def rows_to_json(rows):
    """
    Mutate rows such that each tuple in rows is converted to a JSON string representing the same information.
    """
    for i in range(len(rows)):
        rows[i] = json.dumps({'ListingID': rows[i][0],
                                'CookID': rows[i][1],
                                'Food Name': rows[i][2],
                                'Price': rows[i][3],
                                'Location': rows[i][4],
                                'Image': rows[i][5]})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
    # host="0.0.0.0", port=80
