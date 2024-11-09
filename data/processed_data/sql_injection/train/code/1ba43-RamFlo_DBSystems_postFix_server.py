from flask import Flask, request
from db import Database
from datetime import datetime, timedelta
from log import Logger
import sql_queries
import simplejson

logger = Logger().logger 
app = Flask(__name__)
port_number = 40327

database = Database()

cuisine_discovery_cache = {}
unique_ingredients_cache = {}
cache_persistence_time = timedelta(days=1)

geodist = 0.12  # used for restaurant geosearching - defines L1 radius


@app.route('/')
def index():
    return app.send_static_file('TheFoodCourt.html')


@app.route('/ingredient_prefix/<string:prefix>')
def get_ingredient_by_prefix(prefix):
    query_res = database.find_ingredients_by_prefix(prefix)
    if query_res == -1:
        return None
    logger.info("GET get_ingredient_by_prefix query")
    return query_res


@app.route('/get_cuisines')
def get_cuisines():
    query_res = database.get_cuisines()
    if query_res == -1:
        return None
    logger.info("GET get_cuisines query")
    return query_res


@app.route('/discover_new_cuisines/<int:cuisine_id>')
def discover_new_cuisines(cuisine_id):
    logger.info("GET discover_new_cuisines query")
    if cuisine_id in cuisine_discovery_cache:
        insert_time, data = cuisine_discovery_cache[cuisine_id]
        if datetime.now() < insert_time + cache_persistence_time:
            return data

    query_res = database.discover_new_cuisines_from_cuisine(cuisine_id)
    if query_res == -1:
        return None
    cuisine_discovery_cache[cuisine_id] = (datetime.now(), query_res)
    return query_res


@app.route('/restaurants/<ingredient>/')
def query_restaurants_by_ingredient(ingredient):
    """
    To query this method, use :
    '/restaurants/<ingredient>/?key=value&key=value&...' where keys are optional
    strings from ['loclat', 'loclng', 'price_category', 'online_delivery', 'min_review']
    for example: '/restaurants/flour/?min_review=3.5&price_category=2'
    """
    logger.info("GET query_restaurants_by_ingredient query")
    loclat, loclng = request.args.get('loclat'), request.args.get('loclng')
    price_category = request.args.get('price_category')
    online_delivery = request.args.get('online_delivery')
    min_review = request.args.get('min_review')
    base_query = sql_queries.restaurants_by_ingredient % ingredient
    if loclat != None and loclng != None:
        <fix/>try:
            lat_range = [float(loclat) - geodist, float(loclat) + geodist]
            lng_range = [float(loclng) - geodist, float(loclng) + geodist]
        except:
            logger.error("Error translating location to floats in "
                     "query_restaurants_by_ingredient, passed values: "
                     "%s, %s" % (loclat, loclng))
            return None</fix>
    else:
        lat_range = None
        lng_range = None
    filtered_query = database.restaurant_query_builder(base_query,
                                                       lat_range, lng_range,
                                                       price_category,
                                                       min_review, online_delivery)
    if filtered_query == -1:
        logger.error("Restaurant query builder failed to process inputs for "
                     "query_restaurants_by_ingredient: %s, %s, %s" %
                     (price_category,
                     min, online_delivery))
    limited_query = database.order_by_and_limit_query(filtered_query,
                                                    "agg_review DESC", 20)
    query_res = database.run_sql_query(limited_query)
    if query_res == -1:
        return None
    return query_res


@app.route('/restaurants/<saltiness>/<sweetness>/<sourness>/<bitterness>/')
def query_restaurants_by_taste(saltiness, sweetness, sourness, bitterness):
    """
    To query this method, use :
    '/restaurants/<saltiness>/<sweetness>/<sourness>/<bitterness>/?key=value&key=value&...'
    where keys are optional strings from
    ['loclat', 'loclng', 'price_category', 'online_delivery', 'min_review']
    and tastes (e.g. 'saltiness') are either 0 or 1
    for example: '/restaurants/0/1/0/1/?min_review=3.5&price_category=2'
    """
    logger.info("GET query_restaurants_by_taste query")
    try:
        saltiness, sweetness, sourness, bitterness = int(saltiness), \
                                                     int(sweetness), \
                                                     int(sourness), int(bitterness)
    except:
        logger.error("Error translating flavors to int in "
                     "query_restaurants_by_taste, passed values: "
                     "%s/%s/%s/%s" % (saltiness, sweetness, sourness, bitterness))
        return None

    restaurant_query = sql_queries.restaurant_by_taste % (
        get_taste_condition(saltiness),
        get_taste_condition(sweetness),
        get_taste_condition(sourness),
        get_taste_condition(bitterness),
        get_taste_condition(1 - saltiness),
        get_taste_condition(1 - sweetness),
        get_taste_condition(1 - sourness),
        get_taste_condition(1 - bitterness),
    )
    loclat, loclng = request.args.get('loclat'), request.args.get('loclng')
    price_category = request.args.get('price_category')
    online_delivery = request.args.get('online_delivery')
    min_review = request.args.get('min_review')
    if loclat != None and loclng != None:
        <fix/>try:
            lat_range = [float(loclat) - geodist, float(loclat) + geodist]
            lng_range = [float(loclng) - geodist, float(loclng) + geodist]
        except:
            logger.error("Error translating location to floats in "
                     "query_restaurants_by_taste, passed values: "
                     "%s, %s" % (loclat, loclng))
            return None</fix>
    else:
        lat_range = None
        lng_range = None
    filtered_query = database.restaurant_query_builder(restaurant_query,
                                                       lat_range, lng_range,
                                                       price_category,
                                                       min_review, online_delivery)
    if filtered_query == -1:
        logger.error("Restaurant query builder failed to process inputs for "
                     "query_restaurants_by_taste: %s, %s, %s" %
                     (price_category,
                     min, online_delivery))
    limited_query = database.order_by_and_limit_query(filtered_query,
                                                    "agg_review DESC", 20)
    query_res = database.run_sql_query(limited_query)
    if query_res == -1:
        return None
    return query_res


def get_taste_condition(value):
    if value == 1:
        return "0.6 AND 1"
    else:
        return "0.0 AND 0.4"


@app.route('/unique_ingredients/<cuisine_id>')
def find_unique_ingredients_from_cuisine(cuisine_id):
    logger.info("GET find_unique_ingredients_from_cuisine query")
    if cuisine_id in unique_ingredients_cache:
        insert_time, data = unique_ingredients_cache[cuisine_id]
        if datetime.now() < insert_time + cache_persistence_time:
            return data

    try:
        cuisine_id_int = int(cuisine_id)
    except:
        logger.error("Error translating cuisine_id to int in "
                     "find_unique_ingredients_from_cuisine, passed value: "
                     "%s" % cuisine_id)
        return None

    query_res = database.find_unique_ingredients_of_cuisine(cuisine_id_int, 500)
    if query_res == -1:
        return None
    if len(simplejson.loads(query_res)) == 0:  # try again with smaller filter
        query_res = database.find_unique_ingredients_of_cuisine(cuisine_id_int,
                                                                250)
        if query_res == -1:
            return None
        unique_ingredients_cache[cuisine_id] = (datetime.now(), query_res)
        return query_res
    else:
        unique_ingredients_cache[cuisine_id] = (datetime.now(), query_res)
        return query_res


@app.route('/new_franchise/<lat>/<lng>')
def set_up_new_franchise(lat, lng):
    try:
        lat, lng = float(lat), float(lng)
    except:
        logger.error("Error translating location to floats in "
                     "set_up_new_franchise, passed values: "
                     "lat: %s, lng: %s" % (lat, lng))

    query_res = database.set_up_new_franchise(lat, lng, 0.015)
    if query_res == -1:
        return None
    return query_res


@app.route('/get_common_ingredients_with/<ingredient>')
def get_common_ingredients_with(ingredient):
    result = database.query_common_ingredients_with(ingredient)
    if result == -1:
        return None
    else:
        return result


if __name__ == '__main__':
    app.run(port=port_number)