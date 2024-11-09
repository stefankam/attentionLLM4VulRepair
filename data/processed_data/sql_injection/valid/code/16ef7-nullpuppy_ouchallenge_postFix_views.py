from itemPrices.models import ItemSale
from django.db import connection
from django.db.models import Count
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status


NOT_FOUND_JSON_RESPONSE = {
    'status': 404,
    'content': {
        'message': 'Not found',
    }
}


class ItemPriceService(APIView):
    """
    """
    def get(self, request):
        item = request.query_params.get('item')
        city = request.query_params.get('city')

        # If item and city are both excluded from the request, return
        # a json blob with a status of 404.
        if not item and not city:
            return Response(NOT_FOUND_JSON_RESPONSE)

        # Raw SQL using built-in mode() function within postgres
        # Returns a single row with the highest most frequent list price and
        # count of items found for the given search parameters.
        <fix/>#
        # -- This code is unsafe and is vulnerable to sql injection. incoming
        #    parameters need to be escaped in some manner to be safe to query on.
        #
        # sql = '''SELECT
        #             mode() WITHIN GROUP (ORDER BY list_price DESC) AS model_value,
        #             count(*)
        #          FROM
        #             "itemPrices_itemsale"
        #       '''
        # if item and city:
        #     sql = "{} WHERE city = '{}' and title = '{}'".format(sql, city, item)
        # elif item:
        #     sql = "{} WHERE title = '{}'".format(sql, item)
        # elif city:
        #     sql = "{} WHERE city = '{}'".format(sql, city)</fix>

        <fix/># with connection.cursor() as c:
        #     c.execute(sql)
        #     price_mode, count = c.fetchone()</fix>

        # More traditional django ORM route of doing the above.
        # The above seems to be slightly faster, based on the
        # throughput I observed in jmeter, but is database specific.
        # Adding caching and reworking the ORM query might be a better
        # choice moving forward.

        query = ItemSale.objects
        if item:
            query = query.filter(title__startswith=item)
        if city:
            query = query.filter(city=city)

        # # Get total item count for given parameters.
        count = query.count()

        # Find list_price mode for given parameters.
        query = query.order_by('list_price').values('list_price').annotate(price_count=Count('list_price'))
        price_mode = query.order_by('-price_count', '-list_price').first()
        if price_mode:
            price_mode = price_mode.get('list_price')

        # If we didn't find anything, return 404 response, just as if item and
        # city weren't passed in.
        if count == 0:
            return Response(NOT_FOUND_JSON_RESPONSE)

        return Response({
            'status': 200,
            'content': {
                'item': item or 'Not specified',
                'item_count': count,
                'price_suggestion': price_mode,
                'city': city or 'Not specified',
            }
        })

