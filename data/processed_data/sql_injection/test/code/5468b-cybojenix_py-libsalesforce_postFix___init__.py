<fix/>from .api import get_query_manager, get_filter_builder</fix>

<fix/>__all__ = ("get_query_manager", "get_filter_builder")</fix>


"""
<fix/>from libsalesforce.client import Client
from libsalesforce.model import NamedModel
from libsalesforce.query import *
client = Client()
Opportunity = NamedModel('Opportunity')
qm = get_query_manager(Opportunity, client)
O = get_filter_builder()
A = get_filter_builder()</fix>

<fix/>opportunities = qm.run(</fix>
    {
        'id': o.Id,
        'accounts': [
            {'id': a.Id}
            <fix/>for a in o.Accounts(where=
                (A.Id == "dfvfdbvdfv")
            )</fix>
        ]
    }
    <fix/>for o in qm(where=
        (O.Id == "hio")
    )</fix>
)
"""
