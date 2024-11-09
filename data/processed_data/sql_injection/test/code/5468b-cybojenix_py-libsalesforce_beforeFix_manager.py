<vul/>from typing import Iterator, TypeVar</vul>

from .construct import construct_select_statement
<vul/>from .interface import IRow</vul>
from .spy import Spy

T = TypeVar("T")


class QueryManager:
    def __init__(self, from_object: str, client):
        self.from_object = from_object
        self.client = client

    def run(self, iterator: Iterator[T]) -> Iterator[T]:
        # Currently we don't actually do much here.
        # We need to drop the first element of the iterator, as this is
        # where we used a Spy.
        # While through magical powers at a future time, we could make the Spy
        # actually change its stripes into actual data, it's more effort than it's
        # worth.

        # As we could have a list or generator, we should change to an iterator to have
        # a common interface to work with.
        iterator = iter(iterator)
        next(iterator)  # discards
        return iterator

    def __iter__(self) -> Iterator[IRow]:
        # We need to find out what fields are being requested
        spy = Spy()
        yield spy

        # Now we've collected all access points, turn it into an SOQL statement
        <vul/>query_string = construct_select_statement(spy, self.from_object)</vul>
        print(query_string)

        # The client can take over from here, as that is in charge of fetching + building objects.
        yield from self.client.query(query_string)
