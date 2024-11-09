<fix/>from typing import Iterator, TypeVar, Optional</fix>

from .construct import construct_select_statement
<fix/>from .interface import IRow, SupportsRendering</fix>
from .spy import Spy

T = TypeVar("T")
T_QM = TypeVar("T_QM", bound='QueryManager')


class QueryManager:
    where: Optional[SupportsRendering]

    def __init__(self, from_object: str, client):
        self.from_object = from_object
        self.client = client
        self.where = None

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

    def __call__(self: T_QM, *, where: SupportsRendering = None) -> T_QM:
        self.where = where
        return self

    def __iter__(self) -> Iterator[IRow]:
        # We need to find out what fields are being requested
        spy = Spy()
        yield spy

        # Now we've collected all access points, turn it into an SOQL statement
        <fix/>query_string = construct_select_statement(spy, self.from_object, where=self.where)</fix>
        print(query_string)

        # The client can take over from here, as that is in charge of fetching + building objects.
        yield from self.client.query(query_string)
