from collections import defaultdict
<fix/>from typing import Dict, Iterator, TypeVar, Any, Union, Optional
from typing_extensions import Literal</fix>

<fix/>from .interface import ISpy, SupportsRendering</fix>

T = TypeVar("T", bound=ISpy)


class Spy:
    selected_fields: Dict[str, ISpy]
    is_subquery: bool
    where: Optional[SupportsRendering]

    def __init__(self):
        <fix/>self.selected_fields = defaultdict(Spy)</fix>
        self.is_subquery = False
        self.where = None

    def __call__(self: T, *, where: Optional[SupportsRendering] = None) -> T:
        self.where = where
        return self

    def __getattr__(self, name: str) -> ISpy:
        return self.selected_fields[name]

    def __iter__(self: T) -> Iterator[T]:
        self.is_subquery = True
        yield self
