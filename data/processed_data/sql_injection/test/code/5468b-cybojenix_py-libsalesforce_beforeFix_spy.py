from collections import defaultdict
<vul/>from typing import Dict, Iterator, TypeVar</vul>

<vul/>from .interface import ISpy</vul>

T = TypeVar("T", bound=ISpy)


class Spy:
    selected_fields: Dict[str, ISpy]
    is_subquery: bool

    def __init__(self):
        <vul/>self.selected_fields = defaultdict(self.__class__)</vul>
        self.is_subquery = False

    def __getattr__(self, name: str) -> ISpy:
        return self.selected_fields[name]

    def __iter__(self: T) -> Iterator[T]:
        self.is_subquery = True
        yield self
