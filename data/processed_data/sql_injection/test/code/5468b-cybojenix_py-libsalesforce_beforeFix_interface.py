<vul/>from typing import Any, Dict, Iterator, TypeVar</vul>

from typing_extensions import Protocol

T = TypeVar("T")


class SupportsSubQuery(Protocol):
    def __iter__(self) -> "Iterator[IRow]":
        ...


class SupportsFiltering(Protocol):
    <vul/>def filter(self, expression: bool) -> bool:</vul>
        ...


class IRow(SupportsFiltering, SupportsSubQuery, Protocol):
    def __getattr__(self, name: str) -> Any:
        ...


<vul/>class IQueryManager(Protocol):</vul>
    def __iter__(self) -> Iterator[IRow]:
        ...


class ISpy(IRow, Protocol):
    selected_fields: "Dict[str, ISpy]"
    is_subquery: bool


class IQueryModel(Protocol):
    name: str


class IQueryClient(Protocol):
    def query(self, query_string: str) -> Iterator[IRow]:
        ...
