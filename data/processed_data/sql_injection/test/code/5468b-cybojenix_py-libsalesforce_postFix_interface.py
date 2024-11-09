<fix/>from typing import Any, Dict, Iterator, TypeVar, Optional</fix>

from typing_extensions import Protocol

T = TypeVar("T")


class SupportsSubQuery(Protocol):
    def __iter__(self) -> "Iterator[IRow]":
        ...


class SupportsRendering(Protocol):
    def render(self) -> str:
        ...


class SupportsFiltering(Protocol):
    <fix/>def __call__(self: T, *, where: Optional[SupportsRendering] = None) -> T:</fix>
        ...


class IRow(SupportsFiltering, SupportsSubQuery, Protocol):
    def __getattr__(self, name: str) -> Any:
        ...


<fix/>class IQueryManager(SupportsFiltering, Protocol):</fix>
    def __iter__(self) -> Iterator[IRow]:
        ...


class ISpy(IRow, Protocol):
    selected_fields: "Dict[str, ISpy]"
    is_subquery: bool
    where: Optional[SupportsRendering]


class IQueryModel(Protocol):
    name: str


class IQueryClient(Protocol):
    def query(self, query_string: str) -> Iterator[IRow]:
        ...

class IFilterBuilder(SupportsRendering, Protocol):
    def __getattr__(self: T, name: str) -> T:
        pass
