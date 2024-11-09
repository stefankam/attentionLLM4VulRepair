<fix/>from typing import Iterator, TypeVar, Optional</fix>

<fix/>from .interface import ISpy, SupportsRendering</fix>

T = TypeVar("T")


<fix/>def construct_select_statement(spy: ISpy, from_: str, *, where: Optional[SupportsRendering] = None) -> str:
    clause = f"""SELECT {', '.join(construct_selects(spy))} FROM {from_}"""
    if where:
        clause += f' WHERE {where.render()}'
    return clause</fix>


def construct_selects(spy: ISpy, current_name: str = "") -> Iterator[str]:
    if spy.is_subquery:
        yield construct_subquery(spy, name=current_name)
    elif not spy.selected_fields:
        yield current_name
    else:
        for field_name, field_spy in spy.selected_fields.items():
            joined_name = f"{current_name}.{field_name}".lstrip(".")
            yield from construct_selects(field_spy, joined_name)


def construct_subquery(spy: ISpy, name: str) -> str:
    select_fields = _flatten(
        construct_selects(field_spy, field_name)
        for field_name, field_spy in spy.selected_fields.items()
    )
    <fix/>inner_clause = f"""SELECT {', '.join(select_fields)} FROM {name}"""
    if spy.where:
        inner_clause += f" WHERE {spy.where.render()}"
    return f"({inner_clause})"</fix>


def _flatten(iterables: Iterator[Iterator[T]]) -> Iterator[T]:
    for iterable in iterables:
        yield from iterable
