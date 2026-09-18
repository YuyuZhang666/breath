from collections.abc import Iterable


def has_item(items: Iterable[str], name: str) -> bool:
    expected = name.casefold()
    return any(item.casefold() == expected for item in items)


def count_item(items: Iterable[str], name: str) -> int:
    expected = name.casefold()
    return sum(item.casefold() == expected for item in items)
