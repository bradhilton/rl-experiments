import black


def black_print(object: object) -> None:
    print(black.format_str(str(object), mode=black.Mode()))
