def read_lines_to_list(filename):
    with open(filename, encoding='utf-8') as f:
        return [line for line in f]
