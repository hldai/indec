import math


def log_factorial(x):
    r = 0
    for i in range(1, x + 1):
        r += math.log(i)
    return r


def read_lines_to_list(filename):
    with open(filename, encoding='utf-8') as f:
        return [line.strip() for line in f]


def write_list_to_lines(str_list, filename):
    with open(filename, 'w', encoding='utf-8', newline='\n') as fout:
        for s in str_list:
            fout.write('{}\n'.format(s))


def remove_lines(src_file, lines, dst_file):
    f = open(src_file, encoding='utf-8')
    fout = open(dst_file, 'w', encoding='utf-8', newline='\n')
    for i, line in enumerate(f):
        if i in lines:
            continue
        fout.write(line)
    f.close()
    fout.close()
