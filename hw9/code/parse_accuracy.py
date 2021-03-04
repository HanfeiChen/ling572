import sys

with open(sys.argv[1], 'r') as f:
    acc = 0.
    for line in f:
        if len(line.strip()) > 0:
            head, count = line.strip().split(':')
            acc = eval(count)
    print(acc)
