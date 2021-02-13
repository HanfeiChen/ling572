import sys
correct = 0
total = 0
with open(sys.argv[1], 'r') as f:
    for line in f:
        if len(line.strip()) > 0 and not line.startswith('%'):
            line = line.strip()
            name, gold, pred, prob = line.split()
            total += 1
            if gold == pred:
                correct += 1
print(correct / total)
