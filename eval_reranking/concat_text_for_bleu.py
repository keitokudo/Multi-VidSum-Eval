import sys

n = int(sys.argv[2])
with open(f'{sys.argv[1]}', 'r') as f, open(f'{sys.argv[1]}.concat', 'w') as of:
    texts = []
    for line in f:
        line = line.rstrip()
        texts.append(line)
        if len(texts) % n == 0: 
            of.write(' '.join(texts) + '\n')
            texts = []


