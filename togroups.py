from datetime import datetime as dd

def get_t(s):
    return dd.strptime(s.strip('"'), "%Y-%m-%d %H:%M")

inp = open('statistics.csv', 'r').read().strip().split('\n')

conv = [0 for i in range(600 * 60 * 60)]

groups = [('<1min', (0, 60)), ('<1hour', (60, 60 * 60)), ('<2hours', (60 * 60, 2 * 60 * 60)),\
        ('<6hours', (2 * 60 * 60, 6 * 60 * 60)), ('<1day', (6 * 60 * 60, 24 * 60 * 60)),\
        ('<3days', (24 * 60 * 60, 3 * 24 * 60 * 60)), ('>3days', (3 * 24 * 60 * 60, 600 * 60 * 60))]

files = []
for g in groups:
    files.append(open(g[0], 'w'))
    files[-1].write(inp[0])
    for i in range(g[1][0], g[1][1]):
        conv[i] = len(files) - 1

for x in inp[1:]:
    row = x.split('\t')

    a = get_t(row[2])
    b = get_t(row[3])

    tm = int((b - a).total_seconds())

    files[conv[tm]].write(x + '\n')
