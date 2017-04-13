data = []
row = [0] * 4
event = [0, 3, 3, 2]
for i in event:
    row[i] = 1
    data.append(row)
print data

idx = 0
data = []
for i in range(10):
    idx = idx + 1
    data.append(idx)
print data
