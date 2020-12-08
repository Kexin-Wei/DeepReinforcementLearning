from collections import deque

l = deque(maxlen = 3)
l.append(1)
l.append(2)
l.append(3)
print(l)
l.append(4)
print(l)