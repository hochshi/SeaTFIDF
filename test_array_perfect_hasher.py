import numpy as np
from CMerModel import ArrayPerfectHasher
import time

imap = ArrayPerfectHasher()
imap.reset(2)
pool = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=100000)
sizes = np.random.randint(20, 500, 120000)


start = time.time()
for idx, size in enumerate(sizes):
    if 0 == idx % 1000:
        print "%d done out of %d" % (idx, sizes.size)
    imap.hashArray(np.random.choice(pool, size=size))
end = time.time()
print(end - start)

start = time.time()
imap.done()
end = time.time()
print(end - start)