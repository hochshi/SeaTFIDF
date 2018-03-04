import numpy as np
from np_perfect_hash import generate_hash, Hash1
import time

# month = dict(zip('jan feb mar apr may jun jul aug sep oct mov dec'.split(),
#                  range(1, 13)))

# month = dict(zip('jan feb mar apr may'.split(),
#                  range(1, 6)))

dict_len = 300000
month = dict(zip(np.random.rand(dict_len), xrange(dict_len)))

start = time.time()
f1, f2, G = generate_hash(month, Hash1)
end = time.time()

keys = []
hashvals = []
for k, h in month.items():
    keys.append(k)
    hashvals.append(h)

assert np.all(hashvals == ( G[f1(keys)] + G[f2(keys)] ) % len(G))

print 'OK'
print len(G)
print(end - start)