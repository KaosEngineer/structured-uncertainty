import sys
import numpy as np
for line in sys.stdin:
    line = line[:-3].split(' ')
    line = ' '.join(np.random.permutation(line))
    sys.stdout.write(line+' .\n')