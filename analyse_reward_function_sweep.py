import json
import numpy as np


fn = 'reward_functino_parameter_sweep_.json'
f = file(fn, 'r')
d = json.load(f)

for v_ in d.values():
    print v_


