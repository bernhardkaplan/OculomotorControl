import os
import sys
import numpy as np
import json


test_folder_name_base = 'Test_15__0-4_'


folders = []

for thing in os.listdir('.'):
    if thing.find(test_folder_name_base) != -1:
        folders.append(thing)

fn_out = 'folder_names.json'
f = file(fn_out, 'w')
json.dump(folders, f, indent=2)
print 'fn_out:', fn_out
