import os
from pathlib import Path
import shutil
import itertools

gfilesPath = 'c:/users/eric/downloads/sigcomp2011-trainingset/trainingset/offlinesignatures/chinese/trainingset/offline genuine'
ffilesPath = 'c:/users/eric/downloads/sigcomp2011-trainingset/trainingset/offlinesignatures/chinese/trainingset/offline forgeries'
gfiles = os.listdir(gfilesPath)
ffiles = os.listdir(ffilesPath)

dataPath = 'c:/users/eric/workspace/DL_data/signatures'
gdirs = [list(v) for k,v in itertools.groupby(gfiles, lambda x:x[:3])]
for dir in gdirs:
    path = dataPath+'/'+dir[0][:3]
    if not os.path.exists(path):
        os.makedirs(path)
    for item in dir:
        file = gfilesPath+'/'+item
        shutil.copyfile(file,path+'/'+item)

fdirs = [list(v) for k,v in itertools.groupby(ffiles, lambda x:x[4:7])]
for dir in fdirs:
    path = dataPath+'/'+dir[0][4:7]
    if not os.path.exists(path):
        os.makedirs(path)
    for item in dir:
        file = ffilesPath+'/'+item
        shutil.copyfile(file,path+'/'+item)
# for dir in gdirs:
