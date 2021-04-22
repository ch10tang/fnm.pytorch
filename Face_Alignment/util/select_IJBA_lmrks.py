import os

def select_IJBA_lmrks(index, split, OTS_lmrks):

    for list_name in index:
        tmp = split.FILE[list_name].split('/')
        if tmp[0] == 'frame':
            continue
        else:
            for list_lmrk in os.listdir(OTS_lmrks):
                if list_lmrk.startswith(tmp[1].split('.')[0]):
                    return list_lmrk

    return []
