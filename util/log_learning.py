#!/usr/bin/env python
# encoding: utf-8
from time import strftime, localtime

def log_learning(epoch, steps, modelname, lr, loss, args):
    text = "{0} EPOCH : {1}, step : {2}, {3}_Lr: {4}, {5}, {6}".format(strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()), epoch, steps, modelname, lr, modelname, loss)
    print(text)
    '''
    with open('{}/Learning_Log.txt'.format(args.snapshot_dir),'a') as f:
        f.write("{}\n".format(text))
        '''
