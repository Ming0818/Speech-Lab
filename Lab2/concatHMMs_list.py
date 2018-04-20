from proto2 import *
def concatHMMs_list(phoneHMMs, modellist):
    wordHMMs_list = {}
    wordHMMs_list['o'] = concatHMMs(phoneHMMs, modellist['o'])
    wordHMMs_list['z'] = concatHMMs(phoneHMMs, modellist['z'])
    wordHMMs_list['1'] = concatHMMs(phoneHMMs, modellist['1'])
    wordHMMs_list['2'] = concatHMMs(phoneHMMs, modellist['2'])
    wordHMMs_list['3'] = concatHMMs(phoneHMMs, modellist['3'])
    wordHMMs_list['4'] = concatHMMs(phoneHMMs, modellist['4'])
    wordHMMs_list['5'] = concatHMMs(phoneHMMs, modellist['5'])
    wordHMMs_list['6'] = concatHMMs(phoneHMMs, modellist['6'])
    wordHMMs_list['7'] = concatHMMs(phoneHMMs, modellist['7'])
    wordHMMs_list['8'] = concatHMMs(phoneHMMs, modellist['8'])
    wordHMMs_list['9'] = concatHMMs(phoneHMMs, modellist['9'])
    return wordHMMs_list