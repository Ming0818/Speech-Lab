import numpy as np

def concat_data(data):
    lmfcc = data[0]['lmfcc']
    mspec = data[0]['mspec']
    targets = np.array(data[0]['targets'])
    lmfcc_data = dynamic_feature(lmfcc)
    mspec_data = dynamic_feature(mspec)
    targets_data = targets.reshape(-1,1)

    for i in range(1, data.shape[0]):
        lmfcc = data[i]['lmfcc']
        mspec = data[i]['mspec']
        targets = np.array(data[i]['targets'])
        lmfcc_temp = dynamic_feature(lmfcc)
        mspec_temp = dynamic_feature(mspec)

        lmfcc_data = np.vstack((lmfcc_data, lmfcc_temp))
        mspec_data = np.vstack((mspec_data, mspec_temp))
        targets_data = np.vstack((targets_data, targets.reshape(-1,1)))

    return lmfcc_data,mspec_data,targets_data


def concat_data_withoutDynamic(data):
    lmfcc = data[0]['lmfcc']
    mspec = data[0]['mspec']
    targets = np.array(data[0]['targets'])
    lmfcc_data = lmfcc
    mspec_data = mspec
    targets_data = targets.reshape(-1,1)

    for i in range(1, data.shape[0]):
        lmfcc = data[i]['lmfcc']
        mspec = data[i]['mspec']
        targets = np.array(data[i]['targets'])

        lmfcc_data = np.vstack((lmfcc_data, lmfcc))
        mspec_data = np.vstack((mspec_data, mspec))
        targets_data = np.vstack((targets_data, targets.reshape(-1,1)))

    return lmfcc_data,mspec_data,targets_data


def dynamic_feature(data):
    temp = np.fliplr(data[:, 1:4])
    temp2 = np.fliplr(data[:, -4:-1])
    data_n = np.hstack((temp, data))
    data_n = np.hstack((data_n, temp2))

    data_new = data_n[:,0:7]
    for i in range(1,data.shape[1]):
        if i == data.shape[1]:
            t = data_n[:,i:-1]
        else:
            t = data_n[:,i:i+7]
        data_new = np.hstack((data_new,t))

    return data_new

val_data = np.load('valdata.npz')['valdata']
lmfcc_data,mspec_data,targets_data = concat_data_withoutDynamic(val_data)
np.savez('valdata_new.npz', lmfcc=lmfcc_data,mspec = mspec_data,targets = targets_data)

train_data = np.load('traindata.npz')['traindata']
lmfcc_data,mspec_data,targets_data = concat_data_withoutDynamic(train_data)
np.savez('traindata_new.npz', lmfcc=lmfcc_data,mspec = mspec_data,targets = targets_data)

test_data = np.load('testdata.npz')['testdata']
lmfcc_data,mspec_data,targets_data = concat_data_withoutDynamic(test_data)
np.savez('testdata_new.npz', lmfcc=lmfcc_data,mspec = mspec_data,targets = targets_data)
lmfcc1_data,mspec1_data,targets1_data = concat_data(test_data)
np.savez('testdata_dy.npz', lmfcc=lmfcc1_data,mspec = mspec1_data,targets = targets1_data)