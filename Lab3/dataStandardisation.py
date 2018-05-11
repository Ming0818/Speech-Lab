import numpy as np
from sklearn.preprocessing import StandardScaler

train_data  = np.load('train.npz')['traindata']
val_data  = np.load('val.npz')['valdata']
test_data  = np.load('test.npz')['testdata']

lmfcc_train_x = train_data['lmfcc']
mspec_train_x = train_data['mspec']
train_y = train_data['targets']

lmfcc_val_x = val_data['lmfcc']
mspec_val_x = val_data['mspec']
val_y = val_data['targets']

lmfcc_test_x = test_data['lmfcc']
mspec_test_x = test_data['mspec']
test_y = test_data['targets']

scaler = StandardScaler()
scaler.fit(lmfcc_train_x)
scaler.transform(lmfcc_train_x)
scaler.transform(lmfcc_val_x)
scaler.transform(lmfcc_test_x)

scaler2 = StandardScaler()
scaler2.fit(mspec_train_x)
scaler2.transform(mspec_train_x)
scaler2.transform(mspec_val_x)
scaler2.transform(mspec_test_x)