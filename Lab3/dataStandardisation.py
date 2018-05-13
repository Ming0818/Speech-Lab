import numpy as np
from sklearn.preprocessing import StandardScaler

train_data  = np.load('dy_final/traindata_dy.npz')
val_data  = np.load('dy_final/valdata_dy.npz')
test_data  = np.load('dy_final/testdata_dy.npz')

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
# np.savez('lmf_train.npz', lmfcc=lmfcc_train_x,targets = train_y)
# np.savez('lmf_val.npz', lmfcc=lmfcc_val_x,targets = val_y)
# np.savez('lmf_test.npz', lmfcc=lmfcc_test_x,targets = test_y)


scaler2 = StandardScaler()
scaler2.fit(mspec_train_x)
scaler2.transform(mspec_train_x)
scaler2.transform(mspec_val_x)
scaler2.transform(mspec_test_x)
# np.savez('msp_train.npz', mspec=mspec_train_x)
# np.savez('msp_val.npz', mspec=mspec_val_x)
# np.savez('msp_test.npz', mspec=mspec_test_x)

# train_data  = np.load('lmf_train.npz')
# val_data  = np.load('lmf_val.npz')
# test_data  = np.load('lmf_test.npz')
#
# lmfcc_train_x = train_data['lmfcc']
# train_y = train_data['targets']
#
# lmfcc_val_x = val_data['lmfcc']
# val_y = val_data['targets']
#
# lmfcc_test_x = test_data['lmfcc']
# test_y = test_data['targets']
#
# mspec_train_x  = np.load('msp_train.npz')['mspec']
# mspec_val_x  = np.load('msp_val.npz')['mspec']
# mspec_test_x  = np.load('msp_test.npz')['mspec']


np.savez('traindata_stan_dy.npz', lmfcc=lmfcc_train_x,mspec = mspec_train_x,targets = train_y)
np.savez('valdata_stan_dy.npz', lmfcc=lmfcc_val_x,mspec = mspec_val_x,targets = val_y)
np.savez('testdata_stan_dy.npz', lmfcc=lmfcc_test_x,mspec = mspec_test_x,targets = test_y)

# test = np.load('data_stan_dy/testdata_stan_dy.npz')
# lmfcc = test['lmfcc']
# mspec = test['mspec']
# targets = test['targets']
# np.savez('testdata_stan_dy_lmfcc',lmfcc = lmfcc)
# np.savez('testdata_stan_dy_mspec',mspec =mspec)
# np.savez('testdata_stan_dy_target',targets =targets)

# test = np.load('data_dy/testdata_dy.npz')
# lmfcc = test['lmfcc']
# mspec = test['mspec']
# targets = test['targets']
# np.savez('testdata_dy_lmfcc',lmfcc = lmfcc)
# np.savez('testdata_dy_mspec',mspec =mspec)
# np.savez('testdata_dy_target',targets =targets)
