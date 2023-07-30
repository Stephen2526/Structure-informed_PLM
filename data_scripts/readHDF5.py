import h5py

filename = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/checkpoints/CNN-VAE-test-train/final_model.h5'
fl = h5py.File(filename, 'r')

for key in fl.keys():
  print(key)

print(fl.get('training_config'))
'''
training_config = fl.get('training_config')
print('training_config: ', training_config)

opt_wei = fl.get('optimizer_weights')
print('optimizer_weights: ', opt_wei)

opt = fl['optimizer_weights']
ml = fl['model_weights']

for key in opt.keys():
  print('opt: ', key)

for key in ml.keys():
  print('ml: ', key)

tr = opt['training']
adam = opt['Adam']
for key in tr.keys():
  print('tr: ', key)

for key in adam.keys():
  print('adam: ', adam)

tr_adam = tr['Adam']
#for key in tr_adam.keys():
  #print('tr_adam: ', key)
'''
