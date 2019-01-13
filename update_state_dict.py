import torch


def update(old_key, new_key, state_dict):
    if old_key in state_dict:
        state_dict[new_key] = state_dict[old_key]
        state_dict.pop(old_key)


ckpt = '/srv/glusterfs/xieya/seg_2/model_11_0.pth.tar'
weights = torch.load(ckpt)
print('Weights loaded.')
state_dict = weights['state_dict']

if 'dense_film0.weight' in state_dict:
    state_dict['film0.dense.weight'] = state_dict['dense_film0.weight']
    state_dict.pop('dense_film0.weight')
if 'dense_film0.bias' in state_dict:
    state_dict['film0.dense.bias'] = state_dict['dense_film0.bias']
    state_dict.pop('dense_film0.bias')
if 'dense_film1.weight' in state_dict:
    state_dict['film1.dense.weight'] = state_dict['dense_film1.weight']
    state_dict.pop('dense_film1.weight')
if 'dense_film1.bias' in state_dict:
    state_dict['film1.dense.bias'] = state_dict['dense_film1.bias']
    state_dict.pop('dense_film1.bias')
if 'dense_film2.weight' in state_dict:
    state_dict['film2.dense.weight'] = state_dict['dense_film2.weight']
    state_dict.pop('dense_film2.weight')
if 'dense_film2.bias' in state_dict:
    state_dict['film2.dense.bias'] = state_dict['dense_film2.bias']
    state_dict.pop('dense_film2.bias')
if 'dense_film3.weight' in state_dict:
    state_dict['film3.dense.weight'] = state_dict['dense_film3.weight']
    state_dict.pop('dense_film3.weight')
if 'dense_film3.bias' in state_dict:
    state_dict['film3.dense.bias'] = state_dict['dense_film3.bias']
    state_dict.pop('dense_film3.bias')

update("deconv_5.0.weight", "segnet.deconv_5.0.weight", state_dict)
update("deconv_5.0.bias", "segnet.deconv_5.0.bias", state_dict)
update("deconv_5.1.weight", "segnet.deconv_5.1.weight", state_dict)
update("deconv_5.1.weight", "segnet.deconv_5.1.weight", state_dict)
update("deconv_5.1.bias", "segnet.deconv_5.1.bias", state_dict)
update("deconv_5.1.running_mean", "segnet.deconv_5.1.running_mean", state_dict)
update("deconv_5.1.running_var", "segnet.deconv_5.1.running_var", state_dict)
update("deconv_6.0.weight", "segnet.deconv_6.0.weight", state_dict)
update("deconv_6.0.bias", "segnet.deconv_6.0.bias", state_dict)
update("deconv_6.1.weight", "segnet.deconv_6.1.weight", state_dict)
update("deconv_6.1.bias", "segnet.deconv_6.1.bias", state_dict)
update("deconv_6.1.running_mean", "segnet.deconv_6.1.running_mean", state_dict)
update("deconv_6.1.running_var", "segnet.deconv_6.1.running_var", state_dict)
update("deconv_7.0.weight", "segnet.deconv_7.0.weight", state_dict)
update("deconv_7.0.bias", "segnet.deconv_7.0.bias", state_dict)
update("deconv_7.1.weight", "segnet.deconv_7.1.weight", state_dict)
update("deconv_7.1.bias", "segnet.deconv_7.1.bias", state_dict)
update("deconv_7.1.running_mean", "segnet.deconv_7.1.running_mean", state_dict)
update("deconv_7.1.running_var", "segnet.deconv_7.1.running_var", state_dict)
update("seg_conv.0.weight", "segnet.seg_conv.0.weight", state_dict)
update("seg_conv.0.bias", "segnet.seg_conv.0.bias", state_dict)
update("seg_conv.1.weight", "segnet.seg_conv.1.weight", state_dict)
update("seg_conv.1.bias", "segnet.seg_conv.1.bias", state_dict)
update("seg_conv.1.running_mean", "segnet.seg_conv.1.running_mean", state_dict)
update("seg_conv.1.running_var", "segnet.seg_conv.1.running_var", state_dict)
update("seg_classifier.weight", "segnet.seg_classifier.weight", state_dict)
update("seg_classifier.bias", "segnet.seg_classifier.bias", state_dict)
update('deconv_5.1.num_batches_tracked', 'segnet.deconv_5.1.num_batches_tracked', state_dict)
update('deconv_6.1.num_batches_tracked', 'segnet.deconv_6.1.num_batches_tracked', state_dict)
update('deconv_7.1.num_batches_tracked', 'segnet.deconv_7.1.num_batches_tracked', state_dict)
update('seg_conv.1.num_batches_tracked', 'segnet.seg_conv.1.num_batches_tracked', state_dict)

torch.save(weights, ckpt + '.new')
