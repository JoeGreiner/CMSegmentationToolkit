import os
from datetime import datetime
import matplotlib.pyplot as plt
from csbdeep.utils import axes_dict, plot_history
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
from csbdeep.data import RawData, create_patches
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

patch_size = (32,128,128)
n_patches_per_image = 64

raw_data = RawData.from_folder (
    basepath    = 'data/Confocal_Pinhole/train/',
    source_dirs = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'v2_p06', 'v2_p10', 'v2_p20',
                   'v2_p30', 'v2_p40', 'v2_p50', 'v2_p60'],
    target_dir  = 'clean',
    axes        = 'ZYX',
)

X, Y, XY_axes = create_patches (
    raw_data            = raw_data,
    patch_size          = patch_size,
    n_patches_per_image = n_patches_per_image,
    save_file           ='data/my_training_data.npz',
)
assert X.shape == Y.shape

# defaults
# self.train_epochs = 100
# self.train_steps_per_epoch = 400
# self.train_batch_size = 16
# self.unet_n_depth = 2

number_epochs = 100
train_steps_per_epoch = 400
batch_size = 8
unet_depth = 2

plot_folder = 'results'
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

(X,Y), (X_val,Y_val), axes = load_training_data('data/my_training_data.npz', validation_split=0.1, verbose=True)

c = axes_dict(axes)['C']
n_channel_in, n_channel_out = X.shape[c], Y.shape[c]
config = Config(axes, n_channel_in, n_channel_out, train_steps_per_epoch=train_steps_per_epoch, train_epochs=number_epochs, unet_n_depth=unet_depth, train_batch_size=batch_size)
model = CARE(config, 'confocal_pinhole_v3_big_patch', basedir='models')
history = model.train(X,Y, validation_data=(X_val,Y_val))

plt.figure(figsize=(16,5))
plot_history(history,['loss','val_loss'],['mse','val_mse','mae','val_mae'])
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename_detailed = f'training_history_{timestamp}_{number_epochs}epochs_{train_steps_per_epoch}steps_{batch_size}batchsize_{unet_depth}depth.png'
output_path_history = os.path.join(plot_folder, f'{filename_detailed}.png')
plt.savefig(output_path_history, dpi=300, bbox_inches='tight')
plt.close()

model.export_TF()