from cellpose import io, models, train
import os

io.logger_setup()

train_dir = '/mnt/work/data/CM_Seg_Paper_24/cellpose/train'
val_dir = '/mnt/work/data/CM_Seg_Paper_24/cellpose/val'

train_ortho = os.path.join(train_dir, 'ortho')
val_ortho = os.path.join(val_dir, 'ortho')

out = io.load_train_test_data(train_ortho, val_ortho, image_filter="",
                              mask_filter="_masks", look_one_level_down=False)

images, labels, image_names, test_images, test_labels, image_names_test = out

model = models.CellposeModel(pretrained_model=None, gpu=True)

model_path, train_losses, test_losses = train.train_seg(model.net,
                                                        train_data=images, train_labels=labels,
                                                        channels=[0, 0],
                                                        n_epochs=500,
                                                        learning_rate=0.2,
                                                        SGD=True,
                                                        weight_decay=1e-5,
                                                        momentum=0.9,
                                                        test_data=test_images,
                                                        test_labels=test_labels,
                                                        model_name="Ortho_WGA_CM")
