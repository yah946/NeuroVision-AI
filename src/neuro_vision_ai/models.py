import tensorflow as tf
import Path

def custum_cnn_model(n_class,augmentation,img_size=(224,224)):
    inputs = tf.keras.Input(shape=(*img_size,3))

    # x = layers.RandomRotation(0.03)(inputs)
    # x = layers.RandomTranslation(0.03, 0.03)(x)
    # x = layers.Rescaling(1.0 / 255)(x)
    
    x = augmentation(inputs)

    x = tf.keras.layers.Conv2D(32,3,activation='relu',padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)

    x = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)

    x = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)

    x = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128,activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(n_class,activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs,outputs,name='custom_cnn')

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    return model, None

def mobile_net_v2_frozen_model(n_class,augmentation,img_size=(224,224)):
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3),include_top=False,weights='imagenet')
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=(*img_size,3))

    # x = layers.RandomRotation(0.03)(inputs)
    # x = layers.RandomTranslation(0.03, 0.03)(x)
    # x = layers.Rescaling(1.0 / 255)(x)
    x = augmentation(inputs)

    x = base_model(x,training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(n_class,activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs,outputs, name='mobilenetv2_frozen')

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    return model, base_model

def mobile_net_v2_finetuning_model(model,base_model,unfreeze_last=30):
    base_model.trainable = True

    for layer in base_model.layers[:-unfreeze_last]:
        layer.trainable = False

    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    return model

def callabacks(best_model_path,backup_dir,monitor='val_loss',patience=3,restore_best_weights=True):
    best_model_path = Path(best_model_path)
    backup_dir = Path(backup_dir)
    best_model_path.parents.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(monitor=monitor,patience=patience,restore_best_weights=restore_best_weights),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(best_model_path),monitor=monitor,save_best_only=True),
        tf.keras.callbacks.BackupAndRestore(backup_dir=str(backup_dir))
    ]



# def train_experiment(experiment_name):
#     train_ds, val_ds, test_ds, num_classes, class_weights = (
#         prepare_datasets()
#     )

#     if experiment_name == "custom_cnn":
#         model = build_custom_cnn(num_classes=num_classes)
#         base_model = None

#     elif experiment_name in ("mobilenet_frozen", "mobilenet_finetuned"):
#         model, base_model = build_mobilenet_frozen(
#             num_classes=num_classes
#         )

#     else:
#         raise ValueError(f"Unknown experiment: {experiment_name}")

#     output_dir = Path("artifacts") / experiment_name
#     output_dir.mkdir(parents=True, exist_ok=True)

#     callbacks = [
#         tf.keras.callbacks.EarlyStopping(
#             monitor="val_loss",
#             patience=4,
#             restore_best_weights=True
#         ),
#         tf.keras.callbacks.ModelCheckpoint(
#             filepath=str(output_dir / "best_model.keras"),
#             monitor="val_loss",
#             save_best_only=True
#         ),
#         tf.keras.callbacks.ReduceLROnPlateau(
#             monitor="val_loss",
#             factor=0.2,
#             patience=2,
#             min_lr=1e-7
#         )
#     ]

#     # Stage 1: train the custom CNN or the frozen MobileNetV2 head.
#     model.fit(
#         train_ds,
#         validation_data=val_ds,
#         epochs=EPOCHS,
#         class_weight=class_weights,
#         callbacks=callbacks
#     )

#     # Stage 2: fine-tune selected MobileNetV2 layers.
#     if experiment_name == "mobilenet_finetuned":
#         model = prepare_fine_tuning(model, base_model)

#         model.fit(
#             train_ds,
#             validation_data=val_ds,
#             epochs=10,
#             class_weight=class_weights,
#             callbacks=callbacks
#         )

#     test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)

#     print({
#         "experiment": experiment_name,
#         "test_loss": test_loss,
#         "test_accuracy": test_accuracy
#     })

#     model.save(output_dir / "final_model.keras")

#     return model