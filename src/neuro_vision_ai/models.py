import tensorflow as tf
from pathlib import Path

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

    model = tf.keras.Model(inputs=inputs,outputs=outputs,name='custom_cnn')

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

    model = tf.keras.Model(inputs=inputs,outputs=outputs, name='mobilenetv2_frozen')

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

def callbacks(best_model_path,backup_dir,monitor='val_loss',patience=3,restore_best_weights=True):
    best_model_path = Path(best_model_path)
    backup_dir = Path(backup_dir)
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(monitor=monitor,patience=patience,restore_best_weights=restore_best_weights),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(best_model_path),monitor=monitor,save_best_only=True),
        tf.keras.callbacks.BackupAndRestore(backup_dir=str(backup_dir))
    ]
