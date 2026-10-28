from pathlib import Path
import tensorflow as tf
import cv2
import os
import sklearn
import numpy as np
IMG_SIZE = (224,224)
BATCH_SIZE = 32
RANDON_SEED = 10
VALID_EXTENSIONS = {".jpeg", ".jpg", ".bmp", ".png"}
DATASET_PATH = Path('data/raw')

def load_data(dataset_path=DATASET_PATH,img_size=IMG_SIZE):
    images,labels = [], []
    invalid_files, corrupted_files = [], []
    for class_dir in sorted(dataset_path.iterdir()):
        if not class_dir.is_dir():continue
        class_name = class_dir.name
        for file_path in class_dir.iterdir():
            if not file_path.is_file():continue
            if file_path.suffix.lower() not in VALID_EXTENSIONS:
                invalid_files.append(str(file_path))
                continue
            try:
                image = cv2.imread(str(file_path),cv2.IMREAD_COLOR)
                if image is None:
                    corrupted_files.append(str(file_path))
                    continue
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                image = cv2.resize(image,img_size,interpolation=cv2.INTER_AREA)
                images.append(image)
                labels.append(class_name) 
            except (cv2.error, OSError, ValueError) as error:
                corrupted_files.append(str(file_path))
                print(f"Image ignorée: {file_path}\nerror:\n{error}")
    X = np.asarray(images, dtype=np.uint8)
    y = np.asarray(labels)
    
    print("#=============================")
    print("# RAPPORT")
    print("#=============================")
    print("Shape X :", X.shape)
    print("Shape y :", y.shape)
    print("Extensions invalides :", len(invalid_files))
    print("Images corrompues :", len(corrupted_files))

    return X,y

# class_counts = pd.Series(y).value_counts().sort_index()
def imgs_per_class(y,figsize=(8,5)):
    class_counts = pd.Series(y).value_counts.sort_index()
    plt.figure(figsize=figsize)
    sns.barplot(
        x=class_counts.index,
        y=class_counts.values
    )

    plt.title("Distribution des images par classe")
    plt.xlabel("Classe")
    plt.ylabel("Nombre d'images")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()

    return class_counts

# classes = sorted(np.unique(y))
def read_samples(X,y,samples_per_class=4):
    classes = sorted(np.unique(y))
    _,axis = plt.subplots(len(classes),samples_per_class,figsize=(3*samples_per_class,3*len(classes)),squeeze=False)
    for row, class_name in enumerate(classes):
        indices = np.where(y == class_name)[0][:samples_per_class]

        for col in range(samples_per_class):
            ax = axes[row, col]

            if col < len(indices):
                ax.imshow(X[indices[col]])
                ax.set_title(class_name)

            ax.axis("off")

    plt.tight_layout()
    plt.show()

def train_test_val_split(X,y,train_size=0.5,val_size=0.25):
    test_size = 1 - train_size - val_size
    if test_size <= 0 or train_size <= 0 or val_size <= 0:
        raise ValueError('train, validation and test sizes must be positive!')
    X_train,X_tmp,y_train,y_tmp = sklearn.model_selection.train_test_split(X,y,train_size=train_size,random_state=RANDON_SEED,stratify=y)
    relative_test_val_size = test_size / (1 - train_size)
    X_val,X_test,y_val,y_test = sklearn.model_selection.train_test_split(X_tmp,y_tmp,test_size=relative_test_val_size,random_state=RANDON_SEED,stratify=y_tmp)
    return X_train,y_train,X_val,y_val,X_test,y_test

def label_encoder(y_train_str,y_val_str,y_test_str):
    encoder = sklearn.preprocessing.LabelEncoder()
    y_train_int = encoder.fit_transform(y_train_str)
    y_val_int = encoder.transform(y_val_str)
    y_test_int = encoder.transform(y_test_str)
    return y_train_int,y_val_int,y_test_int,encoder

def input_normalisation(image,mobilenetv2=False):
    image = tf.cast(image, tf.float32)
    if mobilenetv2:
        return image/127.5 - 1
    return image/255.0

def make_dataset(X,y,batch_size=BATCH_SIZE,training=False,mobilenetv2=False):
    ''' return X_train|X_test|X_val and y_train|y_test|y_val to dataset with shuffle of train data'''
    ds = tf.data.Dataset.from_tensor_slices((X,y))
    if training:
        ds = ds.shuffle(buffer_size=len(X),seed=RANDON_SEED,reshuffle_each_iteration=True)
    ds = ds.map(lambda image, label: (input_normalisation(image,mobilenetv2),label),num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds

def data_augmentation(rotation_factor=0.03,height_factor=0.03,width_factor=0.03,zoom_factor=0.05,fill_mode="nearest"):
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(
            factor=rotation_factor,
            fill_mode=fill_mode
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=height_factor,
            width_factor=width_factor,
            fill_mode=fill_mode
        ),
        tf.keras.layers.RandomZoom(
            height_factor=zoom_factor,
            width_factor=zoom_factor,
            fill_mode=fill_mode
        )
    ],name='data_augmentation')
