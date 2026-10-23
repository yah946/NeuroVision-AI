import tensorflow as tf
import mlflow
import os

from data_loader import (
    load_data,
    imgs_per_class,
    read_samples,
    train_test_val_split,
    label_encoder,
    input_normalisation,
    make_dataset,
    data_augmentation
)
from models import (
    callabacks,
    custum_cnn_model,
    mobile_net_v2_frozen_model,
    mobile_net_v2_finetuning_model
)
SEED = 10
BATCH_SIZE = 32
EPOCHS = 30

models = {
    'custom_cnn_model': (custum_cnn_model,False),
    'mobilenetv2_frozen_model': (mobile_net_v2_frozen_model,True),
}

tf.keras.utils.set_random_seed(SEED)

def pipeline(models,best_model_path='artifacts/checkpoints',backup_dir='artifacts/backups'):
    X,y = load_data()
    results = []
    X_train,y_train,X_val,y_val,X_test,y_test = train_test_val_split(X,y)
    y_train_encoded,y_val_encoded,y_test_encoded,encoder = label_encoder(y_train,y_val,y_test)
    classes_name = encoder.classes_.tolist()
    num_classes = len(encoder.classes_)
    mlflow.set_experiment('image-classification')
    for name,(model,is_mobilenetv2_model) in models.items():
        with mlflow.start_run(run_name=name):
            run_id = mlflow.active_run().info.run_id
            checkpoint_path = Path(best_model_path)/run_id/'best_model.keras'
            run_backup_dir = Path(backup_dir)/run_id
            train_ds = make_dataset(X_train,y_train_encoded,training=True, mobilenetv2=is_mobilenetv2_model)
            val_ds = make_dataset(X_val,y_val_encoded,mobilenetv2=is_mobilenetv2_model)
            test_ds = make_dataset(X_test,y_test_encoded,mobilenetv2=is_mobilenetv2_model)
            augmentation = data_augmentation()
            mlflow.log_params({
                'model name': name,
                'num_classes': num_classes,        
                'batch_size': BATCH_SIZE,
                'epochs': EPOCHS,
                'train_size': 0.5,
                'test_size': 0.25,
                'val_size': 0.25,
                'normalization': ('[-1,1]' if is_mobilenetv2_model else '[0,1]'),
                'augmentation': 'rotation_translation_zoom'
            })
            history = model.fit(train_ds,validation_data=val_ds,epochs=EPOCHS,callbacks=callabacks(best_model_path,run_backup_dir))

            for metric,values in history.history.items():
                for epoch, value in enumerate(values):
                    mlflow.log_metric(metric, float(value), step=epoch+1)
                    plot(metric, "test "+metric)
                    mlflow.log_artifact("test "+metric)
            
            test_metrics = model.evaluate(test_ds,return_dict=True)
            for metric, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric}", float(value))
            
            mlflow.log_dict({"classes": encoder.classes_.tolist()},"label_classes.json")
            
            y_pred_proba = model.predict(test_ds)
            y_pred = np.argmax(y_pred_proba,axis=1)

            cm = confusion_matrix(y_test_encoded, y_pred, labels=np.arange(num_classes))
            fig, ax = plt.figure(figsize=(8, 6))
            convention_matrix = sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax
            )
            ax.xlabel("Predicted class")
            ax.ylabel("True class(Reality)")
            ax.set_title(f"{name}'s Confusion Matrix")

            lower_case_model_name = name.lower()
            model_name_without_spaces = lower_case_model_name.replace(" ", "_")
            confusion_matrix_path = '../mlruns/'+'confusion_matrix_'+model_name_without_spaces+'.png'

            mlflow.log_figure(fig,confusion_matrix_path)
            plt.close()
            
            mlflow.tensorflow.log_model(model,artifact_path="model")
            results[name] = test_metrics
    return results
