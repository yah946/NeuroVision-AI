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
    callbacks,
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
def evaluate_model(name,model,history,classes_name,num_classes,y_test_encoded,test_ds):
    safe_name = name.lower().replace(' ','_')

    history_metrics = history.history
    epochs = np.arange(1, len(history.epoch) + 1)

    for metric,values in history_metrics.items():
        for epoch, value in enumerate(values,start=1):
            mlflow.log_metric(metric, float(value), step=epoch)

    for metric, values in history_metrics.items():
        if metric.startswith("val_"):
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, values, label=metric)
        val_metric = f"val_{metric}"
        if val_metric in history_metrics:
            ax.plot(epoch,history_metrics[val_metric],label=val_metric)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_title(f"{name}: {metric}")
        ax.legend()
        fig.tight_layout()
        mlflow.log_figure(fig,f"training_plots/{safe_name}_{metric}.png")
        plt.close(fig)

    test_metrics = model.evaluate(test_ds,return_dict=True,verbose=0)

    for metric, value in test_metrics.items():
        mlflow.log_metric(
            f"test_{metric}",
            float(value),
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([metric], [value])
        ax.set_ylabel(metric)
        ax.set_title(f"{name}: test {metric}")
        fig.tight_layout()

        mlflow.log_figure(
            fig,
            f"test_plots/{safe_name}_test_{metric}.png",
        )
        plt.close(fig)
    
    y_pred_proba = model.predict(test_ds)
    y_pred = np.argmax(y_pred_proba,axis=1)

    cm = confusion_matrix(y_test_encoded, y_pred, labels=np.arange(num_classes))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes_name,
        yticklabels=classes_name,
        ax=ax
    )
    ax.xlabel("Predicted class")
    ax.ylabel("True class(Reality)")
    ax.set_title(f"{name}'s Confusion Matrix")

    mlflow.log_figure(fig,f"evalutation/{name}_confusion_matrix.png")
    plt.close(fig)

    mlflow.tensorflow.log_model(model,artifact_path="model")

    return test_metrics


def pipeline(models,best_model_path='artifacts/checkpoints',backup_dir='artifacts/backups'):
    # ================================================
    # Load & Split data into train, test and val
    # ================================================
    X,y = load_data()
    X_train,y_train,X_val,y_val,X_test,y_test = train_test_val_split(X,y)
    # ================================================
    # Encode target using label encoding
    # ================================================
    y_train_encoded,y_val_encoded,y_test_encoded,encoder = label_encoder(y_train,y_val,y_test)
    # ================================================
    # Get the number of classes, together with names
    # ================================================
    classes_name = encoder.classes_.tolist()
    num_classes = len(encoder.classes_)
    # ================================================
    # Create Datasets for the Custom CNN
    # ================================================
    train_ds = make_dataset(X_train,y_train_encoded,training=True)
    val_ds = make_dataset(X_val,y_val_encode)
    test_ds = make_dataset(X_test,y_test_encode)
    # ================================================
    # Set the MLflow Experiment
    # ================================================
    mlflow.set_experiment('image-classification')
    # results = {}
    with mlflow.start_run(run_name='custom_cnn'):
        run_id = mlflow.active_run().info.run_id
        checkpoint_path = Path(best_model_path)/run_id/'best_model.keras'
        run_backup_dir = Path(backup_dir)/run_id
        
        augmentation = data_augmentation()
        cnn_model,_ = custum_cnn_model(num_classes,augmentation)
        
        mlflow.log_params({
            'model name': name,
            'num_classes': num_classes,        
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'train_size': len(X_train)/len(X),
            'val_size': len(X_val)/len(X),
            'test_size': len(X_test)/len(X),
            'normalization': ('[-1,1]' if is_mobilenetv2_model else '[0,1]'),
            'augmentation': 'rotation_translation_zoom'
        })

        mlflow.log_dict({"classes": encoder.classes_.tolist()},"label_classes.json")
        cnn_history = cnn_model.fit(train_ds,validation_data=val_ds,epochs=EPOCHS,callbacks=callbacks(checkpoint_path,run_backup_dir))

        cnn_model = tf.keras.models.load_model(checkpoint_path)
        evaluate_model('custom_cnn',cnn_model,cnn_histroy,classes_name,num_classes,y_test_encoded,test_ds)
        mlflow.tensorflow.log_model(cnn_model,artifact_path="model")
    
    # ================================================
    # Create MobileNetV2 Normalized Datasets
    # ================================================
    train_ds = make_dataset(X_train,y_train_encoded,training=True,mobilenetv2=True)
    val_ds = make_dataset(X_val,y_val_encode,mobilenetv2=True)
    test_ds = make_dataset(X_test,y_test_encode,mobilenetv2=True)
    # ================================================
    # Train MobileNetV2 with Frozen Backbone
    # ================================================
    with mlflow.start_run(run_name='frozen_mobilenetv2_model'):
        run_id = mlflow.active_run().info.run_id
        checkpoint_path = Path(best_model_path)/run_id/'best_model.keras'
        run_backup_dir = Path(backup_dir)/run_id
        
        augmentation = data_augmentation()
        model,base_model = mobile_net_v2_frozen_model(num_classes,augmentation)
        
        mlflow.log_params({
            'model name': name,
            'num_classes': num_classes,        
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'train_size': len(X_train)/len(X),
            'val_size': len(X_val)/len(X),
            'test_size': len(X_test)/len(X),
            'normalization': ('[-1,1]' if is_mobilenetv2_model else '[0,1]'),
            'augmentation': 'rotation_translation_zoom'
        })

        mlflow.log_dict({"classes": encoder.classes_.tolist()},"label_classes.json")
        history = model.fit(train_ds,validation_data=val_ds,epochs=EPOCHS,callbacks=callbacks(checkpoint_path,run_backup_dir))

        model = tf.keras.models.load_model(checkpoint_path)
        evaluate_model('mobile_net_v2_frozen_model',model,histroy,classes_name,num_classes,y_test_encoded,test_ds)
        mlflow.tensorflow.log_model(model,artifact_path="model")

    with mlflow.start_run(run_name='fine_tuning'):
        run_id = mlflow.active_run().info.run_id
        checkpoint_path = Path(best_model_path)/run_id/'best_model.keras'
        run_backup_dir = Path(backup_dir)/run_id
                    
        augmentation = data_augmentation()
        finetuning_model = mobile_net_v2_finetuning_model(model,base_model)
        
        mlflow.log_params({
            'model name': name,
            'num_classes': num_classes,        
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'train_size': len(X_train)/len(X),
            'val_size': len(X_val)/len(X),
            'test_size': len(X_test)/len(X),
            'normalization': ('[-1,1]' if is_mobilenetv2_model else '[0,1]'),
            'augmentation': 'rotation_translation_zoom'
        })

        mlflow.log_dict({"classes": encoder.classes_.tolist()},"label_classes.json")
        finetuning_history = model.fit(train_ds,validation_data=val_ds,epochs=EPOCHS,callbacks=callbacks(checkpoint_path,run_backup_dir))

        model = tf.keras.models.load_model(checkpoint_path)
        evaluate_model('finetuning',finetuning_model,finetuning_history,classes_name,num_classes,y_test_encoded,test_ds)
        mlflow.tensorflow.log_model(finetuning_model,artifact_path="model")