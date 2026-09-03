"""
Pipeline de Deep Learning com TensorFlow/Keras
Mini-Projeto: Inspeção de Qualidade de Peças de Fundição (Indústria 4.0)

Este módulo implementa os requisitos das Sprints 4, 5 e 6:
- Sprint 4: Ingestão de Dados (image_dataset_from_directory) e Data Augmentation dinâmico
- Sprint 5: Construção e Treinamento de Rede Neural Convolucional (CNN) Sequencial
- Sprint 6: Auditoria Gráfica (Curvas de Loss/Val_Loss, Acurácia, Matriz de Confusão e Diagnóstico de Overfitting)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Fixar sementes para reprodutibilidade
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

IMG_SIZE = (256, 256)
BATCH_SIZE = 32
DATASET_DIR = "dataset/casting_512x512"
FIGURES_DIR = "reports/figures"
MODELS_DIR = "models"


def get_data_augmentation_pipeline() -> tf.keras.Sequential:
    """
    Cria a camada de Data Augmentation para imunizar a rede contra
    variações industriais de esteira (rotação, zoom, variações luminosas).
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=SEED),
        layers.RandomRotation(0.2, fill_mode="nearest", seed=SEED),
        layers.RandomZoom(0.1, fill_mode="nearest", seed=SEED),
        layers.RandomBrightness(factor=0.1, seed=SEED),
        layers.RandomContrast(factor=0.1, seed=SEED),
    ], name="data_augmentation")


def load_datasets(dataset_path: str = DATASET_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Carrega o dataset particionado em Treino (80%) e Validação (20%)
    utilizando a API otimizada de datasets do Keras.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Diretório do dataset não encontrado em: {dataset_path}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary"
    )

    class_names = train_ds.class_names
    print(f"Classes identificadas: {class_names} (0: {class_names[0]}, 1: {class_names[1]})")

    # Otimização de performance com Cache e Prefetch
    autotune = tf.data.AUTOTUNE
    train_ds_opt = train_ds.cache().shuffle(1000, seed=SEED).prefetch(buffer_size=autotune)
    val_ds_opt = val_ds.cache().prefetch(buffer_size=autotune)

    return train_ds_opt, val_ds_opt, class_names, train_ds, val_ds


def plot_data_augmentation_samples(raw_train_ds, class_names, output_dir: str = FIGURES_DIR):
    """
    Gera um gráfico demonstrando o efeito do Data Augmentation sobre amostras reais da linha de produção.
    """
    os.makedirs(output_dir, exist_ok=True)
    aug_pipeline = get_data_augmentation_pipeline()

    for images, labels in raw_train_ds.take(1):
        sample_img = images[0]
        sample_label = class_names[int(labels[0].numpy().item())]

        plt.figure(figsize=(12, 6))
        # Imagem original
        plt.subplot(2, 4, 1)
        plt.imshow(sample_img.numpy().astype("uint8"))
        plt.title(f"Original ({sample_label})", fontweight='bold')
        plt.axis("off")

        # 7 Variações com Data Augmentation
        for i in range(7):
            augmented = aug_pipeline(tf.expand_dims(sample_img, 0), training=True)
            plt.subplot(2, 4, i + 2)
            plt.imshow(augmented[0].numpy().astype("uint8"))
            plt.title(f"Augment #{i+1}")
            plt.axis("off")

        plt.suptitle("Sprint 4: Demonstração de Data Augmentation Dinâmico na Peça de Fundição", fontsize=13, fontweight='bold')
        plt.tight_layout()
        out_path = os.path.join(output_dir, "sprint4_data_augmentation.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Salva figura de Data Augmentation em: {out_path}")
        break


def build_cnn_model(input_shape=(256, 256, 3)) -> tf.keras.Model:
    """
    Constrói a arquitetura Sequencial da CNN para classificação binária de peças industriais.
    """
    data_augmentation = get_data_augmentation_pipeline()

    model = models.Sequential([
        layers.Input(shape=input_shape),
        data_augmentation,
        layers.Rescaling(1.0 / 255.0, name="rescaling_norm"),

        # Bloco Convolucional 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name="conv2d_block1"),
        layers.MaxPooling2D((2, 2), name="maxpool_block1"),

        # Bloco Convolucional 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="conv2d_block2"),
        layers.MaxPooling2D((2, 2), name="maxpool_block2"),

        # Bloco Convolucional 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="conv2d_block3"),
        layers.MaxPooling2D((2, 2), name="maxpool_block3"),

        # Bloco Convolucional 4
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="conv2d_block4"),
        layers.MaxPooling2D((2, 2), name="maxpool_block4"),

        # Regularização e Achatamento
        layers.Dropout(0.3, name="dropout_features"),
        layers.Flatten(name="flatten_transition"),

        # Camada Densa com Regularização
        layers.Dense(64, activation='relu', name="dense_features"),
        layers.Dropout(0.2, name="dropout_dense"),

        # Camada de Decisão Binária (Sigmoid)
        layers.Dense(1, activation='sigmoid', name="binary_output")
    ], name="Industrial_Casting_CNN")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    return model


def train_model(model: tf.keras.Model, train_ds, val_ds, epochs: int = 15):
    """
    Executa o treinamento com callbacks industriais (ModelCheckpoint e EarlyStopping).
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_model_path = os.path.join(MODELS_DIR, "casting_cnn_model.keras")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    print("Iniciando treinamento do modelo CNN...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    return history, best_model_path


def plot_training_curves(history, output_dir: str = FIGURES_DIR):
    """
    Gera gráficos analíticos das curvas de Loss e Acurácia (Sprint 6).
    """
    os.makedirs(output_dir, exist_ok=True)
    hist = history.history
    epochs_range = range(1, len(hist['loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfico 1: Curva de Loss
    ax1.plot(epochs_range, hist['loss'], 'o-', color='#e74c3c', label='Treino (Loss)', linewidth=2)
    ax1.plot(epochs_range, hist['val_loss'], 's--', color='#c0392b', label='Validação (Val Loss)', linewidth=2)
    ax1.set_title('Curva de Perda (Binary Cross-Entropy Loss)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Época', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Gráfico 2: Curva de Acurácia
    ax2.plot(epochs_range, hist['accuracy'], 'o-', color='#2980b9', label='Treino (Acurácia)', linewidth=2)
    ax2.plot(epochs_range, hist['val_accuracy'], 's--', color='#27ae60', label='Validação (Val Acurácia)', linewidth=2)
    ax2.set_title('Curva de Acurácia (Accuracy vs Val Accuracy)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Época', fontsize=11)
    ax2.set_ylabel('Acurácia', fontsize=11)
    ax2.legend(loc='lower right', frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.6)

    fig.suptitle('Sprint 6: Auditoria Gráfica do Treinamento CNN - Inspeção de Peças Industriais', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "sprint6_loss_and_accuracy.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Salvo gráfico de curvas de treinamento em: {out_path}")


def evaluate_and_generate_metrics(model: tf.keras.Model, raw_val_ds, class_names, output_dir: str = FIGURES_DIR):
    """
    Avalia o modelo final na base de validação, gerando Matriz de Confusão,
    Relatório de Métricas e Amostras de Inferência.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    y_true = []
    y_pred_probs = []
    val_images_list = []

    for images, labels in raw_val_ds:
        probs = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().flatten())
        y_pred_probs.extend(probs.flatten())
        if len(val_images_list) < 12:
            val_images_list.extend(list(zip(images.numpy(), labels.numpy().flatten(), probs.flatten())))

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred_classes = (y_pred_probs >= 0.5).astype(int)

    # 1. Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f"Pred: {class_names[0]}", f"Pred: {class_names[1]}"],
                yticklabels=[f"Real: {class_names[0]}", f"Real: {class_names[1]}"],
                cbar=False, annot_kws={"size": 14, "weight": "bold"})
    plt.title("Sprint 6: Matriz de Confusão na Validação", fontsize=13, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "sprint6_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Salva Matriz de Confusão em: {cm_path}")

    # 2. Relatório de Classificação
    report = classification_report(y_true, y_pred_classes, target_names=class_names, output_dict=True)
    report_text = classification_report(y_true, y_pred_classes, target_names=class_names)
    print("\n--- Relatório de Classificação Final ---")
    print(report_text)

    # Salvar métricas em JSON
    metrics_summary = {
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }
    with open("reports/training_metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)

    # 3. Grid com Amostras de Inferência
    plt.figure(figsize=(15, 10))
    for i in range(min(12, len(val_images_list))):
        img, true_lbl, prob = val_images_list[i]
        lbl_int = int(np.asarray(true_lbl).item())
        pred_lbl = 1 if prob >= 0.5 else 0
        pred_name = class_names[pred_lbl]
        true_name = class_names[lbl_int]
        is_correct = pred_lbl == lbl_int
        color = "green" if is_correct else "red"

        plt.subplot(3, 4, i + 1)
        plt.imshow(img.astype("uint8"))
        confidence = prob if pred_lbl == 1 else (1 - prob)
        plt.title(f"Real: {true_name}\nPred: {pred_name} ({confidence*100:.1f}%)",
                  color=color, fontweight='bold', fontsize=10)
        plt.axis("off")

    plt.suptitle("Sprint 6: Amostras de Predição no Lote de Validação", fontsize=14, fontweight='bold')
    plt.tight_layout()
    samples_path = os.path.join(output_dir, "sprint6_sample_predictions.png")
    plt.savefig(samples_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Salvas predições de amostra em: {samples_path}")

    return report


def run_full_deep_learning_pipeline(epochs: int = 15):
    """Executa o pipeline completo de deep learning."""
    print("=== SPRINT 4: Ingestão de Dados e Data Augmentation ===")
    train_ds, val_ds, class_names, raw_train_ds, raw_val_ds = load_datasets()
    plot_data_augmentation_samples(raw_train_ds, class_names)

    print("\n=== SPRINT 5: Construção e Treinamento do Modelo CNN ===")
    model = build_cnn_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    model.summary()

    history, best_model_path = train_model(model, train_ds, val_ds, epochs=epochs)

    print("\n=== SPRINT 6: Auditoria Gráfica e Avaliação Final ===")
    plot_training_curves(history)
    best_model = tf.keras.models.load_model(best_model_path)
    evaluate_and_generate_metrics(best_model, raw_val_ds, class_names)
    print("\nPipeline de Deep Learning finalizado com sucesso!")


if __name__ == "__main__":
    run_full_deep_learning_pipeline(epochs=15)
