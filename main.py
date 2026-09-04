"""
Ponto de Entrada Principal (Main Pipeline)
Mini-Projeto: Inspeção de Qualidade de Peças de Fundição (Indústria 4.0)

Este script orquestra a execução das 6 Sprints do projeto:
- Sprint 1: Verificação de ambiente e integridade dos dados
- Sprints 2 & 3: Visão Computacional Clássica (OpenCV)
- Sprints 4, 5 & 6: Deep Learning (Keras/TensorFlow) e Auditoria Gráfica
"""

import argparse
import sys
import os

from src.classical_vision import generate_classical_vision_reports
from src.model_pipeline import run_full_deep_learning_pipeline


def check_environment():
    """Verifica se as pastas e dados essenciais estão configurados."""
    print("=" * 60)
    print(" SPRINT 1: VERIFICAÇÃO DE AMBIENTE E DADOS")
    print("=" * 60)
    dataset_dir = "dataset/casting_512x512"
    if not os.path.exists(dataset_dir):
        print(f"[ERRO] Diretório do dataset '{dataset_dir}' não encontrado!")
        print("Certifique-se de baixar e extrair o arquivo dataset.zip.")
        sys.exit(1)
    
    def_count = len(os.listdir(os.path.join(dataset_dir, "def_front")))
    ok_count = len(os.listdir(os.path.join(dataset_dir, "ok_front")))
    print(f"[OK] Dataset carregado:")
    print(f"     - Peças Defeituosas (def_front): {def_count} imagens")
    print(f"     - Peças Aprovadas (ok_front):    {ok_count} imagens")
    print(f"     - Total de Imagens:             {def_count + ok_count} imagens")


def run_classical_pipeline():
    """Executa as etapas de visão computacional clássica (Sprints 2 e 3)."""
    print("\n" + "=" * 60)
    print(" SPRINTS 2 & 3: VISÃO COMPUTACIONAL CLÁSSICA (OpenCV)")
    print("=" * 60)
    def_sample = "dataset/casting_512x512/def_front/cast_def_0_0.jpeg"
    ok_sample = "dataset/casting_512x512/ok_front/cast_ok_0_1018.jpeg"
    
    print(f"Processando amostras: \n - Defeituosa: {def_sample}\n - Normal: {ok_sample}")
    generate_classical_vision_reports(def_sample, ok_sample)
    print("[OK] Sprints 2 e 3 concluídas com sucesso. Figuras salvas em reports/figures/")


def run_deep_learning_pipeline(epochs: int = 15):
    """Executa as etapas de ingestão, augmentation, CNN e auditoria (Sprints 4, 5 e 6)."""
    print("\n" + "=" * 60)
    print(" SPRINTS 4, 5 & 6: DEEP LEARNING E AUDITORIA (TensorFlow/Keras)")
    print("=" * 60)
    run_full_deep_learning_pipeline(epochs=epochs)
    print("[OK] Sprints 4, 5 e 6 concluídas com sucesso.")


def main():
    parser = argparse.ArgumentParser(description="Pipeline de Inspeção de Qualidade Industrial (OpenCV + CNN)")
    parser.add_argument(
        "--mode",
        choices=["all", "classical", "deep_learning", "check"],
        default="all",
        help="Modo de execução: 'all' (completo), 'classical' (OpenCV), 'deep_learning' (CNN) ou 'check' (ambiente)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Número de épocas de treinamento para a CNN (padrão: 15)"
    )
    args = parser.parse_args()

    check_environment()

    if args.mode in ["all", "classical"]:
        run_classical_pipeline()

    if args.mode in ["all", "deep_learning"]:
        run_deep_learning_pipeline(epochs=args.epochs)

    print("\n" + "=" * 60)
    print(" PIPELINE EXECUTADO COM SUCESSO!")
    print("=" * 60)


if __name__ == "__main__":
    main()
