"""
Módulo de Visão Computacional Clássica (OpenCV)
Mini-Projeto: Inspeção de Qualidade de Peças de Fundição (Indústria 4.0)

Este módulo implementa os requisitos das Sprints 2 e 3:
- Conversão para Escala de Cinza
- Suavização de Ruídos Industriais (Gaussian Blur e Median Blur)
- Detecção de Bordas (Canny e Sobel)
- Limiarização (Thresholding Otsu e Adaptativo)
- Operações Morfológicas (Erosão, Dilatação, Abertura, Fechamento e Gradiente)
- Isolamento e Destaque Visual de Defeitos (Fissuras/Porosidades)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converte imagem BGR para escala de cinza."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_blur_filters(gray_image: np.ndarray, kernel_size: int = 5) -> dict:
    """
    Aplica filtros de suavização para eliminação de ruídos de textura metálica e esteira.
    Retorna dicionário com Gaussian Blur e Median Blur.
    """
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    gaussian = cv2.GaussianBlur(gray_image, (k, k), sigmaX=1.5)
    median = cv2.medianBlur(gray_image, k)
    return {
        "gaussian": gaussian,
        "median": median
    }


def apply_edge_detection(blurred_image: np.ndarray, low_thresh: int = 50, high_thresh: int = 150) -> dict:
    """
    Aplica algoritmos de detecção de bordas estruturais: Canny e Sobel.
    """
    # Canny Edge Detector
    canny = cv2.Canny(blurred_image, low_thresh, high_thresh)
    
    # Sobel Operator (Gradientes X e Y)
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_mag = np.uint8(np.clip(sobel_mag / (sobel_mag.max() + 1e-6) * 255, 0, 255))
    
    return {
        "canny": canny,
        "sobel_x": sobel_x,
        "sobel_y": sobel_y,
        "sobel": sobel_mag
    }


def apply_thresholding(blurred_image: np.ndarray) -> dict:
    """
    Aplica técnicas de limiarização: Otsu e Limiarização Adaptativa.
    """
    # Otsu Thresholding
    _, otsu = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Adaptive Thresholding (Gaussian)
    adaptive = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    return {
        "otsu": otsu,
        "adaptive": adaptive
    }


def apply_morphological_ops(binary_image: np.ndarray, kernel_size: int = 3) -> dict:
    """
    Aplica operações morfológicas para refinar o isolamento de defeitos.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    erosion = cv2.erode(binary_image, kernel, iterations=1)
    dilation = cv2.dilate(binary_image, kernel, iterations=1)
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=1)
    gradient = cv2.morphologyEx(binary_image, cv2.MORPH_GRADIENT, kernel)
    
    return {
        "erosion": erosion,
        "dilation": dilation,
        "opening": opening,
        "closing": closing,
        "gradient": gradient
    }


def isolate_and_highlight_defects(orig_image_bgr: np.ndarray) -> dict:
    """
    Pipeline completo de processamento clássico para destacar defeitos estruturais.
    Retorna o mapa de defeito e a imagem original com contornos/máscara destacando o defeito em vermelho.
    """
    gray = convert_to_grayscale(orig_image_bgr)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    
    # Detecção de bordas
    canny = cv2.Canny(blurred, 40, 120)
    
    # Máscara circular da região de interesse (elimina bordas externas da peça circular de fundição)
    h, w = gray.shape
    center = (w // 2, h // 2)
    radius = int(min(h, w) * 0.42)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(roi_mask, center, radius, 255, -1)
    
    # Aplicar ROI mask nas bordas internas
    internal_canny = cv2.bitwise_and(canny, roi_mask)
    
    # Fechamento e dilatação morfológica para unir trincas/fissuras
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_edges = cv2.dilate(internal_canny, kernel, iterations=2)
    closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Encontrar contornos na área interna
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    highlighted = orig_image_bgr.copy()
    defect_detected = False
    defect_count = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area < 15000:  # Filtrar ruídos muito pequenos ou área total
            defect_detected = True
            defect_count += 1
            # Desenhar contorno em vermelho
            cv2.drawContours(highlighted, [cnt], -1, (0, 0, 255), 2)
            # Desenhar bounding box
            x, y, bw, bh = cv2.boundingRect(cnt)
            cv2.rectangle(highlighted, (x, y), (x + bw, y + bh), (0, 255, 255), 2)
            cv2.putText(highlighted, f"Defeito #{defect_count}", (x, max(15, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return {
        "gray": gray,
        "blurred": blurred,
        "canny": canny,
        "closed_edges": closed_edges,
        "highlighted": highlighted,
        "defect_detected": defect_detected,
        "defect_count": defect_count
    }


def generate_classical_vision_reports(def_img_path: str, ok_img_path: str, output_dir: str = "reports/figures"):
    """
    Gera e salva figuras completas de alta resolução demonstrando todas as etapas das Sprints 2 e 3.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    img_def_bgr = cv2.imread(def_img_path)
    img_ok_bgr = cv2.imread(ok_img_path)
    
    if img_def_bgr is None or img_ok_bgr is None:
        raise ValueError(f"Não foi possível carregar as imagens: {def_img_path} ou {ok_img_path}")
        
    img_def_rgb = cv2.cvtColor(img_def_bgr, cv2.COLOR_BGR2RGB)
    img_ok_rgb = cv2.cvtColor(img_ok_bgr, cv2.COLOR_BGR2RGB)
    
    # ----------------------------------------------------
    # FIGURA 1: Sprint 2 - Escala de Cinza e Suavização de Ruído
    # ----------------------------------------------------
    gray_def = convert_to_grayscale(img_def_bgr)
    gray_ok = convert_to_grayscale(img_ok_bgr)
    
    blur_def = apply_blur_filters(gray_def, kernel_size=5)
    blur_ok = apply_blur_filters(gray_ok, kernel_size=5)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Sprint 2: Análise Exploratória Clássica - Escala de Cinza e Suavização de Ruído", fontsize=14, fontweight='bold')
    
    # Linha 1: Peça com Defeito
    axes[0, 0].imshow(img_def_rgb)
    axes[0, 0].set_title("Defeituosa: Original RGB")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(gray_def, cmap="gray")
    axes[0, 1].set_title("Defeituosa: Grayscale")
    axes[0, 1].axis("off")
    
    axes[0, 2].imshow(blur_def["gaussian"], cmap="gray")
    axes[0, 2].set_title("Defeituosa: Gaussian Blur (5x5)")
    axes[0, 2].axis("off")
    
    axes[0, 3].imshow(blur_def["median"], cmap="gray")
    axes[0, 3].set_title("Defeituosa: Median Blur (5x5)")
    axes[0, 3].axis("off")
    
    # Linha 2: Peça Normal (OK)
    axes[1, 0].imshow(img_ok_rgb)
    axes[1, 0].set_title("Peça OK: Original RGB")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(gray_ok, cmap="gray")
    axes[1, 1].set_title("Peça OK: Grayscale")
    axes[1, 1].axis("off")
    
    axes[1, 2].imshow(blur_ok["gaussian"], cmap="gray")
    axes[1, 2].set_title("Peça OK: Gaussian Blur (5x5)")
    axes[1, 2].axis("off")
    
    axes[1, 3].imshow(blur_ok["median"], cmap="gray")
    axes[1, 3].set_title("Peça OK: Median Blur (5x5)")
    axes[1, 3].axis("off")
    
    plt.tight_layout()
    sprint2_path = os.path.join(output_dir, "sprint2_grayscale_and_blur.png")
    plt.savefig(sprint2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Salvo relatório Sprint 2 em: {sprint2_path}")
    
    # ----------------------------------------------------
    # FIGURA 2: Sprint 3 - Limiarização, Bordas e Morfologia
    # ----------------------------------------------------
    edge_def = apply_edge_detection(blur_def["gaussian"])
    edge_ok = apply_edge_detection(blur_ok["gaussian"])
    
    thresh_def = apply_thresholding(blur_def["gaussian"])
    thresh_ok = apply_thresholding(blur_ok["gaussian"])
    
    morph_def = apply_morphological_ops(thresh_def["otsu"], kernel_size=3)
    morph_ok = apply_morphological_ops(thresh_ok["otsu"], kernel_size=3)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Sprint 3: Destaque de Características - Limiarização, Bordas (Canny/Sobel) e Morfologia", fontsize=14, fontweight='bold')
    
    # Linha 1: Peça Defeituosa
    axes[0, 0].imshow(thresh_def["otsu"], cmap="gray")
    axes[0, 0].set_title("Defeituosa: Otsu Threshold")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(edge_def["canny"], cmap="gray")
    axes[0, 1].set_title("Defeituosa: Canny Edges")
    axes[0, 1].axis("off")
    
    axes[0, 2].imshow(edge_def["sobel"], cmap="gray")
    axes[0, 2].set_title("Defeituosa: Sobel Magnitude")
    axes[0, 2].axis("off")
    
    axes[0, 3].imshow(morph_def["dilation"], cmap="gray")
    axes[0, 3].set_title("Defeituosa: Dilatação Morfológica")
    axes[0, 3].axis("off")
    
    axes[0, 4].imshow(morph_def["closing"], cmap="gray")
    axes[0, 4].set_title("Defeituosa: Fechamento Morfológico")
    axes[0, 4].axis("off")
    
    # Linha 2: Peça OK
    axes[1, 0].imshow(thresh_ok["otsu"], cmap="gray")
    axes[1, 0].set_title("Peça OK: Otsu Threshold")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(edge_ok["canny"], cmap="gray")
    axes[1, 1].set_title("Peça OK: Canny Edges")
    axes[1, 1].axis("off")
    
    axes[1, 2].imshow(edge_ok["sobel"], cmap="gray")
    axes[1, 2].set_title("Peça OK: Sobel Magnitude")
    axes[1, 2].axis("off")
    
    axes[1, 3].imshow(morph_ok["dilation"], cmap="gray")
    axes[1, 3].set_title("Peça OK: Dilatação Morfológica")
    axes[1, 3].axis("off")
    
    axes[1, 4].imshow(morph_ok["closing"], cmap="gray")
    axes[1, 4].set_title("Peça OK: Fechamento Morfológico")
    axes[1, 4].axis("off")
    
    plt.tight_layout()
    sprint3_path = os.path.join(output_dir, "sprint3_edges_and_morphology.png")
    plt.savefig(sprint3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Salvo relatório Sprint 3 em: {sprint3_path}")
    
    # ----------------------------------------------------
    # FIGURA 3: Destaque e Isolamento Visual dos Defeitos (Comparativo Final OpenCV)
    # ----------------------------------------------------
    iso_def = isolate_and_highlight_defects(img_def_bgr)
    iso_ok = isolate_and_highlight_defects(img_ok_bgr)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Isolamento Visual e Segmentação de Defeitos de Fundição (OpenCV)", fontsize=14, fontweight='bold')
    
    # Defeituosa
    axes[0, 0].imshow(img_def_rgb)
    axes[0, 0].set_title("Peça com Defeito (Original)")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(iso_def["closed_edges"], cmap="gray")
    axes[0, 1].set_title("Mapa Binário de Trincas/Fissuras")
    axes[0, 1].axis("off")
    
    axes[0, 2].imshow(cv2.cvtColor(iso_def["highlighted"], cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f"Defeitos Isolados (Detectados: {iso_def['defect_count']})")
    axes[0, 2].axis("off")
    
    # OK
    axes[1, 0].imshow(img_ok_rgb)
    axes[1, 0].set_title("Peça sem Defeito (Original OK)")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(iso_ok["closed_edges"], cmap="gray")
    axes[1, 1].set_title("Mapa Binário Limpo (Sem trincas)")
    axes[1, 1].axis("off")
    
    axes[1, 2].imshow(cv2.cvtColor(iso_ok["highlighted"], cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("Inspeção Aprovada (Sem defeitos)")
    axes[1, 2].axis("off")
    
    plt.tight_layout()
    sprint3_iso_path = os.path.join(output_dir, "sprint3_defect_isolation_comparison.png")
    plt.savefig(sprint3_iso_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Salvo comparativo de isolamento em: {sprint3_iso_path}")


if __name__ == "__main__":
    def_sample = "dataset/casting_512x512/def_front/cast_def_0_0.jpeg"
    ok_sample = "dataset/casting_512x512/ok_front/cast_ok_0_1018.jpeg"
    generate_classical_vision_reports(def_sample, ok_sample)
