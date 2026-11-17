import numpy as np
import matplotlib.pyplot as plt

def generar_imagen_circulo():
    img = np.zeros((120, 120), dtype=np.uint8)
    # Centro conocido y radio fijo R = 30
    cx, cy, R = 60, 60, 30
    angles = np.linspace(0, 2 * np.pi, 200)
    for a in angles:
        x = int(cx + R * np.cos(a))
        y = int(cy + R * np.sin(a))
        if 0 <= x < 120 and 0 <= y < 120:
            img[y, x] = 255
    return img, cx, cy, R

def transformada_hough_circulos(img, R):
    height, width = img.shape
    accumulator = np.zeros((height, width), dtype=np.int32)
    
    y_idxs, x_idxs = np.nonzero(img)
    theta = np.deg2rad(np.arange(0, 360, 2))
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    for x, y in zip(x_idxs, y_idxs):
        for t in range(len(theta)):
            x0 = int(x - R * cos_t[t])
            y0 = int(y - R * sin_t[t])
            if 0 <= x0 < width and 0 <= y0 < height:
                accumulator[y0, x0] += 1
                
    return accumulator

# Ejecución de prueba
if __name__ == "__main__":
    imagen, cx, cy, R = generar_imagen_circulo()
    acumulador = transformada_hough_circulos(imagen, R)
    
    # Encontrar el centro detectado (máximo en el acumulador)
    min_y, min_x = np.unravel_index(np.argmax(acumulador), acumulador.shape)
    print(f"Centro real: ({cx}, {cy}) | Centro detectado por Hough: ({min_x}, {min_y})")
    
    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(imagen, cmap='gray')
    axes[0].set_title(f"Circunferencia (Radio R={R})")
    axes[0].axis('off')
    
    axes[1].imshow(acumulador, cmap='hot')
    axes[1].set_title("Acumulador de Centros (Plano X0-Y0)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()