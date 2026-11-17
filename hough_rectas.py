import numpy as np
import matplotlib.pyplot as plt

def generar_imagen_rectas():
    # Crea una imagen sintética en negro (zeros) de 100x100
    img = np.zeros((100, 100), dtype=np.uint8)
    # Dibuja una línea horizontal
    img[20, :] = 255
    # Dibuja una línea diagonal (y = x)
    for i in range(10, 90):
        img[i, i] = 255
    return img

def transformada_hough_rectas(img):
    height, width = img.shape
    # Diagonal de la imagen para definir el rango máximo de rho
    diag = int(np.ceil(np.sqrt(height**2 + width**2)))
    thetas = np.deg2rad(np.arange(-90, 90, 1))
    rhos = np.arange(-diag, diag + 1)
    
    # Acumulador (rho x theta)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
    
    # Obtener coordenadas de los píxeles de borde (valores > 0)
    y_idxs, x_idxs = np.nonzero(img)
    
    # Votación en el espacio de Hough
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    
    for x, y in zip(x_idxs, y_idxs):
        for t_idx in range(len(thetas)):
            rho = x * cos_t[t_idx] + y * sin_t[t_idx]
            r_idx = int(rho + diag)
            accumulator[r_idx, t_idx] += 1
            
    return accumulator, rhos, thetas

# Ejecución de prueba
if __name__ == "__main__":
    imagen = generar_imagen_rectas()
    acumulador, rhos, thetas = transformada_hough_rectas(imagen)
    
    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(imagen, cmap='gray')
    axes[0].set_title("Imagen Sintética (Rectas)")
    axes[0].axis('off')
    
    axes[1].imshow(acumulador, extent=[np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), rhos[-1], rhos[0]], cmap='jet', aspect='auto')
    axes[1].set_title("Acumulador de Hough (Espacio rho-theta)")
    axes[1].set_xlabel("Theta (grados)")
    axes[1].set_ylabel("Rho (píxeles)")
    
    plt.tight_layout()
    plt.show()