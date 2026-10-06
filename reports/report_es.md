# SliceGAN con discriminadores Vision Transformer y segmentación con SAM

Vision Transformers — FIUBA. 

Trabajo individual.

## 1. Objetivo del proyecto

Las microestructuras tridimensionales son necesarias para calcular propiedades efectivas de materiales (por
ejemplo, como elementos de volumen representativos para análisis por elementos finitos), pero la
adquisición 3D (micro-CT, FIB-SEM) es costosa y a menudo no está disponible, mientras que las micrografías
2D son baratas, rápidas y en general de mayor resolución. Este proyecto genera microestructuras 3D de dos
fases a partir de una única micrografía 2D con SliceGAN (Kench & Cooper, 2021) y estudia si los Vision Transformers la mejoran,
ya sea como crítico adversarial o como etapa previa de segmentación.

### 1.1 Antecedentes: SliceGAN

**El problema.** Una micrografía 2D de un material contiene, estadísticamente, buena parte de la
información de su estructura 3D: en un material isótropo (sin dirección preferencial), todo corte plano
del volumen tiene la misma estadística (fracciones de fase, tamaños y formas de los rasgos, correlaciones
espaciales) que cualquier otro. La tarea es entonces producir volúmenes 3D cuyas secciones 2D sean
estadísticamente indistinguibles de la micrografía. Los métodos clásicos de reconstrucción optimizan un
volumen 3D para ajustar descriptores estadísticos elegidos (por ejemplo, la correlación de dos puntos), son
lentos (horas para 10⁶ vóxeles) y solo reproducen los descriptores que se les pidió ajustar.

**Redes generativas adversariales.** Una GAN entrena dos redes enfrentadas: un *generador* G que transforma
ruido aleatorio en muestras, y un *discriminador* (crítico) D que intenta distinguir las muestras generadas
de las reales. G se actualiza para engañar a D. En el equilibrio, la distribución generada coincide con la
real. Las GAN aprenden la estadística directamente de los datos en lugar de descriptores elegidos a mano y,
una vez entrenadas, generan muestras nuevas en segundos.

**La idea clave de SliceGAN: expansión de dimensionalidad mediante cortes** (Kench & Cooper, 2021).
Normalmente una GAN necesita datos de entrenamiento con la misma dimensionalidad que su salida, pero aquí no
hay datos 3D. SliceGAN combina un **generador 3D** con un **discriminador 2D** y los conecta mediante un
paso de corte:

1. G transforma un tensor latente z (ruido gaussiano de forma 32 × 4 × 4 × 4 en este proyecto) en un
   volumen de 64³ con un canal por fase (softmax, es decir, una codificación one-hot suave).
2. El volumen generado se corta en sus 64 rebanadas a lo largo de cada uno de los tres ejes
   (3 × 64 = 192 rebanadas).
3. D, una red convolucional 2D, puntúa cada rebanada y un recorte aleatorio del mismo tamaño de la
   micrografía real. Para un material isótropo, un único D sirve para las tres direcciones.
4. Se combinan las pérdidas de todas las rebanadas y se actualizan ambas redes. Como cada rebanada del
   volumen debe parecerse a la micrografía, G aprende una estructura 3D cuyas secciones en x, y y z
   reproducen la estadística 2D.

![Entrenamiento de SliceGAN](figures/slicegan_schematic.png)

El entrenamiento en la versión original usa la pérdida de Wasserstein con penalización de gradiente (WGAN-GP; Gulrajani et al.,
2017), en la que D es un *crítico* no acotado que estima la distancia de Wasserstein entre rebanadas reales
y generadas, con 5 actualizaciones del crítico por cada actualización del generador. Usar WGAN implica
que esa distancia sólo es válida si el crítico es una función
1-Lipschitz (norma del gradiente respecto de la entrada ≤ 1). En lugar de recortar los pesos como la WGAN
original, la GP impone la restricción de forma suave: se interpola al azar entre una rebanada real x y una
generada x̃, x̂ = εx + (1−ε)x̃ con ε ~ U[0, 1], y se suma a la pérdida del crítico el término
λ(‖∇D(x̂)‖₂ − 1)², con λ = 10. El discriminador ve las 64 rebanadas por dirección de cada volumen generado y usa un lote del generador
del doble del lote del crítico (m_G = 2 m_D), configuración que los autores encontraron más eficiente.

**Diseño del generador: densidad de información uniforme.** Las primeras versiones de SliceGAN producían
peor calidad cerca de los bordes del volumen. La causa es la convolución transpuesta: un vóxel cerca del
borde de la salida recibe contribuciones de menos posiciones del kernel que uno central, de modo que la
información queda distribuida de forma desigual. En microestructuras, donde los bordes importan tanto como
el centro, los autores derivan reglas para el tamaño de kernel k, el stride s y el padding p (s < k,
k mod s = 0, p ≥ k − s) y usan {k, s, p} = {4, 2, 2}. Además, dan al latente z un tamaño espacial de 4 en
lugar de 1, de modo que la primera capa ya aprende salidas de kernel superpuestas. Como consecuencia, se
pueden generar volúmenes mayores que 64³ después del entrenamiento simplemente agrandando z. Por su parte, el crítico es una CNN 2D
simple de cinco convoluciones con paso (rebanada de 64 × 64 → un puntaje).

**Alcance y límites.** SliceGAN reproduce la estadística de la micrografía sin descriptores elegidos a
mano, entrena en pocas horas en una GPU y genera volúmenes en segundos. Fue validado contra datos 3D reales
de un electrodo de batería y luego aplicado a 87 materiales de la biblioteca MicroLib (Kench et al., 2022).
En este trabajo, se limita a casos isótropos (los materiales anisótropos necesitan dos o tres micrografías
perpendiculares y críticos separados) y un campo de visión representativo del material.

### 1.2 Antecedentes: Swin-T y SAM, y dónde entran en el pipeline

**Swin Transformer (Swin-T; Liu et al., 2021)** es un Vision Transformer jerárquico:

- la imagen se divide en parches de 4 × 4, cada uno embebido como un token;
- la autoatención se calcula dentro de ventanas locales de 7 × 7 tokens, que se desplazan entre capas
  consecutivas para que la información también fluya entre ventanas;
- cuatro etapas fusionan parches entre ellas, lo que da mapas de características de resolución
  decreciente (como una CNN);
- la posición se codifica con un sesgo de posición relativa dentro de cada ventana, no con embeddings
  absolutos, por lo que la red también funciona sobre rebanadas pequeñas de 64 × 64;
- Swin-T tiene 28 M de parámetros y está preentrenado en ImageNet-1k.

En este proyecto Swin-T reemplaza (M2, M3) o complementa (M4, M5) a la CNN de SliceGAN como crítico 2D,
aportando características preentrenadas y atención sobre toda la rebanada. Se sabe que los críticos
Transformer desestabilizan el entrenamiento de GANs (ViTGAN; Lee et al., 2022), y esa resultó ser la
dificultad central (Sección 5.2).

**Segment Anything (SAM; Kirillov et al., 2023)** es un modelo de segmentación guiado por prompts: un
codificador de imagen ViT, un codificador de prompts (puntos, cajas) y un decodificador de máscaras liviano,
entrenado con mil millones de máscaras. En modo automático, una grilla de puntos produce una máscara por
cada objeto, sin entrenamiento adicional (zero-shot).

**Cómo se incluye SAM.** SliceGAN necesita una micrografía segmentada (una etiqueta por fase) como dato de
entrenamiento. El enfoque baseline la obtiene con un umbral global de nivel de gris obtenido mediante el método de Otsu. Este enfoque funciona bien para casos bifásicos, aunque puede extenderse a multifásicos. En este trabajo, SAM se usa como
etapa previa alternativa para este paso de segmentación, y el resto del pipeline no cambia:

1. SAM ViT-B (`facebook/sam-vit-base`, sin ajuste fino) se ejecuta sobre parcelas superpuestas de 256 px de
   la micrografía con una grilla de 32 × 32 puntos por parcela (una sola pasada sobre la imagen completa
   pierde la mayoría de las islas pequeñas);
2. las máscaras superpuestas (IoU > 0,3) se fusionan;
3. cada grupo fusionado se etiqueta como inclusión o matriz según su nivel de gris medio (una división en
   dos clases de las medias de los grupos; los píxeles no cubiertos son matriz);
4. el mapa de fases resultante reemplaza al mapa de Otsu como imagen de entrenamiento del modelo M3.

![Segmentación con SAM](figures/sam_frontend.png)

### 1.3 Preguntas del proyecto:

- **PI1 — crítico ViT.** ¿Reemplazar el discriminador CNN de SliceGAN por un Swin Transformer (Swin-T)
  mejora las microestructuras generadas? (M1 vs M2, con una ablación M1 + DiffAug)
- **PI2 — segmentación con SAM.** ¿Segmentar la micrografía con SAM, en lugar de un umbral global, las
  mejora? (M2 vs M3)
- **PI3 — ViT como crítico adicional.** ¿Agregar un crítico Swin-T preentrenado junto a la CNN
  de SliceGAN, como en Vision-aided GAN (Kumari et al., 2022), ayuda, ya sea desde cero (M4) o como etapa
  de ajuste fino de un SliceGAN ya entrenado (M5, comparado contra M1 entrenado los mismos pasos extra)?

La calidad se mide con los descriptores estadísticos usados en la literatura de generación de
microestructuras (fracción de fase, correlación de dos puntos, camino lineal), ya que no existe una
referencia 3D.

## 2. Arquitectura general

![Pipeline](figures/pipeline.png)

**Datos.** En el caso isótropo, el set de entrenamiento es una sola micrografía 2D, y la misma imagen sirve de referencia para los tres planos (xy, xz, yz). Para materiales anisótropos se pueden dar hasta 3 imágenes, una por plano ortogonal, y cada dirección tiene su propio crítico.

En este trabajo, el caso principal es la entrada `000210` de MicroLib (biblioteca de micrografías DoITPoMS): una
micrografía óptica de islas oscuras en una matriz clara, de 437 × 800 px tras quitar la barra de escala,
0,687 µm/px. Se eligió al azar entre las entradas de dos fases de MicroLib con una diferencia de nivel de
gris entre fases ≥ 80 y se aceptó tras un control de isotropía (cociente de longitudes de correlación x/y
de 1,11). 

El segundo caso de estudio es una micrografía sintética (discos sin superposición, φ = 0,25, renderizada con desenfoque y
ruido, con máscara de referencia exacta). A diferencia del caso anterior, se conoce la máscara de fases exacta, de modo que tanto la segmentación (Otsu o SAM) como los descriptores de referencia (φ, S₂) se miden sin error de anotación.

**Segmentación (etapa previa).** M1, M2, M4 y M5 entrenan sobre un mapa de etiquetas de Otsu (φ = 0,232).
M3 entrena sobre un mapa de SAM: SAM ViT-B zero-shot sobre parcelas de 256 px, máscaras superpuestas
fusionadas (IoU > 0,3) y grupos cuyo gris medio se aparta del de la matriz etiquetados como inclusiones
(φ = 0,217).

**Generador.** El generador 3D de SliceGAN con resize-convolution, sin cambios en todos los modelos
(40,1 M de parámetros): latente gaussiano de 32 × 4 × 4 × 4 → volumen de 64³, softmax sobre las dos fases.

**Cortador y críticos.** Cada volumen generado se corta en sus 64 rebanadas en x, y y z. Las rebanadas y
recortes reales de 64 × 64 de la imagen de entrenamiento se puntúan con un crítico 2D:

| Modelo | Imagen de entrenamiento | Crítico 2D | Pérdida | Parámetros entrenables del crítico |
|--------|-------------------------|------------|---------|------------------------------------|
| M1 | Otsu | CNN de SliceGAN (5 convoluciones con paso) | WGAN-GP | 2,8 M |
| M2 | Otsu | Swin-T (ImageNet), etapas 3–4 + cabeza entrenables, DiffAug, lr 2e-5 | WGAN-GP | 26,3 M |
| M3 | SAM | mismo crítico que M2 | igual que M2 | igual que M2 |
| M4 | Otsu | CNN de SliceGAN **+** Swin-T congelado con cabezas por escala (ensamble) | CNN: WGAN-GP, Swin: hinge | 2,8 M + 0,18 M |
| M5 | Otsu | como M4, generador inicializado desde el mejor checkpoint de M1 | como M4 | como M4 |
| M1-ext | Otsu | como M1, generador inicializado desde el mejor checkpoint de M1 | WGAN-GP | 2,8 M |

**Métricas.** Para cada modelo, 128 volúmenes de 64³ (semillas 0–127) se comparan con la imagen 2D mediante
φ, S₂(r) y L(r), en total y por orientación de corte, con intervalos de confianza bootstrap (Sección 4).

## 3. Implementación técnica

**Stack.** Python 3.11, PyTorch 2.5.1 (CUDA 12.4), timm 1.0.11, Hugging Face `transformers` 4.46.3,
NumPy/SciPy/scikit-image, entorno Poetry (`setup.sh`), configuraciones YAML con herencia y superposición
por dataset, logging a archivo y consola, seguimiento de experimentos con MLflow, 88 tests de pytest.
Entrenado en una NVIDIA RTX A2000 (12 GB).

**Modelos preentrenados.**

- Swin-T: `timm/swin_tiny_patch4_window7_224.ms_in1k` en el Hugging Face Hub (el checkpoint ImageNet-1k
  de Microsoft, con los mismos pesos que `microsoft/swin-tiny-patch4-window7-224`). Se usa timm en lugar de
  `transformers` porque Swin-T solo funciona de forma nativa sobre rebanadas de 64 px si las ventanas de
  atención de las etapas 3–4 se reducen a 4 × 4 y 2 × 2 con un sesgo de posición relativa redimensionado,
  algo que timm soporta y `transformers` 4.46 no. El embedding de parches RGB se convierte a dos canales de
  fase, de modo que una rebanada one-hot se ve como una imagen de grises centrada (matriz −0,5,
  inclusión +0,5).
- SAM: `facebook/sam-vit-base` a través del pipeline de generación de máscaras de `transformers`, zero-shot.

**Integración de SliceGAN.** El repositorio original es un submódulo de git (`external/SliceGAN`, MIT) y no
se modifica.

-  `src/models/slicegan_wrapper.py` construye el generador y el crítico CNN con la fábrica de
redes original y las listas de capas exactas de `run_slicegan.py`, y reutiliza su penalización de
gradiente. 
- `src/models/train.py` es una bifurcación del bucle de entrenamiento original con el mismo
esquema WGAN-GP (Adam 1e-4, β = (0,9, 0,99), λ_GP = 10, 5 pasos del crítico por paso del generador) y la
regla de lotes de SliceGAN: el crítico ve las 64 rebanadas por eje de m_D = 1 volumen y el
paso del generador usa m_G = 2 m_D volúmenes. 
- Agregados: ramas de crítico (crítico único o ensamble CNN +
Swin), pérdida hinge, aumentación diferenciable de las entradas del crítico (DiffAug: traslación, cutout,
rotaciones de 90° y reflexiones), warm-start del generador y selección de checkpoints. 
- Las traslaciones (hasta ±8 px) y las posiciones del cutout no se alinean a propósito con la grilla de parches de
4 px de Swin-T: los recortes reales también se toman en desplazamientos arbitrarios, y un desplazamiento no
alineado cambia el contenido de cada parche, de modo que el crítico no puede aprovechar la posición de un rasgo
dentro de un parche.

**Selección de checkpoints y protocolo de evaluación.** Después de cada época (100 pasos del generador), el
generador produce 16 volúmenes con semillas reservadas (1000–1015), y se conserva la época con menor
S₂ MAE respecto de la imagen de entrenamiento del propio modelo. El checkpoint conservado se evalúa luego
sobre 128 semillas *distintas* (0–127), de modo que los números reportados nunca se miden sobre los
volúmenes usados para elegir el checkpoint. La misma regla se aplica a todos los modelos. El protocolo se
endureció dos veces durante el proyecto, porque un volumen de 64³ es una muestra pequeña de la
microestructura (≈ 30 islas; la fracción de fase varía ±0,05 de un volumen a otro): con 2 semillas de
selección y 4 de evaluación (corrida v5), el ranking de modelos se invirtió al reevaluar con 32 semillas,
y un checkpoint elegido como el mejor con 2 semillas podía ser tres veces peor que la última época sobre
semillas nuevas. Incluso con 16 semillas, elegir la mejor de 50 puntuaciones ruidosas es optimista (la
"maldición del ganador": el checkpoint seleccionado de M1 obtuvo S₂ MAE 0,0016 en sus semillas de selección
pero 0,020 en 128 semillas nuevas); evaluar sobre semillas independientes elimina ese sesgo de los números
reportados.

**Exportación de RVE.** `src/models/export_rve.py` escribe las microestructuras generadas para códigos de
homogeneización por FEM/FFT (VTK `.vti`, MetaImage `.mhd/.raw`, malla de vóxeles Abaqus `.inp` con
conjuntos de elementos por fase y conjuntos de nodos por cara, NumPy, TIFF, más metadatos JSON). Los
volúmenes mayores que 64³ se obtienen agrandando el latente de entrada; los RVE periódicos usan el mosaico
del latente de SliceGAN con un recorte de 2 vóxeles, que resultó ser el período correcto (el recorte de
1 vóxel del código original duplica una rebanada en la costura); la periodicidad se verifica comparando la
discrepancia entre caras opuestas con la discrepancia entre rebanadas interiores vecinas (cociente 1,0–2,0
frente a 8–13 sin mosaico).

**Módulos principales.**

- `src/data/make_dataset.py`: descarga, recorte, Otsu, recortes.
- `src/features/sam_segment.py`: segmentación con SAM.
- `src/features/descriptors.py`: φ, S₂, L.
- `src/models/discriminator_swin.py`: críticos Swin.
- `src/models/train.py`.
- `src/models/generate.py`: volúmenes + métricas.
- `src/visualization/visualize.py`: figuras, `metrics.csv`.
- `reports/viewer/`: un visor interactivo de
volúmenes.
- `notebooks/00_eda.ipynb`: notebook de EDA.

## 4. Evaluación

No existe una referencia 3D (no hay micro-CT), así que los volúmenes generados se comparan
estadísticamente con la imagen 2D de entrenamiento. En un material isótropo, cualquier sección plana de la
microestructura 3D tiene la misma estadística que la micrografía 2D, por lo que los descriptores 2D de la
imagen de entrenamiento deben coincidir con los descriptores promediados sobre las rebanadas del volumen
generado; esa es la justificación de las métricas siguientes (SliceGAN, Kench & Cooper 2021). Todas están
implementadas en `src/features/descriptors.py` y probadas en `tests/test_descriptors.py` contra casos con
solución analítica. La etiqueta 1 es la fase inclusión/poro e `I(x)` su función indicadora; las curvas se
evalúan para `r = 0 … 32` px sobre N = 128 volúmenes generados de 64³ por modelo (semillas distintas; la
consigna exige N ≥ 4).

- **Fracción de fase** `φ = ⟨I(x)⟩`, reportada como media ± desvío sobre los N volúmenes, junto con
  `|Δφ| = |φ̄_gen − φ_train|`.
- **Correlación de dos puntos** `S₂(r) = P[I(x) = 1, I(x + r) = 1]`, calculada por autocorrelación vía FFT
  (cada desplazamiento normalizado por su número de pares de píxeles válidos, sin suponer periodicidad) y
  promediada radialmente sobre los vectores con `round(|r|) = r`. `S₂(0) = φ` y `S₂(r) → φ²` para puntos
  no correlacionados. En un volumen, `S₂` se promedia sobre todas las rebanadas xy, xz e yz (192 rebanadas
  para 64³).
- **Camino lineal** `L(r)` (como en Micro3Diff, Lee & Yun 2024; Lyu & Ren 2024): probabilidad de que un
  segmento recto de `r + 1` píxeles quede completamente en la fase inclusión, promediada sobre ambas
  direcciones del plano (2D) o los tres ejes (3D). `L(0) = φ`; a diferencia de `S₂`, `L` es sensible a la
  conectividad a lo largo de líneas.
- **Errores**: MAE sobre `r` entre las curvas generadas y de entrenamiento (`S₂ MAE`, `L MAE`, usando la
  curva media sobre los N volúmenes) y la tasa de error de Micro3Diff
  `err(S₂) = mean|S₂_gen − S₂_train| / mean|S₂_train|` (análogamente `err(L)`).
- **Control de isotropía 3D**: `φ`, `S₂ MAE` y `L MAE` por orientación de corte (xy, xz, yz). Si un plano
  diverge, el volumen no es isótropo. Promediando sobre todas las rebanadas, `φ_xy = φ_xz = φ_yz = φ` por
  construcción, por lo que también se reporta la dispersión de la `φ` por rebanada a lo largo de cada eje.
- **Incertidumbre.** Los intervalos de confianza del 95 % para `|Δφ|`, `S₂ MAE` y `L MAE` provienen de un
  bootstrap sobre los 128 volúmenes (2000 remuestreos, cada uno recalculando la métrica exactamente como se
  reporta). Cada modelo se compara con la línea base M1 mediante la distribución bootstrap de la diferencia;
  un modelo se declara mejor o peor solo si ese intervalo excluye el cero.
- **Calidad típica durante el entrenamiento (sin selección).** Las puntuaciones reservadas por época son
  insesgadas individualmente; solo elegir su mínimo es optimista. La mediana y el rango intercuartílico del
  S₂ MAE reservado por época en la segunda mitad del entrenamiento, y el número de épocas colapsadas
  (φ reservada < 0,05), resumen cuán bueno y cuán estable es un modelo sin ninguna selección.
- **Referencia común.** Cada modelo se evalúa contra su propia imagen de entrenamiento y contra una
  referencia compartida por todos los modelos de un dataset, para que M3 (entrenado sobre el mapa de SAM)
  sea comparable con M2: el mapa de Otsu para MicroLib (no existe referencia exacta) y la máscara exacta
  para los datos sintéticos.
- **Calidad de la segmentación de SAM** (etapa previa de M3): IoU `= |P ∩ G| / |P ∪ G|` y Dice
  `= 2|P ∩ G| / (|P| + |G|)` del mapa de etiquetas `P` contra la referencia exacta `G` en la imagen
  sintética y, en MicroLib, contra una referencia curada: el punto medio de los dos niveles de gris de fase
  anotados por los autores de MicroLib (8 y 133) aplicado a la micrografía original.

Escalas de referencia: en el dataset sintético, la segmentación de Otsu frente a la máscara exacta da
`S₂ MAE = 0,0021`. Volúmenes aleatorios no correlacionados de 64³ con la `φ` correcta de MicroLib 000210
dan `S₂ MAE = 0,042` (`err = 0,42`) y `L MAE = 0,077` (`err = 0,89`), un piso que cualquier modelo útil
debe superar.

## 5. Resultados y ejemplos

### 5.1 Resultados principales (MicroLib 000210)

Protocolo entrenamiento / validación / prueba (Sección 3): el checkpoint de cada modelo se elige entre el
mejor por época, la última época y las instantáneas de entrenamiento sobre 128 semillas de validación, y
luego se evalúa sobre 128 volúmenes de prueba contra la referencia común (mapa de Otsu, φ = 0,232). Entre
corchetes: intervalos bootstrap del 95 %. "Típico" es la mediana sin selección del S₂ MAE reservado en la
segunda mitad del entrenamiento (16 semillas por época, por lo que está en otra escala que las columnas de
prueba).

| Modelo | Checkpoint | φ | \|Δφ\| | S₂ MAE | L MAE | S₂ típico | vs M1 |
|--------|------------|---|--------|--------|-------|-----------|-------|
| M1 CNN (SliceGAN) | última | 0,235 | 0,003 [0,000, 0,012] | 0,0029 [0,0008, 0,0087] | 0,0011 [0,0003, 0,0059] | **0,0051** | — |
| M1 + DiffAug | última | 0,229 | 0,003 [0,000, 0,012] | **0,0009** [0,0006, 0,0066] | 0,0014 [0,0010, 0,0059] | 0,0136 | sin diferencia clara |
| M2 Swin + DiffAug | mejor | 0,239 | 0,007 [0,000, 0,017] | 0,0077 [0,0017, 0,0141] | 0,0055 [0,0010, 0,0109] | 0,0132 | sin diferencia clara |
| M3 Swin + DiffAug, mapa SAM | última | 0,174 | 0,058 [0,050, 0,066] | 0,0328 [0,0281, 0,0371] | 0,0262 [0,0223, 0,0299] | 0,0143 | **peor** (todas las métricas) |
| M4 CNN + Swin congelado | mejor (= última) | **0,232** | **0,000** [0,000, 0,009] | 0,0014 [0,0006, 0,0067] | 0,0015 [0,0006, 0,0058] | 0,0096 | sin diferencia clara |
| M5 M1 + ajuste fino con Swin | última | 0,243 | 0,010 [0,001, 0,020] | 0,0097 [0,0033, 0,0161] | 0,0092 [0,0038, 0,0147] | 0,0072 | peor en L |
| M1 extendido | última | 0,238 | 0,006 [0,000, 0,015] | 0,0059 [0,0010, 0,0124] | 0,0059 [0,0019, 0,0114] | 0,0097 | sin diferencia clara |

Todos los errores están muy por debajo del piso de volúmenes aleatorios (S₂ MAE 0,042, L MAE 0,077). Los
modelos basados en CNN (M1, M1 + DiffAug, M4) forman el mejor grupo: sus estimaciones puntuales son las más
bajas y sus intervalos se superponen. En esta corrida M4, el ensamble CNN + Swin congelado, reproduce
exactamente la fracción de fase (0,232) y no necesitó selección de checkpoint (su mejor época es la última);
una segunda corrida (Sección 5.7) no repite ninguna de las dos propiedades. Ningún modelo
es significativamente mejor que M1 con 128 volúmenes de prueba; M3 es significativamente peor, porque su
imagen de entrenamiento (el mapa de SAM) tiene menor fracción de fase y distinta morfología que la
referencia de Otsu, y su último checkpoint subestima φ aún más.

![Curvas de descriptores](figures/microlib_000210_descriptors.png)

![Comparación cualitativa](figures/microlib_000210_qualitative.png)

Curvas de entrenamiento por modelo: `figures/microlib_000210_training.png`; todos los volúmenes pueden
explorarse en el visor interactivo (`reports/viewer/`).

### 5.2 Entrenar un crítico Swin: estudio de estabilización

Se sabe que los críticos Transformer vuelven inestable el entrenamiento de GANs (ViTGAN, Lee et al., 2022).
Con la configuración WGAN-GP de SliceGAN, el crítico Swin-T fue inestable en todas las configuraciones
probadas; la tabla resume las primeras 12 épocas en MicroLib (objetivo de φ reservada 0,232; S₂ MAE de las
mejores épocas de M1: 0,002–0,004).

| Corrida | Crítico Swin | Pérdida | DiffAug | Épocas 1–12 | Resultado |
|---------|--------------|---------|---------|-------------|-----------|
| v1 | etapas 3–4 entrenables, lr 1e-4, último checkpoint | WGAN-GP | no | realista en la época 10, vacío en la 15 | oscila, terminó colapsado (φ = 0,003) |
| v2 | etapas 3–4, lr 2e-5 | WGAN-GP | no | vacío en 4–5, S₂ MAE 0,010 en 9–11 | oscila, existen épocas al nivel de M1 |
| v2 repetida | misma configuración y semilla | WGAN-GP | no | buena en la época 2, luego φ ≈ 0 durante 10 épocas | variabilidad entre corridas: kernels de GPU no deterministas, amplificados por la dinámica de la GAN |
| v3 | v2 + ViTGAN: spectral norm mejorada, Adam β₁ = 0, EMA de G | WGAN-GP | no | φ 0,002–0,08 en todo momento | crítico demasiado fuerte |
| sonda etapa 4 | solo la etapa 4 entrenable | WGAN-GP | no | φ 0,08–0,29, mejor S₂ MAE 0,037 | sin colapso total, borroso |
| A | congelado, cabeza multiescala con pooling | WGAN-GP | sí | φ = 1,0 en las épocas 1–6, luego 0,51 → 0,31; mejor S₂ MAE 0,049 | falla: la penalización de gradiente no puede cumplirse a través de un backbone congelado |
| B | etapas 3–4, lr 2e-5 | WGAN-GP | sí | φ ≈ 0 en las épocas 1–5; desde la 7, φ 0,20–0,30, S₂ MAE 0,0091 / 0,018 / **0,0035** / 0,024 / 0,049 / **0,0035** | aprueba: épocas al nivel de M1, más consistente que v2 |
| C | congelado, cabezas multiescala por posición | WGAN-GP | sí | φ = 1,0 en las épocas 1–2, 0,89–0,95 hasta la 8, luego 0,38 → 0,08 → 0,11; mejor S₂ MAE 0,060 | falla como A: el problema es la GP, no el pooling |
| A-hinge | congelado, cabeza con pooling, spectral norm en la cabeza | hinge | sí | φ 0,57–1,0, en aumento; mejor S₂ MAE 0,26 | falla: sin colapso, pero φ sin control |
| C-hinge | congelado, cabezas por posición, spectral norm en la cabeza | hinge | sí | φ 0,31–0,56; mejor S₂ MAE 0,12 | falla solo; mejor que A-hinge, usado dentro de M4/M5 |

![Sondas del crítico Swin](figures/stabilization_probes.png)

Los registros por época de cada sonda están en `reports/probes/`.

**Decisión.** M2/M3 usan la sonda B (el único crítico solo-Swin que alcanza épocas al nivel de M1); M4/M5
usan las cabezas congeladas por posición de C-hinge dentro de un ensamble con el crítico CNN.

Tres hallazgos dieron forma a los modelos finales. Primero, un backbone preentrenado congelado no puede
entrenarse con WGAN-GP: la penalización de gradiente exige norma unitaria del gradiente respecto de la
entrada, que las capas Swin congeladas fijan y la pequeña cabeza solo puede reescalar, de modo que la
penalización domina (valores de 10–355 en lugar de ≈ 1) y el crítico no da una señal útil. Los críticos
con backbone congelado de la literatura (Projected GAN, Vision-aided GAN) usan pérdidas hinge o BCE.
Segundo, Vision-aided GAN reporta que los críticos preentrenados usados *solos* divergen y solo ayudan en
un ensamble con el discriminador original, lo que motiva M4 y M5. Tercero, con pérdida hinge el crítico
Swin congelado ya no colapsa, pero no controla por sí solo la fracción de fase (φ deriva a 0,3–0,8); una
causa plausible es la LayerNorm por token de Swin, que elimina buena parte de la información de intensidad
absoluta de la fase. El crítico CNN de M4/M5 aporta esa restricción. Por último, DiffAug es lo que hizo
funcionar al crítico Swin entrenable (B vs v2), en línea con la literatura de GANs con pocos datos (Zhao et
al., 2020; Karras et al., 2020): de otro modo, el crítico sobreajusta las pocas centenas de vistas
distintas de una única micrografía.

**Diagnóstico con sonda lineal** (Vision-aided GAN, Sec. 3.2): una regresión logística sobre
características Swin congeladas y con pooling separa recortes reales de 64 × 64 de rebanadas del mejor
generador de M1 con 73 % de exactitud reservada a 64 px de entrada (solo la etapa 2: 75 %), 83 % a 128 px
y 83 % a 224 px; los volúmenes aleatorios con la φ correcta se separan al 100 %. Las características
congeladas contienen entonces una señal real-vs-falso utilizable, más fuerte a la resolución 8 × 8 de la
etapa 2, y una entrada más grande podría sumar unos 10 puntos.

### 5.3 Ajuste fino con un crítico Swin (M5 vs M1 extendido)

Ambas corridas parten del checkpoint seleccionado de M1 y entrenan 20 épocas más con críticos nuevos;
solo difieren en los críticos (CNN vs CNN + Swin congelado con cabezas por posición). Ninguna mejora a M1:
en los volúmenes de prueba, M1 extendido alcanza S₂ MAE 0,0059 y M5 0,0097 (M1: 0,0029), y sus mejores
puntuaciones reservadas por época se alcanzaron en la primera época, es decir, en el punto de partida. Una
vez que SliceGAN convergió, agregar el crítico Swin como etapa de ajuste fino (la receta de Vision-aided
GAN) no ayuda en este caso. Una versión anterior de esta comparación (corrida v5, selección con 2 semillas)
sugería lo contrario; ese resultado era ruido de selección y desapareció con más semillas (Sección 3). En el
dataset sintético (Sección 5.6) ambas corridas mejoran a M1 y M5 tiene el menor S₂ MAE de todos los modelos
(0,0013 frente a 0,0020 de M1 extendido), pero la diferencia entre ellas no es significativa (intervalo del
95 % de M5 − M1 extendido: −0,0028 a +0,0023): la mejora proviene del entrenamiento adicional, no del
crítico Swin.

![M5 vs M1 extendido (trazas de selección de la corrida v5)](figures/m5_vs_m1_extended.png)

### 5.4 Segmentación con SAM

| Dataset | φ SAM | φ Otsu | Referencia (φ) | IoU / Dice SAM | IoU / Dice Otsu |
|---------|-------|--------|----------------|----------------|-----------------|
| sintético | 0,290 | 0,254 | máscara exacta (0,250) | 0,864 / 0,927 | 0,913 / 0,955 |
| MicroLib 000210 | 0,217 | 0,232 | umbral anotado de MicroLib (0,219) | 0,845 / 0,916 | 0,941 / 0,970 |

SAM zero-shot necesita teselado en esta imagen (una sola pasada con 16 puntos por lado encuentra
φ = 0,07), e incluso con teselado tiene menor IoU/Dice que Otsu frente a ambas referencias. El notebook
exploratorio (`notebooks/00_eda.ipynb`) muestra de dónde vienen los errores:

- **Sintético (referencia exacta):** el error de SAM es una dilatación sistemática de un píxel. Todos sus
  píxeles erróneos están a menos de 2 px de una interfaz real y casi todos son falsas inclusiones, lo que
  suma una sobreestimación de φ del 16 %. Los errores de Otsu están en las mismas interfaces pero se
  compensan, por lo que su φ es casi exacta.
- **MicroLib:** ambos mapas coinciden en el 96 % de los píxeles. Dos tercios de la discrepancia son el halo
  gris alrededor de cada partícula oscura, que Otsu (umbral 79, por encima del punto medio curado 70)
  etiqueta como inclusión. El resto son algunas regiones de gris intermedio que SAM toma como objetos
  completos. En consecuencia, la φ, S₂ y L de SAM están *más cerca* de la referencia curada que las de Otsu
  (S₂ MAE 0,0009 vs 0,0085). Aun así, Otsu tiene mayor IoU, porque la referencia es en sí misma un umbral
  global y favorece estructuralmente a Otsu.

Por lo tanto, la elección de la etapa de segmentación desplaza el *objetivo* de la GAN en unos 0,009 de
S₂ MAE, más que la diferencia entre los mejores modelos de MicroLib (0,001–0,003). Por eso cada modelo se
evalúa contra su propio mapa de entrenamiento y, por separado, contra una referencia común (§4).

El análisis exploratorio también muestra que la micrografía tiene un leve bandeado en x: la longitud de
correlación es de unos 23 px en x frente a 19 px en y (S₂ dentro del 5 % de su meseta). SliceGAN supone la
misma estadística en los tres planos, y la aumentación D4 de los recortes simetriza x e y a propósito, de
modo que los generadores aprenden una versión isótropa de la estructura. Las métricas con promedio radial
son insensibles a esto, pero ningún modelo reproduce aquí la dirección del bandeado.

**Sonda de etiquetado: una imagen donde un umbral global falla.** Las dos imágenes de entrenamiento
anteriores favorecen a un umbral global, por lo que no pueden mostrar lo que SAM aporta. Una tercera imagen
sintética, `synthetic_sam` (`configs/data/synthetic_sam.yaml`), conserva los mismos discos sin superposición
(φ = 0,250, disposición propia) pero agrega una rampa de iluminación en x mayor que el contraste entre fases,
ruido por fase que hace que los dos histogramas de grises se superpongan y un borde oscuro delgado alrededor de
cada disco, de modo que cada objeto sigue siendo visible localmente. Es una sonda de etiquetado, no un conjunto
de entrenamiento: Otsu y SAM se evalúan contra su máscara limpia (`src/features/sam_probe.py`,
`reports/sam_probe.json`) y no se entrena ninguna GAN con ella.

| Etiquetado (vs máscara limpia, φ 0,250) | IoU | IoU mitad oscura / mitad clara | φ |
|-----------------------------------------|-----|--------------------------------|---|
| Otsu (umbral global) | 0,430 | 0,691 / 0,329 | 0,467 |
| SAM, regla de etiquetado global (la usada en M3) | 0,455 | 0,019 / 0,909 | 0,123 |
| SAM, etiquetado por contraste local | **0,931** | 0,932 / 0,930 | **0,257** |

![Sonda de etiquetado con SAM](figures/sam_probe.png)

Otsu corta a través de la rampa: pierde inclusiones en el lado oscuro y etiqueta como inclusión la mayor parte
de la matriz del lado claro (φ 0,467). Las máscaras de SAM encuentran todos los discos, pero la regla del
pipeline que asigna una fase a cada grupo de máscaras es a su vez global (un único corte de los grises medios de
los grupos), así que todos los discos de la mitad oscura, más oscuros que la matriz de la mitad clara, quedan
etiquetados como matriz. Evaluar cada grupo contra un anillo de 3 px justo por fuera de él (`sam.classify:
local`, opcional; M3 usó la regla global) cancela la rampa: IoU 0,931 y φ a 0,007 del valor real, por encima de
Otsu en la imagen sintética limpia (0,913). Las máscaras de instancia de SAM no son, por lo tanto, el punto
débil; lo es el paso que convierte máscaras en fases.

### 5.5 ¿Qué miran los críticos?

Saliencia SmoothGrad (Smilkov et al., 2017): el gradiente del puntaje de cada crítico entrenado respecto
de su entrada one-hot, promediado sobre 16 copias con ruido, en 32 recortes reales y 32 rebanadas
generadas (`src/visualization/critic_saliency.py`, valores en `reports/critic_saliency.json`).

![Saliencia de los críticos](figures/critic_saliency.png)

| Crítico | Saliencia sobre bordes de fase (la banda de borde = 12 % de los píxeles) | P(puntaje real > puntaje generado) |
|---------|--------------------------------------------------------------------------|------------------------------------|
| M1 CNN de SliceGAN | 25 % (2,1 × su área) | 0,75 |
| M2 Swin-T, etapas 3–4 entrenadas | 22 % (1,8 ×) | 0,79 |
| M4 Swin-T congelado, cabezas por posición | 16 % (1,3 ×) | 0,67 |

El crítico CNN se concentra en los contornos de las inclusiones: juzga interfaces. El Swin-T entrenado
también favorece los bordes, pero de forma más difusa, y su saliencia muestra la grilla de parches de
4 × 4 del transformer. Las cabezas del Swin-T congelado reparten su atención sobre toda la rebanada:
responden a la textura y la disposición global más que a los bordes locales, y son las que peor separan,
por sí solas, rebanadas reales de generadas. Los dos críticos de M4 usan, por lo tanto, señales
complementarias (CNN: interfaces; Swin congelado: textura global), una razón plausible de que el ensamble
entrene de forma confiable, mientras que el Swin congelado solo no puede controlar la fracción de fase
(Sección 5.2).

### 5.6 Dataset sintético: comparación contra una referencia exacta

Los siete modelos se reentrenaron sobre la micrografía sintética con las mismas configuraciones y el mismo
protocolo. Aquí la referencia común es la máscara exacta (φ = 0,250), por lo que las puntuaciones miden
cuán cerca llega cada modelo a la estructura *verdadera*, incluido el error de su etapa de segmentación. Como
escala, el propio mapa de Otsu obtiene S₂ MAE 0,0021 frente a esta máscara.

| Modelo | Checkpoint | φ | \|Δφ\| | S₂ MAE | L MAE | S₂ típico | vs M1 |
|--------|------------|---|--------|--------|-------|-----------|-------|
| M1 CNN (SliceGAN) | instantánea ép. 30 | 0,240 | 0,010 [0,003, 0,017] | 0,0040 [0,0023, 0,0080] | 0,0052 [0,0042, 0,0066] | 0,0084 | — |
| M1 + DiffAug | instantánea ép. 20 | 0,240 | 0,010 [0,004, 0,017] | 0,0045 [0,0028, 0,0080] | 0,0056 [0,0043, 0,0072] | 0,0074 | sin diferencia clara |
| M2 Swin + DiffAug | mejor (ép. 29) | 0,246 | 0,004 [0,000, 0,013] | 0,0093 [0,0075, 0,0140] | 0,0216 [0,0189, 0,0260] | 0,0150 | **peor** (S₂, L) |
| M3 Swin + DiffAug, mapa SAM | mejor (ép. 15) | 0,289 | 0,039 [0,032, 0,045] | 0,0298 [0,0252, 0,0344] | 0,0317 [0,0282, 0,0350] | 0,0106 | **peor** (todas las métricas) |
| M4 CNN + Swin congelado | instantánea ép. 35 | 0,253 | 0,002 [0,000, 0,008] | 0,0039 [0,0026, 0,0075] | 0,0052 [0,0046, 0,0067] | **0,0032** | sin diferencia clara |
| M5 M1 + ajuste fino con Swin | mejor (ép. 8 de 20) | 0,248 | 0,002 [0,000, 0,008] | **0,0013** [0,0008, 0,0046] | 0,0053 [0,0049, 0,0061] | 0,0050 | sin diferencia clara |
| M1 extendido | mejor (ép. 8 de 20) | 0,249 | 0,002 [0,000, 0,008] | 0,0020 [0,0018, 0,0045] | 0,0056 [0,0051, 0,0065] | 0,0050 | sin diferencia clara |

![Curvas de descriptores, dataset sintético](figures/synthetic_descriptors.png)

![Comparación cualitativa, dataset sintético](figures/synthetic_qualitative.png)

Los resultados sintéticos confirman las conclusiones de MicroLib (una corrida de entrenamiento por modelo; la
Sección 5.7 muestra cuánto puede diferir una segunda corrida):

- **Swin como único crítico (PI1).** M2 es significativamente peor que M1 en S₂ y L, y también peor que la
  ablación justa M1 + DiffAug (diferencia en S₂ +0,0048 [+0,0006, +0,0098], en L +0,0161 [+0,0129, +0,0206]).
  Sus volúmenes pierden la morfología de discos/esferas (inclusiones alargadas y fusionadas en el panel
  cualitativo). Además son **anisótropos**: las rebanadas xy coinciden bien (S₂ MAE 0,0054), las xz e yz no
  (0,0134 y 0,0101). La primera corrida de M2 en MicroLib muestra el mismo patrón (xy 0,0027 frente a xz 0,0114
  e yz 0,0098), pero su segunda corrida en MicroLib es isótropa (Sección 5.7), de modo que la anisotropía es un
  modo de falla de algunas corridas con crítico Swin, no una propiedad sistemática; los modelos basados en CNN
  son isótropos en todas las corridas.
- **Segmentación con SAM (PI2).** M3 reproduce fielmente su propio mapa de entrenamiento (S₂ MAE 0,0029
  frente al mapa de SAM, φ 0,289 frente a 0,290), así que la GAN funciona; el error proviene por completo de
  la etapa de segmentación, cuya dilatación de un píxel (Sección 5.4) el generador aprende como un rasgo real.
  Frente a la estructura verdadera, M3 es el peor modelo y es significativamente peor que M2, su equivalente
  entrenado con Otsu, en todas las métricas.
- **Swin como crítico adicional (PI3).** M4 está a la par de M1 en todas las métricas, reproduce φ con un
  error de 0,002, es significativamente mejor que M2 (S₂ −0,0054 [−0,0102, −0,0015], L −0,0163 [−0,0207,
  −0,0134]) y fue el modelo más estable durante el entrenamiento en esta corrida: su mediana sin selección del S₂ MAE reservado
  (0,0032, rango intercuartílico 0,0023–0,0039) es la más baja de los siete, menos de la mitad de la de M1
  (0,0084). Como etapa de ajuste fino, M5 da el menor S₂ MAE de prueba (0,0013, por debajo del 0,0021 de la
  propia segmentación de Otsu, porque el generador suaviza el ruido de la segmentación), pero no se distingue
  de M1 extendido.

### 5.7 Corridas repetidas y un crítico Swin a 128 px (MicroLib)

Dos experimentos adicionales ponen a prueba la robustez de las conclusiones. Primero, M1, M2 y M4 se
entrenaron una segunda vez con otra semilla de entrenamiento (43 en lugar de 42) y la misma configuración;
las semillas de validación y de prueba son las mismas que antes. Segundo, M4 se entrenó con su rama Swin a
128 px: las rebanadas se sobremuestrean de 64 a 128 px antes del Swin-T congelado, de modo que sus ventanas
de atención se mantienen en 7 × 7 en las primeras etapas, motivado por la sonda lineal (73 % a 64 px frente a
83 % a 128 px). Esta corrida necesitó una pasada hacia atrás del generador por cada orientación de corte en
lugar de una para las tres (el mismo gradiente, un tercio de las activaciones del crítico en memoria); de lo
contrario se quedaba sin memoria de GPU. Valores: `reports/metrics_seeds_microlib.csv`.

| Modelo | Corrida | Checkpoint | φ | \|Δφ\| | S₂ MAE | L MAE | S₂ típico | S₂ MAE xy / xz / yz |
|--------|---------|------------|---|--------|--------|-------|-----------|---------------------|
| M1 CNN | 1 | última | 0,235 | 0,003 [0,000, 0,012] | 0,0029 [0,0008, 0,0088] | 0,0011 [0,0003, 0,0059] | 0,0051 | 0,0047 / 0,0008 / 0,0040 |
| M1 CNN | 2 | instantánea ép. 35 | 0,220 | 0,012 [0,003, 0,020] | 0,0055 [0,0012, 0,0112] | 0,0045 [0,0015, 0,0093] | 0,0127 | 0,0074 / 0,0048 / 0,0042 |
| M2 Swin + DiffAug | 1 | mejor | 0,239 | 0,007 [0,000, 0,017] | 0,0077 [0,0019, 0,0140] | 0,0055 [0,0009, 0,0108] | 0,0132 | 0,0027 / 0,0114 / 0,0098 |
| M2 Swin + DiffAug | 2 | última | 0,229 | 0,003 [0,000, 0,012] | 0,0017 [0,0007, 0,0072] | 0,0010 [0,0004, 0,0060] | 0,0067 | 0,0051 / 0,0043 / 0,0049 |
| M4 CNN + Swin congelado | 1 | mejor (= última) | 0,232 | 0,000 [0,000, 0,009] | 0,0014 [0,0006, 0,0068] | 0,0015 [0,0006, 0,0059] | 0,0096 | 0,0019 / 0,0075 / 0,0014 |
| M4 CNN + Swin congelado | 2 | instantánea ép. 35 | 0,222 | 0,010 [0,002, 0,018] | 0,0044 [0,0015, 0,0096] | 0,0043 [0,0012, 0,0087] | 0,0104 | 0,0065 / 0,0027 / 0,0060 |
| M4, Swin a 128 px | 1 | instantánea ép. 40 | 0,240 | 0,008 [0,000, 0,016] | 0,0042 [0,0013, 0,0100] | 0,0022 [0,0007, 0,0070] | 0,0084 | 0,0065 / 0,0014 / 0,0056 |

- **La variabilidad entre corridas es tan grande como las diferencias entre modelos.** Para cada modelo, las
  dos corridas no son significativamente distintas, pero sus estimaciones puntuales difieren en factores de
  2 a 4 (M2: S₂ MAE 0,0077 frente a 0,0017). El orden de M1, M2 y M4 cambia entre el primer y el segundo
  conjunto de corridas, y dentro del segundo conjunto ningún par es significativamente distinto. En MicroLib,
  por lo tanto, los tres críticos alcanzan la misma calidad; un ranking basado en una sola corrida por modelo
  no sería confiable.
- **Propiedades que no se repitieron.** La φ exacta de M4 y su "mejor época = última", y la anisotropía de
  M2, fueron rasgos de corridas individuales. La puntuación típica sin selección de M4 es la más reproducible
  entre corridas (0,0096 y 0,0104, frente a 0,0051 y 0,0127 de M1), lo que es evidencia débil de un
  entrenamiento más predecible, no de uno mejor.
- **Swin a 128 px.** Sin diferencia clara respecto de M4 a 64 px (diferencia en S₂ +0,0018 [−0,0037,
  +0,0086]), M1 o M2. Su checkpoint seleccionado obtuvo 0,0010 en las semillas de validación pero 0,0042 en
  las de prueba, y sus últimas épocas oscilan (última época 0,027). La mejor separación real-vs-generado de
  las características congeladas a 128 px no se tradujo en mejores volúmenes, por lo que no se corrió la
  repetición de esta variante en el dataset sintético.

### 5.8 Tamaño de los modelos y costo computacional

Medido en la RTX A2000 (12 GB) usada en todas las corridas (`src/visualization/compute_cost.py`,
`reports/compute_cost.json`). Los FLOPs cuentan 2 por multiplicación-suma sobre una entrada; la estimación de
entrenamiento por paso del generador sigue el esquema de SliceGAN (el crítico ve 1464 rebanadas y el generador
7 volúmenes por paso, hacia adelante más hacia atrás ≈ 3× la pasada hacia adelante; sin contar la penalización
de gradiente ni DiffAug).

| Componente | Parámetros (entrenables) | GFLOPs por pasada hacia adelante | Uso |
|------------|--------------------------|----------------------------------|-----|
| Generador 3D (todos los modelos) | 40.1 M | 27.1 por volumen de 64³ (145 por 128³) | entrenamiento e inferencia |
| Crítico CNN de SliceGAN | 2.76 M | 0.21 por rebanada de 64 × 64 | entrenamiento (M1, M4, M5) |
| Crítico Swin-T, etapas 3–4 entrenadas | 27.5 M (26.3 M) | 0.85 por rebanada | entrenamiento (M2, M3) |
| Swin-T congelado + cabezas por posición | 27.7 M (0.18 M) | 0.85 por rebanada (4.1 a 128 px) | entrenamiento (M4, M5) |
| SAM ViT-B (segmentación) | 93.7 M (no se entrena) | — | preparación de datos de M3, una vez |

| Modelo | TFLOPs estimados por paso del generador | s medidos por paso del generador | Tiempo de entrenamiento (MicroLib) |
|--------|-----------------------------------------|----------------------------------|------------------------------------|
| M1 CNN | 1.5 | 0,31 | 27 min |
| M1 + DiffAug | 1.5 | 0,40 | 36 min |
| M2 / M3 crítico Swin | 4.3 | 1,93 | 163 / 162 min |
| M4 CNN + Swin congelado | 5.2 | 1,16 | 98 min |
| M4, Swin a 128 px | 19.5 | 3,61 | 303 min |
| M5 / M1 extendido (20 épocas desde M1) | 5.2 / 1.5 | 1,16 / 0,31 | 40 / 11 min (+ M1) |

- **La inferencia cuesta lo mismo para todos los modelos.** Los críticos solo se usan para entrenar; todos los
  modelos comparten el mismo generador de 40.1 M de parámetros, que produce un
  volumen de 64³ en 14 ms en la GPU (0.25 s en CPU) y uno de 128³ en
  67 ms. Los críticos Vision Transformer no agregan costo de despliegue.
- **El costo de entrenamiento lo fija el crítico.** Una rebanada en Swin-T cuesta unas 4× los FLOPs de una en la
  CNN, y el crítico se evalúa sobre unas 1500 rebanadas por paso del generador, así que domina el entrenamiento.
  M2 / M3 tardan 6× más que M1, más que la relación de FLOPs (2,9×), porque los gradientes atraviesan 26 M de
  pesos del crítico y la doble retropropagación de la penalización de gradiente pasa por la atención. M4 hace más
  FLOPs que M2 pero entrena más rápido (3,7× M1): su backbone Swin está congelado (sin gradientes de pesos) y su
  rama Swin usa pérdida hinge sin penalización de gradiente. Llevar el Swin a 128 px multiplica sus FLOPs por 4,8
  y el tiempo de entrenamiento por 3, sin mejora (Sección 5.7); además necesitó una pasada hacia atrás del
  generador por orientación de corte para entrar en 12 GB.
- **SAM es un costo único.** Segmentar la imagen de MicroLib lleva unos 35 s
  (93,7 M de parámetros, teselas de 256 px), despreciable frente al entrenamiento, y SAM no se usa al generar.
- **Costo frente a beneficio.** Como ninguna variante Swin supera al crítico CNN (Secciones 5.1, 5.6 y 5.7), el
  crítico de 2,8 M de parámetros de SliceGAN es también la mejor opción por hora de GPU. Las corridas finales de
  este informe sumaron 28 horas de GPU; las corridas exploratorias (v1–v5, sondas) no están incluidas.

## 6. Conclusiones y trabajo futuro

**Respuestas a las preguntas del proyecto** (dos datasets, 128 volúmenes de prueba por modelo,
intervalos bootstrap; dos corridas de entrenamiento de M1, M2 y M4 en MicroLib):

- **PI1 — Swin-T en lugar del crítico CNN: no.** Entrenado con la configuración WGAN-GP de SliceGAN, un
  crítico Swin-T fue inestable en todas las configuraciones hasta que se agregó DiffAug. Con DiffAug alcanza
  la calidad de la CNN, pero no más: en MicroLib sus dos corridas encierran a las de la CNN (sin diferencia
  significativa en ninguna), y la única corrida sintética es significativamente peor en S₂ y L (también
  frente a la ablación M1 + DiffAug) y menos isótropa. Los estabilizadores habituales de ViT-GAN
  (menor tasa de aprendizaje, spectral norm mejorada, Adam β₁ = 0, EMA del generador, menor parte entrenable)
  no eliminaron la inestabilidad; un crítico Swin congelado directamente no puede entrenarse con WGAN-GP (la
  penalización de gradiente domina) y, con pérdida hinge, no controla φ.
- **PI2 — SAM en lugar de un umbral global: no, para este tipo de imagen.** En micrografías de dos fases y
  alto contraste, SAM zero-shot es una segmentación peor que Otsu frente a una referencia exacta (una
  dilatación sistemática de un píxel, φ +16 %). La GAN reproduce fielmente el mapa de SAM, así que M3 hereda
  ese sesgo y es significativamente peor que M2 en ambos datasets. La etapa de segmentación desplaza el
  objetivo de la GAN más que cualquier cambio de crítico: la calidad de la segmentación importa más que la
  arquitectura del crítico. Una sonda de etiquetado donde un umbral global falla (una rampa de iluminación,
  Sección 5.4) muestra la otra cara: las máscaras de SAM siguen encontrando todas las partículas y, con una regla
  de etiquetado por contraste local, SAM alcanza IoU 0,93 frente a 0,43 de Otsu. En este pipeline, la debilidad
  de SAM zero-shot es el paso que asigna fases a sus máscaras, no las máscaras.
- **PI3 — Swin-T como crítico adicional: iguala al baseline, sin una mejora medible.** El ensamble CNN +
  Swin congelado (M4, al estilo Vision-aided GAN) está estadísticamente a la par de SliceGAN en las tres
  comparaciones (dos corridas en MicroLib, una en el sintético), es significativamente mejor que el crítico
  solo-Swin en los datos sintéticos, y la calidad de su entrenamiento es la más reproducible entre corridas.
  Su primera corrida parecía mejor (φ exacta, mejor época = última, menor puntuación típica en el
  sintético), pero la segunda corrida en MicroLib no lo repitió. Los mapas de saliencia muestran que ambos
  críticos usan señales complementarias: la CNN juzga interfaces y el Swin congelado, la textura global.
  Llevar la rama Swin a 128 px no ayudó. Usado como etapa de ajuste fino de un SliceGAN ya
  convergido (M5), el crítico Swin no mejora respecto de entrenar solo la CNN los mismos pasos adicionales.

**En conjunto.** El pequeño crítico CNN de SliceGAN es difícil de superar en microestructuras de dos fases de
64³. Un crítico Swin-T, solo (con DiffAug) o junto a la CNN, alcanza la misma calidad pero no una mejor; solo
es más difícil de entrenar y a veces produce volúmenes anisótropos, mientras que junto a la CNN (una cabeza de
0,18 M de parámetros sobre un backbone congelado) entrena de forma tan confiable como la línea base. La
segunda lección es metodológica y probablemente la más transferible: un volumen de 64³ es una muestra ruidosa
(φ ±0,05), el mismo modelo entrenado dos veces puede diferir en un factor de 2 a 4, y los rankings de modelos
obtenidos con pocas semillas o una sola corrida se invirtieron al reevaluar. Hicieron falta semillas separadas
de selección, validación y prueba, intervalos bootstrap y corridas repetidas para llegar a conclusiones que se
sostienen.

**Limitaciones.** Dos corridas de entrenamiento para M1, M2 y M4 en MicroLib y una para el resto de los
modelos y para el dataset sintético, cuando la Sección 5.7 muestra que la variabilidad entre corridas es tan
grande como las diferencias entre modelos; rebanadas de 64 px, que
obligan a reducir las ventanas de Swin-T a 4 × 4 y 2 × 2 y usan el backbone lejos de su resolución de
preentrenamiento de 224 px; una sola micrografía real, de alto contraste y dos fases, donde un umbral global
ya es casi óptimo.

SliceGAN es el baseline en la literatura; trabajos de 2024 (Micro3Diff, Lee & Yun 2024; DDPM-GAN, Phan et al. 2024) mejoran
descriptores y estabilidad con modelos de difusión, pero están fuera del alcance de este trabajo. La contribución aquí es una evaluación controlada de críticos Swin y de SAM como
etapa de segmentación.

**Trabajo en curso: etiquetado por contraste local en las imágenes de entrenamiento.** La regla de
etiquetado que gana la sonda (Sección 5.4) se está aplicando a las imágenes sintética y de MicroLib existentes,
escribiendo en carpetas separadas para que los mapas con los que se entrenó M3 no cambien, y se evalúa contra
las mismas referencias que Otsu y la regla global. En MicroLib podría eliminar el halo gris que separa el mapa
de SAM del de Otsu; si los nuevos mapas son mejores, M3 se reentrenará con ellos. Sus resultados no forman parte
de este informe.

**Trabajo futuro.** Cinco o más corridas de entrenamiento por modelo, que la dispersión entre corridas exige
antes de cualquier ranking; micrografías más difíciles (bajo contraste, texturas,
tres fases), donde la segmentación por objetos de SAM puede rendir, posiblemente con prompts de puntos o un
decodificador de máscaras ajustado; y generadores basados en difusión.

## 7. Planificación

| Tarea | Responsable | Estado |
|-------|-------------|--------|
| Esqueleto del repositorio, configuraciones, Makefile, entorno Poetry, tests, seguimiento con MLflow | Braian Desia | Hecho |
| Datasets: sintético + MicroLib 000210; notebook exploratorio | Braian Desia | Hecho |
| Descriptores φ / S₂ / L, intervalos bootstrap + tests | Braian Desia | Hecho |
| M1 línea base SliceGAN (+ ablación DiffAug, M1 extendido) | Braian Desia | Hecho |
| M2 crítico Swin-T + estudio de estabilización (v1–v3, sondas A/B/C, variantes hinge) | Braian Desia | Hecho |
| M3 segmentación con SAM | Braian Desia | Hecho |
| M4 / M5 extensiones Vision-aided | Braian Desia | Hecho |
| Corridas repetidas (segundas semillas de M1, M2, M4) y crítico Swin a 128 px | Braian Desia | Hecho |
| Protocolo de evaluación (semillas de entrenamiento / validación / prueba), figuras, saliencia | Braian Desia | Hecho |
| Visor interactivo de volúmenes y explorador en Streamlit | Braian Desia | Hecho |
| Exportación de RVE para códigos FEM / FFT | Braian Desia | Hecho |
| Sonda de etiquetado con SAM (`synthetic_sam`) y etiquetado por contraste local | Braian Desia | Hecho |
| Etiquetado por contraste local en las imágenes sintética y de MicroLib (trabajo en curso, Sección 6) | Braian Desia | En curso |
| Informe (inglés + español, PDF) | Braian Desia | Hecho (5 de octubre) |
| Presentación, 15 min | Braian Desia | 12 de octubre |

## Referencias

- S. Kench, S. J. Cooper. Generating three-dimensional structures from a two-dimensional slice with
  generative adversarial network-based dimensionality expansion. *Nature Machine Intelligence*, 2021.
- S. Kench et al. MicroLib: A library of 3D microstructures generated from 2D micrographs using SliceGAN.
  *Scientific Data*, 2022.
- Z. Liu et al. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. *ICCV*, 2021.
- A. Kirillov et al. Segment Anything. *ICCV*, 2023.
- I. Gulrajani et al. Improved Training of Wasserstein GANs. *NeurIPS*, 2017.
- K. Lee et al. ViTGAN: Training GANs with Vision Transformers. *ICLR*, 2022.
- N. Kumari, R. Zhang, E. Shechtman, J.-Y. Zhu. Ensembling Off-the-shelf Models for GAN Training.
  *CVPR*, 2022.
- A. Sauer et al. Projected GANs Converge Faster. *NeurIPS*, 2021.
- S. Zhao et al. Differentiable Augmentation for Data-Efficient GAN Training. *NeurIPS*, 2020.
- D. Smilkov et al. SmoothGrad: removing noise by adding noise. *arXiv:1706.03825*, 2017.
- T. Karras et al. Training Generative Adversarial Networks with Limited Data. *NeurIPS*, 2020.
- K.-H. Lee, G. J. Yun. Multi-plane denoising diffusion-based dimensionality expansion for 2D-to-3D
  reconstruction of microstructures with harmonized sampling (Micro3Diff). *npj Computational
  Materials*, 2024. doi:10.1038/s41524-024-01280-z
- J. Phan, M. Sarmad, L. Ruspini, G. Kiss, F. Lindseth. Generating 3D images of material microstructures
  from a single 2D image: a denoising diffusion approach. *Scientific Reports*, 2024.
- X. Lyu, X. Ren. Microstructure reconstruction of 2D/3D random materials via diffusion-based deep
  generative models. *Scientific Reports*, 2024.
