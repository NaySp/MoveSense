# Informe de Progreso – Proyecto de Inteligencia Artificial
**Título del Proyecto**: Sistema de Anotación y Clasificación de Actividades Humanas en Video  
**Fecha**: [03/05/2025]  
**Autores**: Nayeli Suarez, Isabela Ocampo y Juan Caviedez

**Curso/Asignatura**: Inteligencia Artificial 2025

---

## Preguntas de interés

Las principales preguntas que guían este proyecto son:

1. ¿Es posible clasificar actividades humanas (como caminar, sentarse, estirarse, etc.) mediante el seguimiento de articulaciones en video?
2. ¿Qué características derivadas del movimiento corporal son más relevantes para diferenciar entre distintas actividades físicas?
3. ¿Se puede realizar una inferencia en tiempo real con precisión aceptable utilizando herramientas accesibles como MediaPipe y clasificadores tradicionales?
4. ¿Cómo podemos construir una interfaz accesible para que cualquier usuario pueda cargar un video y obtener la predicción de actividades?

---

## Tipo de problema

Este proyecto corresponde a un problema de **clasificación supervisada**, específicamente:

- **Dominio**: Visión por computador y aprendizaje automático.
- **Objetivo**: Dado un conjunto de características extraídas del cuerpo humano en movimiento, predecir la clase de actividad que se está realizando.
- **Naturaleza del problema**: Multiclase (ej. caminar, sentarse, saltar, inclinarse...).

---

## Metodología

Seguiremos el enfoque **CRISP-DM (Cross-Industry Standard Process for Data Mining)**, que comprende:

1. **Comprensión del problema**  
   - Detectar y clasificar actividades humanas a partir de video.

2. **Comprensión de los datos**  
   - Extracción de puntos clave del cuerpo usando MediaPipe.
   - Generación de variables derivadas: ángulos, distancias entre articulaciones, velocidad y aceleración articular.

3. **Preparación de los datos**  
   - Limpieza de ruido, interpolación de puntos perdidos.
   - Normalización y balanceo del conjunto de datos.

4. **Modelado**  
   - Clasificadores candidatos: KNN, Random Forest, SVM.
   - Posible aplicación de PCA o selección de características para reducción dimensional.

5. **Evaluación**  
   - Métricas de precisión y generalización sobre conjunto de prueba.

---

## Métricas de evaluación

Para medir el progreso y el desempeño del modelo, utilizaremos:

- **Accuracy**: porcentaje de actividades correctamente clasificadas.
- **Precision / Recall / F1-score**: por clase.
- **Matriz de confusión**: para visualizar errores de clasificación.
- **Tiempo de inferencia por frame**: para evaluar la viabilidad en tiempo real.

---

## Datos recolectados hasta el momento

Actualmente se han recopilado:

Link del dataset: 

https://drive.google.com/drive/folders/1KGiezhSh1gUQatXimF124qroUoBy3VGR?usp=sharing

- Videos grabados por el equipo simulando distintas actividades.
- Datos de puntos clave obtenidos con **MediaPipe Pose**, incluyendo coordenadas 2D de hombros, codos, rodillas, tobillos, etc.
- Anotaciones semiautomáticas y manuales por cada segmento del video.

Cada muestra consta de:
- Coordenadas XY de hasta 33 keypoints.
- Clase de actividad (etiquetada manualmente).
- Variables derivadas: ángulos de articulación, distancias entre segmentos, velocidad relativa.

---

## Análisis exploratorio de datos (EDA)

Se han identificado patrones preliminares:

- Algunas actividades como caminar, estar de pie o estirarse tienen posturas y secuencias articulares distinguibles.
- Hay superposición entre posturas similares como estar de pie vs. quieto.
- La distancia entre caderas y tobillos o los ángulos en rodillas y codos son buenos discriminadores.

Ejemplos de visualizaciones:
- Gráficos de dispersión por pares de keypoints.
- Secuencias de movimiento (frames animados).
- Histograma de duración de actividades por clase.

---

## Estrategias para conseguir más datos

Dado que los modelos de IA requieren una diversidad representativa para generalizar, se proponen las siguientes estrategias:

1. **Grabación de más videos**:
   - Con diferentes personas, entornos, ropa y luz ambiental.
   - Con diferentes cámaras o resoluciones.

2. **Crowdsourcing**:
   - Solicitar a voluntarios que graben y compartan videos cortos.

3. **Aumento de datos sintéticos**:
   - Uso de técnicas como jittering (agregar ruido leve), rotación leve, interpolación temporal

4. **Uso de datasets abiertos**:
   - Integrar conjuntos como **Human3.6M**, **PoseTrack**, o **MPII Human Pose** si el formato es compatible.

---

## Análisis de aspectos éticos

Implementar soluciones de IA con datos de video y movimiento humano conlleva responsabilidades importantes:

1. **Privacidad**:
   - Se debe garantizar que ningún rostro ni elemento identificable se utilice sin consentimiento explícito.
   - Se recomienda anonimizar videos y trabajar solo con datos derivados (keypoints sin imagen).

2. **Consentimiento informado**:
   - Las personas filmadas deben firmar un consentimiento claro sobre el uso de sus movimientos con fines académicos.

3. **No discriminación**:
   - Evitar sesgos introducidos por grabar solo ciertos grupos demográficos. Se debe procurar diversidad de género, edad y contextura física.

4. **Finalidad educativa/no lucrativa**:
   - Aclarar que el proyecto es experimental y no se usará para vigilancia, puntuación física o control de personas.

5. **Responsabilidad del mal uso**:
   - Aclarar en la licencia y en el repositorio los límites éticos de uso.

---

## Próximos pasos

1. Continuar ampliando el conjunto de datos con nuevos videos.
2. Definir formalmente las clases de actividad que se incluirán.
3. Desarrollar un pipeline completo de extracción de características.
4. Entrenar y comparar los clasificadores seleccionados.
5. Evaluar la solución con métricas reales y ajustar hiperparámetros.

---

## Referencias

- CRISP-DM methodology: https://www.sv-europe.com/crisp-dm-methodology/