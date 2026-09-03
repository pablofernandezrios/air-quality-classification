# Clasificación de la calidad del aire

Clasificación multiclase de la calidad del aire a partir de variables
meteorológicas, de contaminación y socioeconómicas, comparando cinco familias de
modelos bajo un protocolo experimental común: **red de neuronas artificial
(ANN)**, **máquina de vectores soporte (SVM)**, **k vecinos más cercanos
(k-NN)**, **árbol de decisión** y **regresión simbólica (DoME)**.

Todos los modelos se evalúan con la misma partición de validación cruzada
estratificada, de modo que las diferencias observadas son atribuibles al
algoritmo y no al reparto de los datos.

## Dataset

`data/pollution.csv` contiene **5.000 instancias** con 9 atributos numéricos y
una etiqueta de clase:

| Atributo | Descripción | Unidad |
| --- | --- | --- |
| `Temperature` | Temperatura | °C |
| `Humidity` | Humedad relativa | % |
| `PM2.5` | Partículas < 2,5 µm | µg/m³ |
| `PM10` | Partículas < 10 µm | µg/m³ |
| `NO2` | Dióxido de nitrógeno | µg/m³ |
| `SO2` | Dióxido de azufre | µg/m³ |
| `CO` | Monóxido de carbono | mg/m³ |
| `Proximity_to_Industrial_Areas` | Distancia a zonas industriales | km |
| `Population_Density` | Densidad de población | hab/km² |
| `Air Quality` | **Clase objetivo**: `Good`, `Moderate`, `Poor`, `Hazardous` | — |

### Limpieza: 5.000 → 4.775 instancias

Se descartan las filas con valores **físicamente imposibles**, comprobando cada
columna contra su rango admisible. Se eliminan **225 filas (4,5 %)**:

| Motivo | Filas |
| --- | --- |
| Humedad relativa > 100 % (máximo observado: 128,1 %) | 195 |
| SO₂ negativo (mínimo observado: −6,2 µg/m³) | 30 |
| PM10 negativo | 1 |
| **Total eliminado** | **225** |

Quedan **4.775 instancias**, con la siguiente distribución de clases (el
conjunto está desbalanceado, de ahí el uso de F1 ponderado y de validación
cruzada estratificada):

| Clase | Instancias | % |
| --- | --- | --- |
| `Good` | 1.983 | 41,5 % |
| `Moderate` | 1.493 | 31,3 % |
| `Poor` | 930 | 19,5 % |
| `Hazardous` | 369 | 7,7 % |

Los límites aplicados están declarados explícitamente en el vector `limits` de
`src/resultados.jl` y `src/grid_search.jl`.

## Metodología

- **Normalización Min-Max** de los 9 atributos al intervalo [0, 1], calculada
  sobre el conjunto ya filtrado.
- **Codificación one-hot** de la variable objetivo para el entrenamiento de la
  ANN.
- **Validación cruzada estratificada de 5 folds**, que preserva la proporción de
  clases en cada partición. Los índices se generan una sola vez y se guardan en
  `data/indices.csv`, de forma que **los cinco modelos se evalúan sobre
  exactamente los mismos folds**.
- **Grid Search** sobre los hiperparámetros de cada familia (`src/grid_search.jl`),
  paralelizado con `Distributed` sobre todos los núcleos disponibles.
- **Semilla fija** (`Random.seed!(67)`) para la inicialización estocástica.
- **Métrica principal**: F1 ponderado por soporte de clase, promediado sobre los
  5 folds y acompañado de su desviación típica. Se registran además accuracy,
  precisión y recall.
- La ANN se entrena **5 veces por fold** y se promedian los resultados, para
  amortiguar la variabilidad de la inicialización de pesos.
- Se exporta el F1 por fold de los dos mejores modelos (`data/folds_ANN.csv`,
  `data/folds_SVM.csv`) para poder contrastar su diferencia con un test de
  Wilcoxon.

### Espacio de búsqueda

| Modelo | Configuraciones | Hiperparámetros explorados |
| --- | --- | --- |
| ANN | 10 | Topologías de 1 y 2 capas ocultas (10–50 neuronas) |
| SVM | 12 | Kernels `linear`, `rbf`, `poly`, `sigmoid`; `C`; `γ` |
| DoME | 10 | `maximumNodes` de 5 a 100 |
| Árbol de decisión | 8 | `max_depth` de 3 a 25 |
| k-NN | 8 | `k` de 3 a 25 |

## Resultados

Mejor configuración de cada familia (F1 ponderado, media ± desviación típica
sobre los 5 folds). Los CSV completos con todas las configuraciones están en
`results/`:

| Modelo | Mejores hiperparámetros | F1 ponderado | Accuracy |
| --- | --- | --- | --- |
| **SVM** | `kernel=rbf`, `C=10.0`, `γ=1.0` | **0,9470 ± 0,0089** | 0,9476 |
| **ANN** | topología `[50, 25]`, `lr=0.01`, 1000 épocas | **0,9438 ± 0,0089** | 0,9444 |
| DoME | `maximumNodes=100` | 0,9361 ± 0,0107 | 0,9380 |
| k-NN | `k=15` | 0,9278 ± 0,0110 | 0,9303 |
| Árbol de decisión | `max_depth=10` | 0,9263 ± 0,0105 | 0,9269 |

**Conclusiones:**

- Los cinco modelos superan un **F1 ponderado de 0,92**, lo que indica que el
  problema es separable con un margen amplio y que ninguna familia queda
  descartada.
- **SVM y ANN ofrecen el mejor rendimiento general** (≈ 0,944–0,947). La
  diferencia entre ambos (0,0032) es inferior a la desviación típica entre folds
  de cualquiera de los dos, por lo que no puede considerarse concluyente sin el
  contraste de hipótesis; para eso se exportan los F1 por fold.
- DoME es el modelo más sensible a su hiperparámetro: pasa de 0,511 con
  `maximumNodes=5` a 0,936 con 100, siendo el único que alcanza rendimiento
  competitivo a costa de expresiones considerablemente más complejas.
- El árbol de decisión y el k-NN se saturan pronto: ampliar la profundidad más
  allá de 10, o `k` más allá de 15, no mejora el resultado.
- El kernel polinómico es la única configuración de SVM claramente inadecuada
  (0,524 con `C=1.0`).

## Cómo ejecutarlo

**Requisitos:** Julia ≥ 1.10 (desarrollado y probado con 1.12).

```bash
git clone https://github.com/pablofernandezrios/ejercicios_faa.git
cd ejercicios_faa
```

Instalar las dependencias declaradas en `Project.toml`:

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

### Evaluación final de los mejores modelos

Entrena las cinco configuraciones ganadoras e imprime, para cada una, las
métricas por clase, el F1 global y la matriz de confusión:

```bash
julia --project src/resultados.jl
```

### Búsqueda de hiperparámetros completa

Reproduce el Grid Search y regenera los CSV de `results/`. Es considerablemente
más costoso, ya que evalúa 48 configuraciones con validación cruzada de 5 folds:

```bash
julia --project src/grid_search.jl
```

Por defecto usa `nproc - 1` workers. Para fijar el número de procesos y de hilos:

```bash
JULIA_NUM_WORKERS=4 julia --project --threads 4 src/grid_search.jl
```

Ambos scripts resuelven sus rutas a partir de su propia ubicación, así que
pueden lanzarse desde cualquier directorio de trabajo.

> **Reproducibilidad:** si `data/indices.csv` existe, los scripts reutilizan esa
> partición en lugar de generar una nueva. Bórralo únicamente si quieres una
> partición distinta; en ese caso, los resultados dejarán de ser comparables con
> los CSV versionados en `results/`.

## Estructura del proyecto

```
.
├── data/
│   ├── pollution.csv                   # Dataset original (5.000 instancias)
│   ├── indices.csv                     # Índices de los 5 folds estratificados
│   ├── folds_ANN.csv                   # F1 por fold de la ANN (test de Wilcoxon)
│   └── folds_SVM.csv                   # F1 por fold del SVM (test de Wilcoxon)
├── src/
│   ├── firmas2.jl                      # Biblioteca base: normalización, one-hot,
│   │                                   #   ANN, validación cruzada, métricas
│   ├── grid_search.jl                  # Búsqueda paralela de hiperparámetros
│   └── resultados.jl                   # Evaluación final de los mejores modelos
├── results/
│   ├── grid_resultados_ANN.csv
│   ├── grid_resultados_SVM.csv
│   ├── grid_resultados_DoME.csv
│   ├── grid_resultados_KNN.csv
│   └── grid_resultados_Decision_Tree.csv
├── Project.toml                        # Dependencias del entorno Julia
└── README.md
```

## Autoría

Trabajo desarrollado **en grupo** para la asignatura de Fundamentos de
Aprendizaje Automático (Universidade da Coruña). Las cuatro personas que
participaron son:

- Pablo Fernández Ríos ([@pablofernandezrios](https://github.com/pablofernandezrios)) — mantenedor del repositorio
- Yare B. Espinosa ([@YareBE](https://github.com/YareBE))
- Paula Rodil ([@paularodil](https://github.com/paularodil))
- Raúl Cayón ([@raulcayon](https://github.com/raulcayon))

El historial de Git conserva la autoría original de cada commit. Para consultar
el reparto real de contribuciones:

```bash
git shortlog -sne
```
