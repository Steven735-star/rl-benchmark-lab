# Revisión científica de los resultados de Reinforcement Learning

**Estado:** auditoría de datos y borrador de artículo; no constituye una entrega final revisada por los integrantes.

## Fuentes y alcance

- Archivo original: `resultados_full_rl.zip` entregado por el equipo.
- Material revisado: 33 JSON de resumen, 33 CSV de evaluación, 33 CSV de episodios, 33 checkpoints, 8 PNG y 2 GIF.
- Entrenamientos: 11 configuraciones × 3 semillas (0, 1, 2); 4.500.000 pasos en total.
- Evaluación: 10 episodios en cada punto periódico y 100 episodios en la evaluación final.
- Los resultados que se muestran a continuación se **calcularon a partir de los archivos originales**, no de resultados simulados.

## Resumen de Tier 1: CartPole-v1

| Configuración | Media final entre semillas | DE muestral | Semillas que superan 475/500 |
|---|---:|---:|---:|
| Q-learning (6 bins) | 161,80 | 37,78 | 0/3 |
| Q-learning (10 bins) | 275,68 | 193,52 | 1/3 |
| Q-learning (20 bins) | **280,87** | 127,96 | 0/3 |
| DQN | 117,46 | 34,45 | 0/3 |

Para el proyecto el umbral de éxito es **475** puntos de retorno promedio, evaluados sobre 100 episodios por semilla. Q-learning con 10 bins alcanzó **497,22** en la semilla 1, pero no lo reprodujo en las otras semillas. Los 20 bins produjeron la mayor media de las tres semillas, pero la diferencia de 5,19 puntos frente a 10 bins es muy inferior a su variabilidad: no permite declarar superioridad robusta.

DQN obtuvo un retorno periódico medio de **222,80 a los 10.000 pasos**, por encima de todas las discretizaciones tabulares en ese punto. Sin embargo, su rendimiento periódico descendió y su media final sobre 100 episodios fue **117,46**. Se observa aprendizaje inicial seguido de pérdida de desempeño en la evaluación, pero los archivos no contienen diagnósticos de valores Q o del error TD capaces de identificar una causa concreta.

## Resumen de Tier 2: Acrobot-v1

| Configuración | Media final entre semillas | DE muestral | Semillas que superan -100 |
|---|---:|---:|---:|
| DQN base | -371,05 | 223,35 | 0/3 |
| LR = 0,0001 | **-254,25** | 217,55 | 1/3 |
| LR = 0,001 | -373,04 | 219,90 | 0/3 |
| Decaimiento ε 20 % | -500,00 | 0,00 | 0/3 |
| Decaimiento ε 80 % | -361,72 | 239,51 | 1/3 |
| Sin target network | -500,00 | 0,00 | 0/3 |

El valor negativo más cercano a cero es mejor. El mejor promedio entre las configuraciones de Tier 2 corresponde a LR = 0,0001, aunque sus resultados entre semillas son muy dispares. La ablación sin red objetivo terminó con retorno **-500,00** en las tres semillas. Es una observación compatible con un papel estabilizador de la red objetivo en este montaje; no constituye una demostración de causalidad universal.

## Resumen de Tier 3: Double DQN

| Semilla | DQN base | Double DQN | Diferencia Double - base |
|---|---:|---:|---:|
| 0 | -113,15 | -213,49 | -100,34 |
| 1 | -500,00 | -94,64 | +405,36 |
| 2 | -500,00 | -500,00 | 0,00 |
| **Media** | **-371,05** | **-269,38** | **+101,67** |

Double DQN mejoró la media general **en 101,67 puntos**, pero no lo hizo en todas las semillas. La desviación estándar muestral de las diferencias pareadas fue aproximadamente **267,74**: con tres observaciones no se puede inferir una mejora estadísticamente sólida. La justificación teórica de Double DQN se refiere a la reducción de la sobreestimación, pero el experimento no registró el error de estimación de Q; por tanto, no podemos afirmar que la sobreestimación se haya reducido efectivamente aquí.

## Cobertura y tiempos

- Tier 1: **12/12** ejecuciones, 2,4 millones de pasos.
- Tier 2: **18/18** ejecuciones, 1,8 millones de pasos.
- Tier 3: **3/3** ejecuciones, 0,3 millones de pasos.
- Tiempo registrado **dentro de las 33 ejecuciones** (incluidas evaluaciones): **1.229,97 segundos** = **20 min 29,97 s**. Esto es ligeramente inferior a los tiempos `real` de la terminal porque no incluye todos los costos de inicio de proceso y trabajos auxiliares.
- Se detectaron **4/33 evaluaciones finales** que cumplen el umbral (1 en CartPole 10 bins; 1 en Acrobot LR=0,0001; 1 en Acrobot ε 80 %; 1 en Double DQN). **Ninguna configuración lo cumplió en las tres semillas.**
- Se incluyen dos GIF válidos: CartPole con 276 fotogramas y Acrobot con 91 fotogramas. Cada GIF corresponde a una sola trayectoria visual, no a la evaluación final completa.

## Verificaciones efectuadas

1. Coinciden los 33 identificadores de ejecución y sus modelos guardados.
2. Cada resumen coincide numéricamente con su fila de evaluación final en CSV.
3. Las evaluaciones periódicas aparecen en pasos 0, 10.000, ... y el paso final previsto.
4. Las evaluaciones periódicas tienen diez episodios y las finales 100.
5. Las marcas de éxito de los resúmenes son consistentes con los umbrales finales.
6. Los metadatos indican entrenamiento en CPU con Python 3.14.4, Gymnasium 1.4.0 y PyTorch 2.15.0+cu132.

## Limitaciones que deben declararse

- **Solo tres semillas:** las diferencias entre configuraciones tienen mucha incertidumbre.
- **Evaluación periódica de 10 episodios:** un valor superior al umbral allí no implica éxito validado. Las marcas `first_provisional_10episode_threshold_step` son provisionales.
- **Se guardan los pesos del último paso:** los mejores resultados intermedios pueden desaparecer durante entrenamiento posterior.
- **No se registran pérdidas o valores Q a lo largo del entrenamiento:** la causa de la inestabilidad no está identificada experimentalmente.
- **Los retornos del entrenamiento no son retornos greedy:** deben graficarse y citarse por separado de la evaluación.
- **El equipo debe verificar y entender el código:** el proyecto permite IA, pero requiere defensa oral y declaración de uso.
- **Reproducción de referencias:** comprobar las versiones y enlaces de documentación antes de la entrega final, junto con el formato requerido por el docente.

## Entregables preparados

- `articulo_RL_DQN_borrador.pdf`: manuscrito académico en inglés, dos columnas, seis páginas, con tablas y figuras basadas en los resultados medidos.
- `articulo_RL_DQN_borrador.tex`: fuente editable LaTeX; requiere la carpeta `paper_figures` para compilar.
- `consolidado_33_ejecuciones.csv`: resumen de 11 configuraciones con medias, desviaciones, aciertos y tiempo.
- `detalle_33_ejecuciones.csv`: las 33 evaluaciones finales con metadatos.
- `auditoria.json`: validación automatizada de integridad del archivo.
- `paper_figures/`: figuras vectoriales regeneradas desde los CSV originales.

**Pendiente antes de entregar:** completar nombres de autores, verificar bibliografía y formato, revisar por ambos integrantes el código y el texto, agregar los prompts realmente utilizados, decidir si se realizan experimentos adicionales y confirmar que el informe cumpla los criterios específicos del docente.
