# Guía de trabajo y defensa (español)

## Qué hace cada archivo

- `envs/discretizer.py`: transforma un estado continuo de CartPole en un índice finito para la tabla Q.
- `agents/tabular.py`: aplica el algoritmo de actualización de Q-learning y guarda/restaura la tabla.
- `agents/dqn.py`: contiene la red neuronal, el buffer de experiencias, la optimización DQN y la variante Double DQN.
- `experiments/configs.py`: declara las 4 configuraciones de Tier 1, las 6 de Tier 2 y la de Tier 3.
- `experiments/trainer.py`: centraliza el entrenamiento y el protocolo de evaluación idéntico entre comparaciones.
- `scripts/run_experiments.py`: ejecuta experimentos reproducibles con semillas controladas.
- `scripts/plot_results.py`: calcula agregados entre semillas y genera figuras a partir de CSV reales.
- `scripts/record_agent.py`: graba una demostración del agente real usando un modelo guardado.
- `tests/`: pruebas automáticas para errores matemáticos y errores del flujo experimental.

## Puntos que ambos deben dominar

1. La ecuación de Bellman y la diferencia entre el valor objetivo y el valor predicho.
2. Por qué Q-learning necesita discretización en CartPole y por qué existe un compromiso entre generalización y resolución.
3. Por qué las observaciones continuas pueden pasar directamente a la red neuronal de DQN.
4. Para qué sirven la memoria de experiencias y la red objetivo.
5. Qué cambia exactamente en la ablación sin red objetivo.
6. Cómo Double DQN reduce potencialmente la sobreestimación separando selección y evaluación de acciones.
7. Por qué `terminated` y `truncated` deben manejarse de forma diferente.
8. Cómo y por qué utilizamos tres semillas, evaluación independiente, curvas ± desviación estándar y el mismo número de pasos.
9. Por qué alcanzar el umbral con 10 episodios no significa haber validado el umbral sobre 100 episodios.
10. Cómo reproducir las figuras y reconocer resultados negativos sin inventar explicaciones.

## Responsabilidad académica

La automatización del código no sustituye la revisión por ambos estudiantes. El profesor puede pedir modificaciones en directo. Cada integrante debe poder explicar las clases, fórmulas, hiperparámetros y gráficos; por eso mantenemos pruebas automáticas y código legible.
