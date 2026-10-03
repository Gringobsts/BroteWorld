# Primer resultado: aprender a desplazarse

Fecha: 2 de octubre de 2026. CPU local, pesos aleatorios, sin modelos preentrenados.

## Condiciones

Tres semillas de inicialización (17, 29, 43); 12.288 acciones de entrenamiento por semilla. Cada 1.024 acciones se cambia de mapa de entrenamiento. La evaluación usa cinco mapas reservados (1001–1005), 512 acciones por mapa y un generador aleatorio independiente. La política genera acciones estocásticas; no recibe rutas ni máscaras de movimientos válidos. No se actualizaron pesos durante evaluación.

Los resultados corresponden a la primera configuración probada, sin ajustar a partir de los errores de estos mapas reservados. Las casillas, sensores locales y recompensas son diseñados, no aprendidos.

## Resultado

| Semilla | Movimiento antes | Movimiento después | Choques antes | Choques después |
| --- | --- | --- | --- | --- |
| 17 | 61.8% | 98.5% | 18.3% | 1.4% |
| 29 | 69.1% | 96.8% | 11.1% | 3.2% |
| 43 | 72.6% | 91.8% | 11.1% | 8.2% |

Promedio: movimientos logrados **67.8% → 95.7%**; choques **13.5% → 4.3%**. La cobertura de terreno en 512 turnos pasó de 27.8% a 31.4%. Mejoró el desplazamiento más que la exploración extensa.

Las tres semillas conservaron exactamente los resultados al cargar el checkpoint. También produjeron los mismos pesos al continuar el entrenamiento con el mismo estado aleatorio. Una prueba adicional ejecutó la restauración y continuación en un proceso nuevo.

## Reloj y aceleración

La prueba de 1× ejecutó 2 turnos en 2,1 segundos reales. A una tasa solicitada de 10.000× ejecutó 20046 turnos en 2.010 segundos: **9973.7 turnos/s efectivos**, con 313 actualizaciones de aprendizaje durante esa partida de prueba. El contador de acciones coincidió con el de turnos. Esta medición breve depende del equipo y la carga; no garantiza sostener esa tasa con más agentes o un mundo más complejo.

La velocidad efectiva se muestra en la interfaz. No se descartan experiencias para acelerarla. A alta velocidad la vista muestrea el estado, pero cada turno del motor se ejecuta.

## Verificaciones y límites

24 pruebas pasaron: separación de evaluación, gradientes del motor anterior, aprendizaje, persistencia, una acción por turno, acumulación de tiempo fraccionario, cambio de velocidad sin crédito fantasma, pausa, avance manual y controles del servicio. El análisis estático y la comprobación de sintaxis de JavaScript pasaron.

La inspección visual desde el navegador integrado quedó bloqueada porque su control de seguridad no pudo verificar la política del administrador. No se evitó ese control; la página queda disponible para abrirla directamente.

Esto demuestra aprendizaje de una habilidad muy limitada de navegación discreta. No demuestra visión desde píxeles, control muscular, tala, construcción, cultivo ni aprendizaje social. La retención entre habilidades y el aprendizaje entre agentes quedan pendientes.

La partida en vivo es independiente de estos checkpoints y continúa aprendiendo mientras el servidor está encendido. El guardado automático ocurre cada 20 segundos y al cierre normal.

Datos completos: [reporte del experimento](../runs/world-experiment/report.json). Configuración y reproducción: [README](../README.md).
