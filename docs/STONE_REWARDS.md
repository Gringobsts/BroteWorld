# Premios condicionados a la misión

Se observó que Brote seguía cobrando por ambas maderas después de superar sus cuotas, mientras piedra seguía pendiente pese a disponer de pico. Se cambió únicamente el premio por recursos: cero después de cumplir la cuota correspondiente, con reactivación en la siguiente misión. No se alteraron sensores, acciones, física ni hiperparámetros de aprendizaje.

## Comprobación de comportamiento

84 pruebas automatizadas pasan. Cubren golpes y recolección después de cumplir cuotas, último recurso necesario, piedras sueltas, consumo de materiales, renovación de misión, penalización del plazo y continuidad exacta de pesos, optimizador y aleatoriedad al procesar experiencias pendientes. El servidor conserva la partida; las pruebas de entrenamiento usan copias independientes.

## Ensayo limitado de aprendizaje

Se continuó el mismo guardado congelado durante 131.072 turnos por condición. El control conservó los premios anteriores; el ensayo los anuló para cuotas cumplidas. Una sola trayectoria por condición: no es una estimación robusta de mejora.

| Recursos obtenidos durante entrenamiento | Control | Premios condicionados |
| --- | ---: | ---: |
| Piedra | 32 | 61 |
| Madera blanda | 34428 | 18000 |
| Madera dura | 34587 | 18492 |

Evaluación sin entrenamiento en semillas 1501, 1502 y 1503, con 2.048 acciones por mundo y generadores de acciones independientes (semilla del mapa + 1039). Para medir piedra se entregaron ambas herramientas y se marcaron cumplidas las otras tres cuotas. Antes y después del ensayo obtuvo una media de 1 piedra y 0 rocas grandes extraídas por mundo. Por tanto, no se ha demostrado aprendizaje útil de minería. La menor tala sin premio es un cambio de conducta, no evidencia de haber dominado piedra.

La evaluación separada de madera blanda pendiente bajó de 499 a 164 unidades de media, por debajo de la retención deseada. Esta prueba corta también señala posible interferencia entre habilidades. Los modelos del ensayo nunca reemplazan la partida; no se garantiza retención por el solo hecho de conservar los pesos.

El script, el guardado inicial y los resultados crudos se archivan en una carpeta nueva bajo `runs/stone-rewards-*`. Para reproducir, ejecutar el script de esa carpeta con `--source` apuntando a `initial.pt`, `--out` a una carpeta nueva y `--versions 1 2` con el código de esta actualización.
