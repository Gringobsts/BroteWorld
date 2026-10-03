# Recursos, herramientas y nueva misión · 2 de octubre de 2026

## Comportamiento implementado

- Árboles blandos: tres golpes a mano, tres de madera blanda. Árboles duros: hacha, tres golpes, tres de madera dura. Ambos bloquean y rebrotan después de 100.000 turnos conservando su tipo.
- Rocas grandes: bloquean, requieren pico y tres golpes; entregan tres de piedra y dejan suelo transitable. No se regeneran.
- Piedras sueltas: están sobre suelo transitable; cada una da una piedra. Caminar hasta la casilla y recoger son dos acciones distintas. Recoger solo consulta la casilla del propio agente, nunca una vecina.
- Hacha y pico: cada uno cuesta dos de madera blanda y tres piedras. Fabricar consume un turno; sin materiales o con la herramienta ya disponible falla sin descontar nada. No hay desgaste en esta etapa.
- Misión: 10.000 monedas, 100.000 de madera blanda, 100.000 de piedra y 100.000 de madera dura. Recolección bruta por misión, independiente del gasto al fabricar. Premio +100 al completar las cuatro; nueva misión inmediata. Penalización −25 cada 50.000 turnos sin cumplirla. Mantiene progreso y plazos al guardar o cambiar de terreno.

## Verificación

72 pruebas automáticas pasan. Cubren bloqueo y transitabilidad, recogida en dos turnos, herramientas correctas, costos y fallos de recetas, sensores, agotamiento de rocas, rebrote, cuatro metas, premio y penalización, inventario, cambios de terreno, checkpoint y continuidad del entrenamiento. Ruff y la comprobación de sintaxis de JavaScript pasan. Una comprobación de lógica de la interfaz ejercita las cuatro barras, inventario, herramientas y acciones con un estado real; no equivale a inspección visual completa.

Se congeló una copia v2 de la partida y se completó su experiencia pendiente con el código anterior. El código nuevo reconstruyó exactamente la observación v2 y, tras completar ese mismo aprendizaje, produjo pesos y momentos de Adam idénticos en todas las dimensiones originales. Se comprobaron el inventario, los turnos y la copia de seguridad. En otra prueba sobre esa copia se entrenaron 4.097 acciones, se guardó/restauró y se continuaron 67 acciones con posiciones, mundo, acciones y pesos idénticos. La partida no fue reemplazada por esa prueba.

La ampliación mantiene las 39 entradas y nueve acciones como prefijo y agrega 33 entradas y siete acciones. Red final: 72 → 64 tanh → 64 tanh, actor de 16 salidas y crítico de una. 9.792 conexiones, 9.937 parámetros y 39.748 bytes float32. Las dimensiones nuevas se inicializan sin pesos preentrenados. La memoria de la etapa anterior se conserva; esto no garantiza ausencia de olvido durante aprendizaje futuro.

## Límites de la evidencia

Las pruebas verifican reglas y entrenamiento ejecutable. No prueban que Brote domine autónomamente la secuencia recoger → fabricar → extraer, ni que termine las nuevas metas de 100.000 unidades. Su aprendizaje continúa. La futura predicción, curiosidad y planificación todavía no están implementadas. La distribución de monedas continúa siendo procedural; 10.000 es la cantidad que hay que recoger por misión, no un límite global de monedas visibles en todo el mundo.

El hacha que mostraba el servidor v2 se conserva en la migración. La madera acumulada se interpreta como blanda. Las nuevas partidas arrancan sin herramientas. Los experimentos y sus checkpoints permanecen separados de `runs/world/live.pt`.
