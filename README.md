# Brain · Un pequeño mundo

Un mundo de píxeles visto desde arriba donde **Brote** aprende continuamente por prueba y error. Su mente empezó con pesos aleatorios: no usa modelos preentrenados, rutas programadas ni reglas que decidan sus acciones. El servidor mantiene el entrenamiento aunque cierres la pestaña.

## Abrir

```bash
cd ~/Gringo/brain
uv sync --locked
uv run brain-world
```

Abrí [el mundo local](http://127.0.0.1:8765). Podés pausar, avanzar un turno, cambiar velocidad, ver el recorrido, guardar o cambiar de terreno. Se guarda cada 20 segundos y al cerrar normalmente, en `runs/world/live.pt`.

## Un mundo grande, una acción por turno

- El terreno mide **1.000.000 × 1.000.000 bloques**. Se calcula por coordenadas y semilla; solo se guardan visitas y cambios. No se crea una matriz de un billón de casillas.
- La cámara de 24 × 18 bloques sigue a Brote. Su visión propia es de **3 × 3**.
- Hay dieciséis acciones: caminar en cuatro direcciones, esperar, talar en cuatro direcciones, fabricar hacha, fabricar pico, picar en cuatro direcciones y recoger una piedra del suelo.
- Cualquier intento consume un turno, incluso fallar. Cada turno equivale a un segundo simulado.
- La velocidad solicitada va de 0,01 a 10.000 turnos por segundo real. Se muestra la velocidad efectiva: no se saltan acciones ni entrenamiento para aparentar velocidad.
- Los árboles blandos requieren tres golpes sin herramienta y dan tres de madera blanda. Los duros requieren hacha, tres golpes y dan tres de madera dura. Ambos bloquean el paso; sus tocones son transitables.
- Cada árbol talado vuelve a crecer **100.000 turnos después de caer**. Si Brote ocupa su casilla, espera a que la deje libre.

```bash
uv run brain-world --speed 10000
```

## Recursos y herramientas

Las rocas grandes bloquean el paso. Se extraen con tres golpes de pico y entregan tres de piedra; la casilla queda transitable y la roca no se regenera. Las piedras pequeñas están sobre suelo transitable. **Moverse a su casilla y recogerlas son dos acciones distintas, en dos turnos.** Recoger solo funciona en la casilla actual; no toma piedras vecinas ni recoge automáticamente al caminar.

- Hacha: **2 de madera blanda + 3 piedras**; permite talar árboles duros.
- Pico: **2 de madera blanda + 3 piedras**; permite extraer rocas grandes.

Fabricar consume una acción y descuenta los materiales del inventario. En esta etapa las herramientas no se desgastan; fabricar una que ya tiene falla sin gastar materiales. Los mundos nuevos comienzan sin herramientas. Brote conserva el hacha inicial que mostraba la versión anterior. Su madera anterior pasa a ser madera blanda, sin perder unidades.

El modelo recibe señales negativas por falta de herramienta, falta de materiales, intentar fabricar duplicados, recoger donde no hay piedra o picar donde no hay roca. El motor comprueba la física; la política sigue pudiendo elegir cualquiera de las dieciséis acciones.

## Misión y señales

La misión requiere **10.000 monedas, 100.000 de madera blanda, 100.000 de piedra y 100.000 de madera dura**. Las monedas aparecen en posiciones pseudoaleatorias reproducibles del suelo transitable. Se generan al consultar zonas y desaparecen al recogerlas; no hay que encontrar solo cien puntos dispersos entre un billón de bloques. Entrar en una casilla con moneda la recoge dentro de la misma acción. Al llegar a 10.000 en la misión dejan de aparecer hasta la próxima. Las posiciones siguen siendo procedurales: la meta es recoger 10.000, no ubicar únicamente 10.000 puntos entre un billón de casillas.

| Suceso | Recompensa |
| --- | ---: |
| Movimiento válido | +0,02 más hasta +0,05 por exploración |
| Chocar con un bloque no transitable | −0,5 |
| Esperar | −0,06 |
| Golpear un árbol | +0,4 |
| Último golpe: árbol talado o roca extraída, 3 de su recurso | +2,4 |
| Recoger una piedra suelta en la casilla actual | +0,8 |
| Fabricar una herramienta que falta | +4 |
| Falta de herramienta | −0,5 |
| Acción de recolección/fabricación inválida | −0,35 |
| Recoger una moneda | +3 adicional |
| Completar las cuatro metas | +100 adicional, una vez por misión |
| Cada 50.000 turnos sin completar las cuatro metas | −25 adicional |

Al completar las cuatro metas, cobra el premio y empieza automáticamente otra misión con las mismas cuatro metas, en el mismo turno. El progreso vuelve a cero, las monedas reciben nuevas posiciones y el plazo vuelve a 50.000 turnos. El inventario acumulado, los árboles, el mundo y la misma mente se conservan; los recursos anteriores no completan la nueva misión. Se cuenta la recolección bruta de cada recurso: gastarlo en herramientas no reduce el progreso; los materiales gastados se conservan como contadores para este cálculo. La interfaz muestra la misión actual y cuántas completó.

El plazo empieza al asignar cada misión, no al iniciar el programa. Guardar, reiniciar y cambiar de terreno conservan el inventario, el progreso y la próxima penalización. Si completa las cuatro metas justo en el límite, recibe el premio sin penalización. Las señales de un mismo turno se suman y se muestran por separado en el estado. No se agrega un castigo general por cada turno con la misión pendiente.

Las recompensas son un diseño experimental, no garantizan que aprenda la estrategia deseada. Con un descuento de 0,95, el agente relaciona sobre todo consecuencias cercanas: una penalización a 50.000 turnos no basta para enseñarle planificación a esa escala. Los premios por avances proporcionan señales inmediatas. Sigue recibiendo experiencia después de completar la misión; no se reinicia ni congela la mente.

## La misma mente, aprendizaje continuo

La red recibe 72 valores: conserva las 39 entradas anteriores y agrega tipo de árbol, resistencia de rocas, piedras sueltas, herramientas, materiales disponibles para fabricar y progreso de piedra y madera dura. Tiene dos capas compartidas de 64 neuronas tanh, un actor de 16 salidas y un crítico de una salida: 9.937 parámetros, 9.792 conexiones y 39.748 bytes de parámetros float32. No conoce el mapa global ni aprende visión desde las imágenes de la pantalla. Actualmente es una red pequeña sin memoria neuronal recurrente; la memoria temporal explícita está en las visitas y el estado del mundo.

La política y el estimador de valor se entrenan con PPO cada 64 experiencias, hasta cuatro pasadas. Se mantiene un 10 % de exploración uniforme y temperatura 3 en las probabilidades para que una habilidad muy consolidada no impida probar acciones nuevas. El muestreo y la actualización usan la misma distribución.

**La partida conserva a Brote:** pesos, optimizador, generador aleatorio, experiencias pendientes, turnos, inventario, misión, daño de árboles y plazos de rebrote. Al ampliar la versión anterior de 18 sentidos y 5 acciones a 72 sentidos y 16 acciones, se terminan sus experiencias pendientes con la política anterior y se copian todas sus conexiones y momentos del optimizador. Solo se agregan las nuevas entradas/salidas. La migración de 39 entradas/9 acciones a 72/16 también aprende primero la experiencia pendiente con los sensores y probabilidades anteriores. Se copian los pesos y momentos existentes, y solo se inicializan las dimensiones añadidas. Se conservan `live.v1-backup.pt` y/o `live.v2-backup.pt` según el formato de origen.

Las pruebas se ejecutan sobre copias independientes. Sus modelos entrenados nunca sustituyen la partida. Conservar el entrenamiento no garantiza ausencia de olvido; la evaluación mide locomoción y tala por separado.

## Etapas

| Etapa | Estado |
| --- | --- |
| Caminar y explorar | Implementada; seguir midiendo retención. |
| Monedas, dos tipos de madera, piedra y herramientas | Física, sensores, recetas y misión implementados; aprendizaje en curso. |
| Construir y cultivar | Pendiente. |
| Sociedad y aprendizaje entre agentes | Pendiente: mentes separadas, comunicación e interacción. |

El experimento original de español se conserva en `src/brain`, `data/v1` y `runs/baseline-*`; no controla el mundo. Su [plan original](https://chatgpt.com/space/page_ca90102425588191973c27a34975bfa7) es una referencia histórica.

## Verificar y reproducir

```bash
uv run pytest -q
uv run ruff check src tests
uv run python -m brain.world.forest_experiment --checkpoint runs/world/live.v1-backup.pt --out runs/forest-reproduction
```

El experimento guarda su punto inicial, entrena una copia durante 131.072 turnos y evalúa cinco bosques reservados y cinco terrenos sin árboles, antes y después. Comprueba restauración y continuación exactas. Las misiones de entrenamiento se reinician con cada terreno de ensayo; este ensayo mide habilidades, no la finalización de la misión completa. No carga sus resultados en el servidor.

[Resultados de la etapa anterior](docs/RESULTS.md) son históricos y no describen esta versión. [Evaluación del bosque anterior](docs/FOREST_RESULTS.md) registra resultados de v0.2. [Verificación de recursos y herramientas](docs/RESOURCE_RESULTS.md) describe esta actualización. PPO sigue la [descripción de OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html). Dependencias fijadas en `uv.lock`; sin servicios ni claves externas.

## Código

- `src/brain/world/environment.py`: terreno, física, sensores, misión y señales.
- `src/brain/world/learning.py`: política, PPO, migración y persistencia.
- `src/brain/world/server.py`: entrenamiento continuo y servicio local.
- `src/brain/world/clock.py`: reloj y presupuesto de turnos.
- `src/brain/world/static/`: interfaz Canvas.
- `src/brain/world/forest_experiment.py`: evaluación independiente.
- `tests/test_world.py`, `tests/test_forest.py`, `tests/test_mission.py`: turnos, física, objetivos y reinicios.

## Experimentación autónoma: etapa futura

El usuario quiere que Brote llegue a experimentar por sí mismo. Está planificado ampliar la misma mente con predicción de consecuencias condicionada por acciones, memoria temporal, elección de pruebas que reduzcan incertidumbre y, después, planificación con su modelo aprendido. La predicción deberá probarse en situaciones reservadas e intervenciones; agrupaciones de activaciones o bajo error no demostrarán comprensión causal por sí solos. La curiosidad deberá distinguir lo que puede aprender de variaciones impredecibles. Referencias: [World Models](https://worldmodels.github.io/) y [Curiosity-driven Exploration](https://pathak22.github.io/noreward-rl/). Es una dirección futura: esta versión sigue siendo una política y un crítico entrenados con PPO, sin predictor ni planificación interna.
