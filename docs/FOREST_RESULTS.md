# Bosque, misión y continuidad · 2 de octubre de 2026

## Qué se verificó

41 pruebas automáticas pasan: una acción por turno, actualización real de pesos, terreno disperso de un billón de bloques, sensores locales, tala, inventario, rebrote a 100.000 turnos, monedas, premio único, penalización a 50.000 turnos y sus múltiplos, persistencia y migración de la mente anterior. También se verificaron los datos del panel de conexiones y el servidor HTTP. La sintaxis de la interfaz pasa la comprobación de JavaScript.

La migración conserva cada peso previo y los momentos de Adam en sus dimensiones originales. Si quedan experiencias pendientes, primero se aprenden con la política anterior. Las dimensiones nuevas se agregan sin reiniciar las anteriores; se respalda el archivo v1. El muestreo cambia para permitir explorar las nuevas acciones. Esto conserva el estado aprendido, pero no garantiza ausencia de olvido futuro.

## Aprendizaje: resultados mixtos

Un piloto de desarrollo de 131.072 turnos mostró 373,5 de madera por 2.048 acciones, en promedio, sobre dos mapas de validación (701 y 702). No fue una prueba final y otros pilotos no aprendieron tala. Una política de locomoción muy consolidada puede seguir caminando y no descubrir cómo completar los tres golpes.

Después se hizo una prueba independiente, partiendo de otra copia de la partida en vivo en el turno 2.797.895. Se entrenó durante 131.072 acciones en semillas 17–48 y se evaluaron sin entrenamiento cinco bosques nuevos (1201–1205, 2.048 acciones cada uno) y cinco terrenos sin árboles (1301–1305, 1.024 acciones cada uno).

| Medida media | Antes | Después |
| --- | ---: | ---: |
| Madera por bosque reservado | 1,2 | 0 |
| Monedas por bosque reservado | 31,8 | 67 |
| Movimientos válidos / intentos de movimiento, sin árboles | 99,82 % | 99,88 % |
| Turnos con movimiento válido, sin árboles | 85,47 % | 94,69 % |

**No se demostró aprendizaje robusto de tala.** La copia final mejoró la recolección de monedas y conservó locomoción, pero prefirió desplazarse. La diferencia con el piloto demuestra variabilidad entre puntos de partida; no debe presentarse el piloto como una garantía. La física y el aprendizaje continuo están implementados, pero alcanzar la misión completa requiere más trabajo de entrenamiento y evaluación. No se evaluó inteligencia social.

La evaluación después de guardar/restaurar fue idéntica y continuar 65 acciones —incluida una actualización— produjo acciones, mundo y pesos idénticos. Las pruebas también verifican restauración en otro proceso. El ensayo cambia de terreno cada 4.096 acciones y reinicia sus misiones de prueba; no acredita haber alcanzado 100 monedas y 10.000 de madera, ni haber aprendido a responder al plazo de 50.000 turnos. Es una sola corrida, sin intervalos de confianza.

## Separación de partida y evaluación

Los checkpoints de pruebas se guardan fuera de `runs/world/live.pt`. **No se reemplaza a Brote por la copia que rinda mejor.** La partida sigue con su propia mente y experiencia. `initial.pt`, `trained.pt` y `report.json` permiten auditar la corrida independiente; se guardan en una carpeta propia bajo `runs/`.

La red actual tiene 7.232 conexiones, 7.370 parámetros y 29.480 bytes de parámetros float32. El guardado completo incluye además optimizador, experiencia, mundo e inventario; su tamaño cambia con la exploración. La cantidad de conexiones es fija en esta arquitectura: aprender modifica sus valores, no agrega conexiones automáticamente.

La vista no pudo inspeccionarse con el navegador automatizado porque ese mecanismo no logró verificar su política de acceso. No se presenta como comprobación visual completa; el usuario puede revisar la interfaz del servidor local.
