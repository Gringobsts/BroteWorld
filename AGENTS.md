# Brain

El objetivo actual es un mundo de píxeles con agentes que aprendan continuamente desde pesos aleatorios. Primero caminar, luego tala, construcción, cultivo y sociedad. El usuario reemplazó el objetivo original de español; ese código se conserva como referencia.

- Una acción por agente por turno. Intentos fallidos y espera también consumen la acción.
- Un turno equivale a un segundo simulado. Separar reloj, velocidad real y renderizado.
- La tasa solicitada llega a 10.000 turnos/s. Mostrar la efectiva; no saltar acciones, turnos ni aprendizaje para aparentar velocidad.
- Aprendizaje continuo por defecto mientras el servidor esté encendido. Pausar el mundo pausa su experiencia; reanudar conserva la mente y el optimizador.
- No introducir modelos preentrenados ni reglas que elijan acciones por el agente. Documentar física, sensores y recompensa diseñados.
- Conservar pesos, optimizador, aleatoriedad y experiencias pendientes al reiniciar.
- Evaluar en mundos y generadores aleatorios independientes. No entrenar con el examen.
- Registrar resultados negativos y semillas. Locomoción no equivale a biomecánica ni inteligencia social.
- Ejecutar `uv run pytest -q` y `uv run ruff check src tests`.
- Guardar experimentos nuevos en carpetas distintas. La partida usa `runs/world/live.pt`.
- Conservar `data/v1` y la referencia previa de español sin mezclarlos con la política del mundo.

## Continuidad de la mente y misión actual

- La misma mente de la partida aprende cada habilidad nueva. Nunca reemplazar `live.pt` por un modelo de evaluación o reiniciar pesos al ampliar sentidos/acciones.
- Migraciones deben conservar pesos existentes, momentos del optimizador, aleatoriedad y experiencia pendiente; respaldar el formato anterior.
- Mundo de 1.000.000 por 1.000.000: generación por coordenadas y almacenamiento disperso. Nunca iterar/renderizar el área completa.
- Tala: árboles blandos a mano; duros con hacha. Tres golpes, tres de su madera, rebrote a 100.000 turnos; no crecer sobre el agente. Rocas grandes bloquean: tres golpes de pico, tres piedras, sin regeneración. Piedras pequeñas transitables: moverse y recoger en la casilla actual son acciones separadas, nunca recogida automática.
- Misión persistente: 10.000 monedas y 100.000 de cada recurso (madera blanda, piedra, madera dura). Premio máximo una vez por misión al completar las cuatro metas; renovar automáticamente en el mismo turno con progreso nuevo, nuevas monedas y plazo de 50.000 turnos. Conservar inventario acumulado y mente. Penalización cada 50.000 turnos mientras falte alguna.
- Cambiar el terreno conserva inventario y reloj de misión. El entrenamiento continúa después de completarla.
- No añadir castigo general por turno sin una instrucción posterior del usuario. La pregunta sobre su conveniencia no autorizó ese cambio.

- Hacha y pico cuestan cada uno 2 de madera blanda y 3 piedras. Fabricar gasta un turno. Herramientas sin desgaste; duplicados fallan sin consumo. El progreso de misión cuenta recolección bruta, sin disminuir al fabricar.
- Guardados v2 conservan toda la madera como blanda y su hacha inicial. Ampliar las 39/9 dimensiones a 72/16 conservando pesos y momentos y aprendiendo las experiencias pendientes con las reglas anteriores.
- Experimentación autónoma es una etapa futura pedida por el usuario: predictor, memoria, curiosidad y planificación evaluables. No afirmar que ya existen ni que implican comprensión causal automáticamente.
