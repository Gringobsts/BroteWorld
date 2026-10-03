'use strict';
const $ = id => document.getElementById(id);
const canvas = $('world'), ctx = canvas.getContext('2d');
const portrait = $('portrait').getContext('2d');
ctx.imageSmoothingEnabled = portrait.imageSmoothingEnabled = false;
let state = null, prevAgent = null, receivedAt = 0, polling = false, showTrail = false, showGrid = false;
let lastSeed = null, first = true;
const tile = 32, nf = new Intl.NumberFormat('es-AR');
function rect(g,x,y,w,h,color){g.fillStyle=color;g.fillRect(Math.round(x),Math.round(y),w,h);}
function noise(x,y,n=0){let v=(x*374761393+y*668265263+n*1274126177)|0;v=(v^(v>>>13))*1274126177;return ((v^(v>>>16))>>>0)/4294967295;}
function sprite(g,x,y,scale=2,phase=0,direction=2){
  g.save();g.translate(Math.round(x),Math.round(y));g.scale(scale,scale);
  rect(g,-5,6,10,2,'#244e3335');
  const leg=phase ? 1 : 0;
  rect(g,-3,3,2,4+leg,'#435964');rect(g,1,3,2,5-leg,'#435964');
  rect(g,-4,7+leg,3,1,'#3c4c4c');rect(g,1,8-leg,3,1,'#3c4c4c');
  rect(g,-4,-3,8,7,'#bd765d');rect(g,-3,-3,6,5,'#e49371');
  rect(g,-5,-2,2,5,'#cc865f');rect(g,3,-2,2,5,'#e5ab7e');
  rect(g,-4,-10,8,7,'#e3b590');rect(g,-3,-8,7,5,'#f2c9a2');
  rect(g,-4,-12,7,3,'#705c46');rect(g,-5,-10,2,4,'#705c46');rect(g,2,-11,3,3,'#705c46');
  rect(g,-3,-12,5,1,'#897256');
  if(direction===0){rect(g,-4,-10,8,5,'#705c46');rect(g,-2,-4,4,1,'#deaa84');}
  else{rect(g,direction===3?-3:-1,-7,1,1,'#554e40');rect(g,direction===1?3:2,-7,1,1,'#554e40');rect(g,0,-4,2,1,'#ce9373');}
  rect(g,-3,-2,2,4,'#d08061');g.restore();
}
function ground(x,y,t,time){
  const px=x*tile,py=y*tile,n=noise(x+state.world.viewport.x,y+state.world.viewport.y);
  if(t===2){
    rect(ctx,px,py,32,32,['#a4ccd0','#a0c8cc','#a8ced0'][Math.floor(n*3)]);
    const wave=(Math.floor(time/1400)+x+y)%3;
    rect(ctx,px+4+(wave*3),py+9,8,2,'#bbdddd');rect(ctx,px+17,py+23,9,1,'#8fbcc2');
    return;
  }
  const grass=['#b2cc91','#b7ce96','#afc98d','#b9cf95'];
  rect(ctx,px,py,32,32,t===1?(['#e2d6ae','#e5d9b5','#dfd2a8'][Math.floor(n*3)]):grass[Math.floor(n*4)]);
  if(t===1){rect(ctx,px,py+31,32,1,'#cdbf942c');rect(ctx,px+31,py,1,32,'#cdbf942c');rect(ctx,px+7,py+7,2,1,'#c9bc94');rect(ctx,px+22,py+23,3,1,'#f2e8cb');}
  else{
    for(let i=0;i<3;i++){const a=3+Math.floor(noise(x+state.world.viewport.x,y+state.world.viewport.y,i+2)*24),b=4+Math.floor(noise(y+state.world.viewport.y,x+state.world.viewport.x,i+9)*22);rect(ctx,px+a,py+b,1,3,'#97b67a');rect(ctx,px+a+1,py+b+2,2,1,'#a0be80');}
    if(n>.78&&t===0){rect(ctx,px+21,py+20,1,4,'#86a46a');rect(ctx,px+19,py+19,5,2,'#f3e3a0');rect(ctx,px+21,py+18,1,4,'#f8e8b3');}
    if(n<.1&&t===0){rect(ctx,px+10,py+12,3,2,'#f0ead4');rect(ctx,px+11,py+11,1,4,'#f0ead4');}
  }
  if(state){
    for(const [dx,dy] of [[0,-1],[1,0],[0,1],[-1,0]]){
      if(state.world.tiles[y+dy]?.[x+dx]===2){
        if(dy===-1){rect(ctx,px,py,32,3,'#759962');rect(ctx,px+2,py+3,28,2,'#a6bf7c');}
        if(dy===1){rect(ctx,px,py+28,32,2,'#a0b97a');rect(ctx,px,py+30,32,2,'#718e5e');}
        if(dx===1)rect(ctx,px+30,py,2,32,'#829e66');
        if(dx===-1)rect(ctx,px,py,2,32,'#88a76b');
      }
    }
  }
  if(t===3){
    rect(ctx,px+7,py+23,20,4,'#70876755');rect(ctx,px+7,py+12,20,11,'#8d9c8c');rect(ctx,px+10,py+8,14,17,'#8d9c8c');
    rect(ctx,px+9,py+11,16,10,'#adb7a3');rect(ctx,px+12,py+7,10,14,'#adb7a3');rect(ctx,px+12,py+8,9,3,'#c1c8b4');rect(ctx,px+9,py+12,3,6,'#c1c8b4');rect(ctx,px+22,py+15,3,8,'#798c7c');rect(ctx,px+13,py+23,9,2,'#788b77');
  }
}
function render(time){
  if(!state){rect(ctx,0,0,canvas.width,canvas.height,'#c4d9b5');requestAnimationFrame(render);return;}
  for(let y=0;y<state.world.viewport.height;y++)for(let x=0;x<state.world.viewport.width;x++)ground(x,y,state.world.tiles[y][x],time);
  for(let y=0;y<state.world.viewport.height;y++)for(let x=0;x<state.world.viewport.width;x++){
    const px=x*tile,py=y*tile,t=state.world.tiles[y][x];
    if(t===4||t===6){
      rect(ctx,px+5,py+25,24,5,'#42653a44');rect(ctx,px+13,py+13,7,16,'#85633e');rect(ctx,px+14,py+16,2,11,'#b18b54');
      rect(ctx,px+4,py+6,25,15,'#4e7745');rect(ctx,px+8,py+1,18,23,'#4e7745');rect(ctx,px+5,py+5,19,12,'#6c934f');rect(ctx,px+10,py+1,12,14,'#7ea25a');rect(ctx,px+9,py+4,8,3,'#93b66b');
      if(t===6){rect(ctx,px+12,py+14,9,15,'#6b493a');rect(ctx,px+3,py+7,26,14,'#385c4b');rect(ctx,px+7,py+2,19,21,'#426d53');rect(ctx,px+10,py+4,12,9,'#588362');rect(ctx,px+17,py+17,7,4,'#2b5245');}
      const hp=state.world.tree_health[y][x];if(hp<3){rect(ctx,px+7,py+28,18,3,'#68593e');rect(ctx,px+7,py+28,6*hp,3,'#edd694');}
    }else if(t===5){rect(ctx,px+9,py+19,15,9,'#927047');rect(ctx,px+8,py+18,17,5,'#c59e66');rect(ctx,px+12,py+19,9,2,'#e1bf88');}
    if(t===3&&state.world.rock_health[y][x]<3){rect(ctx,px+7,py+28,18,3,'#68593e');rect(ctx,px+7,py+28,6*state.world.rock_health[y][x],3,'#d7e1e1');}
    if(state.world.stones[y][x]){rect(ctx,px+4,py+23,8,4,'#737e7a');rect(ctx,px+5,py+21,6,3,'#c6cfca');rect(ctx,px+20,py+25,6,3,'#8b9790');rect(ctx,px+21,py+24,3,1,'#dae1d9');}
    if(state.world.coins[y][x]){rect(ctx,px+12,py+13,9,11,'#a7802f');rect(ctx,px+10,py+12,9,11,'#edbd4f');rect(ctx,px+12,py+10,5,15,'#edbd4f');rect(ctx,px+12,py+12,2,8,'#fff0a0');}
  }
  if(showTrail){for(let y=0;y<state.world.viewport.height;y++)for(let x=0;x<state.world.viewport.width;x++){
    const visits=state.world.visits[y][x];if(visits>0&&state.world.tiles[y][x]<2){ctx.fillStyle=`rgba(107,137,82,${Math.min(.42,.12+Math.log1p(visits)*.04)})`;ctx.fillRect(x*tile+5,y*tile+5,22,22);rect(ctx,x*tile+12,y*tile+15,3,5,'#68864e88');rect(ctx,x*tile+18,y*tile+11,3,5,'#68864e88');}
  }}
  if(showGrid){ctx.strokeStyle='#7c955a35';ctx.lineWidth=1;ctx.beginPath();for(let x=0;x<=state.world.viewport.width;x++){ctx.moveTo(x*tile+.5,0);ctx.lineTo(x*tile+.5,canvas.height);}for(let y=0;y<=state.world.viewport.height;y++){ctx.moveTo(0,y*tile+.5);ctx.lineTo(canvas.width,y*tile+.5);}ctx.stroke();}
  const agent=state.agent;
  let x=agent.x,y=agent.y;
  const adjacent=prevAgent&&Math.abs(agent.x-prevAgent.x)+Math.abs(agent.y-prevAgent.y)===1;
  const duration=Math.min(200,900/state.speed);
  const fraction=Math.min(1,(performance.now()-receivedAt)/duration);
  if(adjacent&&state.running&&state.speed<=16){x=prevAgent.x+(x-prevAgent.x)*fraction;y=prevAgent.y+(y-prevAgent.y)*fraction;}
  const px=(x-state.world.viewport.x)*tile+16,py=(y-state.world.viewport.y)*tile+17;
  // Selection ring and avatar. At high speeds show sampled positions, never interpolate invented turns.
  ctx.strokeStyle='#f8ffe8';ctx.lineWidth=2;ctx.strokeRect(Math.round(px-13),Math.round(py-11),26,27);
  sprite(ctx,px,py,2,state.running&&fraction<1?Math.floor(time/110)%2:0,agent.action<4?agent.action:agent.action<9&&agent.action>=5?agent.action-5:agent.action>=11&&agent.action<15?agent.action-11:2);
  if(agent.target){
    const tx=(agent.target[0]-state.world.viewport.x)*tile+16,ty=(agent.target[1]-state.world.viewport.y)*tile+16;
    ctx.strokeStyle=agent.reward>0?'#ffe9a0':agent.reward<0?'#b75b4b':'#bac3b2';ctx.lineWidth=2;ctx.strokeRect(tx-12,ty-12,24,24);
    const chopping=agent.action>=5&&agent.action<9,mining=agent.action>=11&&agent.action<15;
    if(chopping&&agent.inventory.axe){rect(ctx,px+9,py-7,3,16,'#85633e');rect(ctx,px+9,py-10,10,6,'#dae0d8');}
    else if(mining&&agent.inventory.pickaxe){rect(ctx,px+9,py-7,3,16,'#85633e');rect(ctx,px+4,py-10,15,3,'#bfcbd0');rect(ctx,px+3,py-7,3,4,'#bfcbd0');rect(ctx,px+18,py-7,3,4,'#bfcbd0');}
    else if(chopping||agent.action===15){rect(ctx,px+9,py-2,6,5,'#f2c9a2');}
  }
  const labelW=50;rect(ctx,px-labelW/2,py-43,labelW,15,'#fafdefed');rect(ctx,px-2,py-28,4,3,'#fafdefed');
  ctx.fillStyle='#697957';ctx.textAlign='center';ctx.font='9px sans-serif';ctx.fillText('Brote',Math.round(px),Math.round(py-33));
  requestAnimationFrame(render);
}
function formatClock(seconds){const days=Math.floor(seconds/86400),h=Math.floor(seconds/3600)%24,m=Math.floor(seconds/60)%60,s=seconds%60;return `${days?days+' d · ':''}${[h,m,s].map(x=>String(x).padStart(2,'0')).join(':')}`;}
function formatBytes(bytes){if(!bytes)return 'Sin guardar';const units=['B','kB','MB','GB'];let i=0;while(bytes>=1000&&i<units.length-1){bytes/=1000;i++;}return nf.format(Math.round(bytes*10)/10)+' '+units[i];}
function setState(next){
  if(state&&next.world.seed===lastSeed)prevAgent=state.agent;else prevAgent=null;
  receivedAt=performance.now();state=next;lastSeed=next.world.seed;
  $('offline').hidden=true;
  $('world-status').textContent=next.error?'Necesita atención':next.running?'El mundo está vivo':'El mundo está en pausa';
  $('training').textContent=next.running?'ACTIVO':'EN PAUSA';
  $('pause').textContent=next.running?'Ⅱ':'▶';$('pause').setAttribute('aria-label',next.running?'Pausar simulación':'Reanudar simulación');
  $('step').disabled=next.running;
  if(document.activeElement!==$('speed'))$('speed').value=String(next.speed);
  $('turn').textContent='Turno '+nf.format(next.clock.turn);
  $('clock').textContent=formatClock(next.clock.elapsed_seconds)+' del mundo';
  $('real-speed').textContent=nf.format(Math.round(next.effective_speed))+' turnos/s reales';
  $('real-speed').title='Solicitados: '+nf.format(next.speed)+' turnos/s. Todos los turnos ejecutan una acción y registran experiencia.';
  $('updates').textContent=nf.format(next.learning.updates);
  $('connections').textContent=nf.format(next.learning.connections);
  $('parameters').textContent=nf.format(next.learning.parameters);
  $('model-size').textContent=formatBytes(next.learning.model_bytes);
  $('model-size').title='Tamaño de sus parámetros en memoria. No incluye el mundo ni el optimizador.';
  $('checkpoint-size').textContent=formatBytes(next.learning.checkpoint_bytes);
  $('checkpoint-size').title='Archivo que conserva la red, el optimizador, experiencias pendientes y mundo.';
  $('mind-detail').textContent=next.learning.inputs+' entradas · '+next.learning.actions+' acciones. Aprende cambiando el valor de sus conexiones; la cantidad no crece sola.';
  $('move-rate').textContent=Math.round(next.metrics.move_rate*100)+' %';
  $('collisions').textContent=Math.round(next.metrics.collision_rate*100)+' %';
  $('coverage').textContent=nf.format(next.metrics.visited);
  $('visited').textContent='Un billón de bloques · cámara de 24 × 18';
  $('buffer').textContent=next.learning.buffer+' / '+next.learning.rollout;
  $('buffer-bar').style.width=(100*next.learning.buffer/next.learning.rollout)+'%';
  $('last-action').textContent=next.agent.action_name.charAt(0).toUpperCase()+next.agent.action_name.slice(1);
  const signals={inicio:'Explorando con los sentidos de alrededor.',avance:'Paso válido: sin premio por caminar.',obstaculo:'Movimiento bloqueado: señal negativa.',espera:'Esperar consume un turno.',sin_arbol:'No hay un árbol para talar ahí.',golpe:'Golpe válido: el árbol perdió resistencia.',arbol_talado:'Árbol blando talado. Ganó 3 de madera blanda.',moneda:'Moneda recogida: recompensa positiva.',objetivo_completo:'¡Misión completada! Premio recibido y una nueva misión en marcha.',plazo_sin_cumplir:'Pasaron 50.000 turnos sin completar las cuatro metas.',falta_hacha:'El árbol duro necesita un hacha.',falta_pico:'La roca grande necesita un pico.',sin_roca:'No hay una roca grande en esa dirección.',materiales_insuficientes:'Faltan materiales para fabricar la herramienta.',herramienta_existente:'Ya tiene esa herramienta.',hacha_fabricada:'Fabricó un hacha. Puede talar árboles duros.',pico_fabricado:'Fabricó un pico. Puede extraer rocas grandes.',arbol_duro_talado:'Ganó 3 de madera dura.',golpe_roca:'La roca perdió resistencia.',roca_extraida:'Roca extraída. Ganó 3 de piedra.',piedra_suelta:'Recogió una piedra de su casilla.',sin_piedra_suelta:'No hay una piedra suelta en su casilla.'};
  $('action').textContent=(signals[next.agent.signal]||next.agent.signal)+(next.agent.reward_reduced?' Cuota cumplida: este recurso ya no da premio.':'');
  $('reward').textContent=(next.agent.reward>0?'+':'')+next.agent.reward.toFixed(2);
  $('reward').style.color=next.agent.reward<0?'#ab6551':next.agent.reward>0?'#657c49':'#7c8276';
  $('reward').title=Object.entries(next.agent.reward_components).map(([k,v])=>k+': '+v.toFixed(2)).join(' · ');
  $('coins').textContent=nf.format(next.goal.coins)+' / '+nf.format(next.goal.coin_target);
  $('wood').textContent=nf.format(next.goal.wood)+' / '+nf.format(next.goal.wood_target);
  $('coins-bar').style.width=Math.min(100,100*next.goal.coins/next.goal.coin_target)+'%';
  $('wood-bar').style.width=Math.min(100,100*next.goal.wood/next.goal.wood_target)+'%';
  for(const resource of ['stone','hard_wood']){const id=resource.replace('_','-');$(id).textContent=nf.format(next.goal[resource])+' / '+nf.format(next.goal[resource+'_target']);$(id+'-bar').style.width=Math.min(100,100*next.goal[resource]/next.goal[resource+'_target'])+'%';}
  $('goal-status').textContent='N.º '+nf.format(next.goal.round)+' · EN CURSO';
  $('missions-completed').textContent=nf.format(next.goal.completed_missions);
  $('total-inventory').textContent=nf.format(next.agent.inventory.coins)+' monedas · '+nf.format(next.agent.inventory.soft_wood)+' blanda · '+nf.format(next.agent.inventory.stone)+' piedra · '+nf.format(next.agent.inventory.hard_wood)+' dura';
  $('axe-state').textContent=next.agent.inventory.axe?'Disponible':'Por fabricar';
  $('pickaxe-state').textContent=next.agent.inventory.pickaxe?'Disponible':'Por fabricar';
  $('deadline').textContent='Señal '+next.goal.penalty+' en '+nf.format(next.goal.turns_until_penalty)+' turnos si faltan metas.';
  $('coordinates').textContent='Brote · '+nf.format(next.agent.x)+', '+nf.format(next.agent.y);
  if(next.error)$('action').textContent=next.error;
  const history=next.learning.history.slice(-60);
  if(history.length>1){const points=history.map((r,i)=>[i*300/(history.length-1),65-r.move_rate*60]);const d=points.map(([x,y],i)=>(i?'L':'M')+x.toFixed(1)+','+y.toFixed(1)).join(' ');$('chart-line').setAttribute('d',d);$('chart-area').setAttribute('d',d+' L300,70 L0,70 Z');}
  $('events').replaceChildren(...next.events.slice(0,3).map(e=>{const div=document.createElement('div');div.className='event';const dot=document.createElement('i'),text=document.createElement('div'),small=document.createElement('small');text.textContent=e.text;small.textContent='Turno '+nf.format(e.step);text.append(small);div.append(dot,text);return div;}));
  if(next.saved_at)$('saved').textContent='Guardado '+new Date(next.saved_at*1000).toLocaleTimeString('es-AR',{hour:'2-digit',minute:'2-digit',second:'2-digit'})+' · automático';
  if(first){portrait.clearRect(0,0,72,72);sprite(portrait,36,45,3,0,2);first=false;}
}
let toastTimer;
function toast(message){$('toast').textContent=message;$('toast').classList.add('show');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').classList.remove('show'),2600);}
async function control(action,value){try{const r=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action,value})});const next=await r.json();if(!r.ok)throw Error(next.error);setState(next);return true;}catch(e){toast(e.message);return false;}}
async function poll(){if(polling)return;polling=true;try{const r=await fetch('/api/state');if(!r.ok)throw Error('No disponible');setState(await r.json());}catch(e){$('offline').hidden=false;$('world-status').textContent='Sin conexión';}finally{polling=false;}}
$('pause').addEventListener('click',()=>state&&control('running',!state.running));
$('step').addEventListener('click',()=>control('step'));
$('save').addEventListener('click',async()=>{if(await control('save'))toast('Mundo y conexiones guardados.');});
$('terrain').addEventListener('click',async()=>{if(await control('terrain'))toast('Otro terreno, la misma mente.');});
$('speed').addEventListener('change',()=>control('speed',Number($('speed').value)));
$('trail').addEventListener('click',()=>{showTrail=!showTrail;$('trail').setAttribute('aria-pressed',String(showTrail));});
$('grid').addEventListener('click',()=>{showGrid=!showGrid;$('grid').setAttribute('aria-pressed',String(showGrid));});
poll();setInterval(poll,180);requestAnimationFrame(render);
