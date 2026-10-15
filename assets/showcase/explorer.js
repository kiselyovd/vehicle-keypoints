import { assetUrl, checkboxField, element, log, selectField, sourceLink, startShowcase } from './ui.js';
import { boundsForView, objectsForView } from './geometry.js';

const SVG = 'http://www.w3.org/2000/svg';

function svgElement(tag, attributes) {
  const node = document.createElementNS(SVG, tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, String(value));
  return node;
}

function objectsFor(sample, layer) {
  return layer === 'prediction' ? sample.predictions || [] : sample.annotations;
}

function visiblePoint(points, index, prediction) {
  const x = points[index * 3];
  const y = points[index * 3 + 1];
  const flag = points[index * 3 + 2];
  return (prediction ? flag > 0.01 : flag > 0) && (x !== 0 || y !== 0);
}

function drawGeometry(svg, object, data, state) {
  const prediction = state.layer === 'prediction';
  const points = object.keypoints;
  const [x, y, width, height] = object.bbox;
  svg.append(svgElement('rect', { x, y, width, height, class: 'sc-box' }));
  if (state.skeleton) {
    for (const [a, b] of data.skeleton) {
      if (!visiblePoint(points, a, prediction) || !visiblePoint(points, b, prediction)) continue;
      svg.append(svgElement('line', { x1: points[a*3], y1: points[a*3+1], x2: points[b*3], y2: points[b*3+1], class: 'sc-skeleton' }));
    }
  }
  for (let index = 0; index < data.keypoints.length; index++) {
    if (!visiblePoint(points, index, prediction)) continue;
    const soft = prediction ? points[index*3+2] < .5 : points[index*3+2] === 1;
    const dim = state.point !== -1 && state.point !== index;
    const classes = ['sc-point', soft ? 'sc-point-soft' : '', dim ? 'sc-point-dim' : ''].join(' ');
    const circle = svgElement('circle', { cx: points[index*3], cy: points[index*3+1], r: Math.max(width*.013, 2), class: classes });
    circle.append(svgElement('title', {}));
    circle.firstChild.textContent = data.keypoints[index].replaceAll('_', ' ');
    svg.append(circle);
  }
}

function renderFrame(stage, sample, objects, data, state, base) {
  const layerDescription = state.layer === 'prediction' ? 'recorded prediction' : state.layer === 'photo' ? 'original photo' : 'CarFusion ground-truth annotations';
  const bounds = boundsForView(sample, objects, state.object, state.view);
  const svg = svgElement('svg', { viewBox: bounds.join(' '), role: 'img', 'aria-label': `${sample.title}: ${layerDescription}` });
  svg.style.aspectRatio = state.view === 'focus' ? '16 / 10' : '16 / 9';
  svg.append(svgElement('image', { href: assetUrl(sample.image, base), width: sample.width, height: sample.height }));
  if (state.layer !== 'photo') {
    for (const object of objectsForView(objects, state.object, state.view)) {
      const index = objects.indexOf(object);
      const group = svgElement('g', { 'data-vehicle': index + 1, class: index === state.object ? 'sc-vehicle-selected' : 'sc-vehicle' });
      drawGeometry(group, object, data, state);
      const fontSize = Math.max(18, bounds[2] * .026);
      const number = svgElement('text', { x: object.bbox[0] + 7, y: object.bbox[1] + fontSize + 4, 'font-size': fontSize, class: 'sc-vehicle-number' });
      number.textContent = String(index + 1);
      group.append(number);
      svg.append(group);
    }
  }
  const caption = element('figcaption', '', `${sample.title}. ${state.layer === 'prediction' ? 'Recorded CPU predictions from the published checkpoint.' : state.layer === 'photo' ? 'Original CarFusion source photo.' : `${sample.annotations.length} vehicles with original CarFusion dataset annotations.`}`);
  stage.replaceChildren(svg, caption);
}

function describeObject(details, sample, object, data, state) {
  details.replaceChildren(element('h3', '', `${state.layer === 'prediction' ? 'Recorded detection' : 'Dataset annotation'} · vehicle ${state.object + 1}`));
  if (state.view !== 'focus' && state.layer !== 'photo') details.append(element('p', '', 'All vehicles in this layer are drawn. Numbers match the vehicle selector; the selected box has a thicker outline.'));
  if (!object) { details.append(element('p', '', 'No object was recorded for this layer. Use the full frame or another example.')); return; }
  const labelled = data.keypoints.filter((_, index) => visiblePoint(object.keypoints, index, state.layer === 'prediction')).length;
  details.append(element('p', '', `${labelled} keypoints with coordinates out of 14 slots.`));
  if (object.score !== undefined) details.append(element('p', '', `Box confidence ${(object.score*100).toFixed(1)}%.`));
  if (state.point >= 0) {
    const index = state.point;
    details.append(element('h3', '', data.keypoints[index].replaceAll('_', ' ')));
    if (visiblePoint(object.keypoints, index, state.layer === 'prediction')) {
      details.append(element('p', '', `x ${object.keypoints[index*3].toFixed(1)}, y ${object.keypoints[index*3+1].toFixed(1)}`));
      details.append(element('p', '', state.layer === 'prediction' ? `Keypoint score ${object.keypoints[index*3+2].toFixed(3)}` : object.keypoints[index*3+2] === 2 ? 'Visible in the dataset annotation.' : 'Self-occluded in the dataset annotation.'));
    } else details.append(element('p', '', 'This point has no labelled coordinates on the selected object.'));
  }
  details.append(sourceLink(data.source));
  if (data.prediction_provenance) details.append(element('p', '', 'These scenes were selected for illustration. The complete test-set benchmark is documented separately.'));
}

function renderVehicle(root, data, base) {
  const controls = root.querySelector('.sc-controls');
  const stage = root.querySelector('.sc-stage');
  const details = root.querySelector('.sc-details');
  const live = root.querySelector('[role="status"]');
  const preferred = data.samples.findIndex(sample => sample.id === data.default_sample);
  const best = preferred >= 0 ? preferred : 0;
  const state = { sample: best, layer: 'annotation', object: 0, point: -1, view: 'all', skeleton: true };
  const sampleField = selectField('Road image', data.samples.map((sample,index)=>[index,`${sample.title} · ${sample.annotations.length} vehicles`]), 'sample');
  const layerOptions = [['annotation','Dataset annotations'],['photo','Original photo']];
  if (data.prediction_provenance) layerOptions.splice(1,0,['prediction','Recorded model prediction']);
  const layerField = selectField('Layer', layerOptions, 'layer');
  const objectField = selectField('Vehicle', [], 'vehicle');
  const viewField = selectField('View', [['all','All vehicles'],['focus','Selected vehicle'],['frame','Full road frame']], 'view');
  const pointField = selectField('Keypoint', [[-1,'All keypoints'], ...data.keypoints.map((point,index)=>[index,point.replaceAll('_',' ')])], 'keypoint');
  const skeletonField = checkboxField('Show skeleton', 'skeleton');
  sampleField.select.value = String(best);
  controls.replaceChildren(...[sampleField,layerField,objectField,viewField,pointField,skeletonField].map(field=>field.wrapper));

  const update = (resetObject = false) => {
    const sample = data.samples[state.sample];
    const objects = objectsFor(sample,state.layer);
    if (resetObject) {
      state.object=0;
      objectField.select.replaceChildren(...objects.map((object,index)=>{const option=element('option','',`${index+1} · ${state.layer==='prediction'?'detection':'annotation'} ${object.id || ''}`);option.value=String(index);return option;}));
    }
    renderFrame(stage,sample,objects,data,state,base);
    describeObject(details,sample,objects[state.object],data,state);
    live.textContent=`${objects.length} ${state.layer==='prediction'?'recorded detections':'labelled vehicles'} in this frame. ${state.layer === 'photo' ? 'Original photo without overlays.' : state.view === 'focus' ? 'Showing the selected vehicle.' : 'Showing all vehicles in this layer.'}`;
    log('vehicle:update',{sample:sample.id,layer:state.layer,object:state.object,point:state.point,view:state.view,drawn:state.layer === 'photo' ? 0 : objectsForView(objects,state.object,state.view).length});
  };
  sampleField.select.addEventListener('change',()=>{state.sample=Number(sampleField.select.value);update(true);});
  layerField.select.addEventListener('change',()=>{state.layer=layerField.select.value;update(true);});
  objectField.select.addEventListener('change',()=>{state.object=Number(objectField.select.value);update();});
  viewField.select.addEventListener('change',()=>{state.view=viewField.select.value;update();});
  pointField.select.addEventListener('change',()=>{state.point=Number(pointField.select.value);update();});
  skeletonField.input.addEventListener('change',()=>{state.skeleton=skeletonField.input.checked;update();});
  update(true);
}

startShowcase('vehicle',renderVehicle,new URL('./data.json?v=multi-vehicle-2',import.meta.url));
