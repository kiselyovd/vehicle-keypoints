import { assetUrl, checkboxField, element, log, selectField, sourceLink, startShowcase } from './ui.js';

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

function focusedBounds(sample, object, focus) {
  if (!focus || !object) return [0, 0, sample.width, sample.height];
  const [x, y, width, height] = object.bbox;
  const margin = Math.max(width, height) * .3;
  const left = Math.max(0, x - margin);
  const top = Math.max(0, y - margin);
  return [left, top, Math.min(sample.width, x + width + margin) - left, Math.min(sample.height, y + height + margin) - top];
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

function renderFrame(stage, sample, object, data, state, base) {
  const layerDescription = state.layer === 'prediction' ? 'recorded prediction' : state.layer === 'photo' ? 'original photo' : 'CarFusion ground-truth annotations';
  const svg = svgElement('svg', { viewBox: focusedBounds(sample, object, state.focus).join(' '), role: 'img', 'aria-label': `${sample.title}: ${layerDescription}` });
  svg.append(svgElement('image', { href: assetUrl(sample.image, base), width: sample.width, height: sample.height }));
  if (state.layer !== 'photo' && object) drawGeometry(svg, object, data, state);
  const caption = element('figcaption', '', `${sample.title}. ${state.layer === 'prediction' ? 'Recorded CPU prediction from the published checkpoint.' : state.layer === 'photo' ? 'Original committed CarFusion smoke sample.' : 'Original committed CarFusion smoke sample and its dataset annotation.'}`);
  stage.replaceChildren(svg, caption);
}

function describeObject(details, sample, object, data, state) {
  details.replaceChildren(element('h3', '', state.layer === 'prediction' ? 'Recorded detection' : 'Dataset annotation'));
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
  if (data.prediction_provenance) details.append(element('p', '', 'Predictions are recorded examples on smoke samples. The complete test-set benchmark is documented separately.'));
}

function renderVehicle(root, data, base) {
  const controls = root.querySelector('.sc-controls');
  const stage = root.querySelector('.sc-stage');
  const details = root.querySelector('.sc-details');
  const live = root.querySelector('[role="status"]');
  const best = data.samples.reduce((winner, sample, index) => sample.annotations[0]?.bbox[2]*sample.annotations[0]?.bbox[3] > data.samples[winner].annotations[0]?.bbox[2]*data.samples[winner].annotations[0]?.bbox[3] ? index : winner, 0);
  const state = { sample: best, layer: 'annotation', object: 0, point: -1, focus: true, skeleton: true };
  const sampleField = selectField('Road image', data.samples.map((sample,index)=>[index,sample.title]), 'sample');
  const layerOptions = [['annotation','Dataset annotations'],['photo','Original photo']];
  if (data.prediction_provenance) layerOptions.splice(1,0,['prediction','Recorded model prediction']);
  const layerField = selectField('Layer', layerOptions, 'layer');
  const objectField = selectField('Vehicle', [], 'vehicle');
  const viewField = selectField('View', [['focus','Selected vehicle'],['frame','Full road frame']], 'view');
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
    renderFrame(stage,sample,objects[state.object],data,state,base);
    describeObject(details,sample,objects[state.object],data,state);
    live.textContent=`${objects.length} ${state.layer==='prediction'?'recorded detections':'labelled vehicles'} in this frame.`;
    log('vehicle:update',{sample:sample.id,layer:state.layer,object:state.object,point:state.point});
  };
  sampleField.select.addEventListener('change',()=>{state.sample=Number(sampleField.select.value);update(true);});
  layerField.select.addEventListener('change',()=>{state.layer=layerField.select.value;update(true);});
  objectField.select.addEventListener('change',()=>{state.object=Number(objectField.select.value);update();});
  viewField.select.addEventListener('change',()=>{state.focus=viewField.select.value==='focus';update();});
  pointField.select.addEventListener('change',()=>{state.point=Number(pointField.select.value);update();});
  skeletonField.input.addEventListener('change',()=>{state.skeleton=skeletonField.input.checked;update();});
  update(true);
}

startShowcase('vehicle',renderVehicle,new URL('./data.json',import.meta.url));
