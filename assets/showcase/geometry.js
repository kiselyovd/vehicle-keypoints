/** Choose visible geometry independently of which vehicle supplies the detail panel. */
export function objectsForView(objects, selected, view) {
  return view === 'focus' ? objects.slice(selected, selected + 1) : objects;
}

/** Fit the traffic group or one vehicle while keeping the crop inside the source frame. */
export function boundsForView(sample, objects, selected, view) {
  if (view === 'frame' || objects.length === 0) return [0, 0, sample.width, sample.height];
  const visible = objectsForView(objects, selected, view);
  if (visible.length === 0) return [0, 0, sample.width, sample.height];
  const left = Math.min(...visible.map(object => object.bbox[0]));
  const top = Math.min(...visible.map(object => object.bbox[1]));
  const right = Math.max(...visible.map(object => object.bbox[0] + object.bbox[2]));
  const bottom = Math.max(...visible.map(object => object.bbox[1] + object.bbox[3]));
  const margin = Math.max(right - left, bottom - top) * (view === 'focus' ? .3 : .08);
  const x = Math.max(0, left - margin);
  const y = Math.max(0, top - margin);
  return [x, y, Math.min(sample.width, right + margin) - x, Math.min(sample.height, bottom + margin) - y];
}
