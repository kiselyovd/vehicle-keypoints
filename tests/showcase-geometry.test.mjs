import assert from 'node:assert/strict';
import test from 'node:test';
import { boundsForView, objectsForView } from '../docs/assets/showcase/geometry.js';

const sample = { width: 1920, height: 1080 };
const objects = [{ bbox: [0, 0, 120, 100] }, { bbox: [400, 300, 220, 110] }, { bbox: [1800, 1000, 120, 80] }];

test('traffic and full-frame views draw every vehicle regardless of detail selection', () => {
  for (const view of ['all', 'frame']) {
    assert.deepEqual(objectsForView(objects, 1, view), objects);
  }
});

test('only the explicit close-up draws one selected vehicle', () => {
  assert.deepEqual(objectsForView(objects, 1, 'focus'), [objects[1]]);
  assert.deepEqual(objectsForView([], 0, 'focus'), []);
});

test('scene and close-up bounds remain inside the image at its corners', () => {
  for (const view of ['all', 'frame', 'focus']) {
    for (const selected of [0, 1, 2]) {
      const [x, y, width, height] = boundsForView(sample, objects, selected, view);
      assert.ok(x >= 0 && y >= 0 && width > 0 && height > 0);
      assert.ok(x + width <= sample.width && y + height <= sample.height);
    }
  }
  assert.deepEqual(boundsForView(sample, [], 0, 'all'), [0, 0, 1920, 1080]);
});
