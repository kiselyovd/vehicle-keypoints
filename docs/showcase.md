# Vehicle Keypoints

Explore real CarFusion annotations and recorded predictions from the published YOLO checkpoint. Choose a road image, focus on a car, then inspect an individual landmark.

<section class="sc-explorer" data-showcase="vehicle" data-owner="kiselyovd" aria-label="Vehicle Keypoints interactive showcase">
  <div class="sc-controls"></div>
  <div class="sc-workspace">
    <figure class="sc-stage"><img src="../assets/showcase/samples/car_butler1__16_04881.jpg" alt="Source example for Vehicle Keypoints"><figcaption>Explore real CarFusion annotations and recorded predictions from the published YOLO checkpoint. Choose a road image, focus on a car, then inspect an individual landmark.</figcaption></figure>
    <aside class="sc-details"><h3>Explore the example</h3><p>Use the controls to inspect this project's published artifacts.</p></aside>
  </div>
  <p class="sc-live" role="status">The source example is available above; interactive controls load when JavaScript is enabled.</p>
  
</section>

## What the layers mean

- **Dataset annotations:** the original 14-point CarFusion schema. Filled points are visible; hollow points are self-occluded.
- **Recorded model prediction:** locally recorded outputs on these six committed smoke images. Point scores and box confidence come from the checkpoint.
- **Original photo:** the same source pixels with no annotation layer.

The model revision, checkpoint hash, image hashes, CPU configuration, and sample provenance are stored in the structured data. These examples are separate from the full CarFusion benchmark of 12,761 images. [Evaluation and limitations](usage-guide.md#results).

CarFusion retains its own research-use terms. The UI does not alter the original images or dataset labels.

## Sources and reproduction

The explorer uses published artifacts from [kiselyovd/vehicle-keypoints](https://github.com/kiselyovd/vehicle-keypoints). [Inspect the pinned source](https://github.com/kiselyovd/vehicle-keypoints/blob/f057d0d9f17720a876dd31dece585d08c2e58e02/data/sample/annotations.json) and the [structured showcase data](assets/showcase/data.json).

UI state stays in the current page. Detailed diagnostic events can be enabled with `?debug=1`; they contain selected IDs and modes, not sample content.

## See also

- [Project overview](index.md)
- [Architecture](architecture.md)
- [Source repository](https://github.com/kiselyovd/vehicle-keypoints)
