# animation-timeline-control spike

## Decision

`animation-timeline-control` is a better incremental candidate than OpenCut for ClipFinder's current editor. It is a standalone MIT canvas control with a UMD build, no runtime dependencies, and an API that can be used from the existing Flask/Jinja app without introducing React or Next.js.

## What was added

- Vendored `animation-timeline-js` 2.3.5 at `static/vendor/animation-timeline.min.js`.
- Added `/timeline-lab` as an isolated route.
- Added `templates/timeline_lab.html` with sample ClipFinder segments mapped to draggable grouped keyframes.

## Fit

The library is not a video editor. It only provides timeline rendering, selection, zooming, playhead movement, and dragging of keyframes/groups. That is useful for replacing the custom timeline interaction layer while keeping ClipFinder's existing video playback, segment model, and export backend.

The important mapping is:

- ClipFinder segment `{ start, end }`
- timeline group with two keyframes
- left keyframe controls segment start
- right keyframe controls segment end
- group drag moves the full segment range

## Risks

- Range editing is modeled through grouped keyframes, so ClipFinder still needs a small adapter layer.
- Segment ordering is still app state, not something this library solves by itself.
- Canvas accessibility and keyboard behavior need custom work.
- The lab uses the UMD bundle directly. If the editor is later moved to a build pipeline, the npm package can replace the vendored file.

## Recommendation

Use this library for a focused replacement of the timeline lane first. Keep OpenCut as a reference for editor actions, but do not fork or embed it unless ClipFinder is rewritten around a React editor architecture.
