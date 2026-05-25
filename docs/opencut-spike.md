# OpenCut editor spike

## Repos inspected

- `E:\ClipFinder-labs\OpenCut` at `238750c`
- `E:\ClipFinder-labs\opencut-classic` at `cf5e79e`

## Findings

OpenCut is MIT licensed, React/Next based, and has a serious editor architecture:

- `apps/web/src/timeline/components/index.tsx` implements the timeline shell, zoom, ruler, track labels, playhead, wheel handling, selection, drag/drop, and resize wiring.
- `apps/web/src/actions/definitions.ts` defines editor actions and default shortcuts.
- `apps/web/src/actions/use-keybindings.ts` captures keyboard shortcuts with a central action layer.
- `apps/web/src/actions/use-editor-actions.ts` implements `split`, `split-left`, and `split-right`.
- OpenCut already maps `q` to split/remove-left and `w` to split/remove-right.

The useful pattern is the action layer:

1. Define actions independently from UI controls.
2. Route keyboard shortcuts and toolbar buttons through the same action handlers.
3. Keep timeline operations in a dedicated timeline manager instead of scattered DOM handlers.

## Integration assessment

OpenCut is not a small timeline package we can install into the current Flask/Jinja app. Its timeline depends on:

- Next/React component tree.
- Editor provider/core singleton.
- Timeline stores/hooks/controllers.
- Media/project model.
- UI components and icon libraries.
- WASM/Rust package integration.

Directly copying `Timeline` into ClipFinder would require porting a large part of OpenCut's frontend architecture. That is a rewrite, not a library integration.

## Recommended path

Do not fork OpenCut as the primary product path yet.

Instead:

1. Keep ClipFinder's current Flask backend and ffmpeg export pipeline.
2. Add an action layer to the current editor first, modeled after OpenCut.
3. Stabilize keyboard shortcuts and timeline operations around that action layer.
4. If the editor continues growing, create a separate React frontend under `apps/editor` or `frontend/` and consume the existing Flask APIs.
5. Use OpenCut as a reference for timeline architecture, shortcuts, track layout, and drag/resize behavior.

## Possible phase plan

### Phase 1: harden current editor

- Add a central `ACTIONS` map in plain JS.
- Route buttons and shortcuts through actions.
- Add visible shortcut help.
- Fix focus handling around iframe/video/input controls.
- Add unit-like browser tests for segment operations.

### Phase 2: React timeline lab

- Add a minimal Vite/React app at `/editor-react` or as a separate dev server.
- Model ClipFinder segments as React state.
- Implement only segment timeline, playhead, shortcuts, and export payload.
- Keep Flask APIs unchanged.

### Phase 3: decide migration

- If React lab improves velocity, migrate the editor UI.
- If not, continue improving custom JS with OpenCut-inspired architecture.

