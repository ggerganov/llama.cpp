/**
 *
 * MODELS DISCOVER - DETAILS - DOWNLOAD OPTIONS
 *
 * The download area of the detail pane: one chip per main quant, the drafts it
 * rides with badged on it, a draft checkbox and type pick at the bottom, and a
 * terminal command preview inside every row.
 *
 */

/**
 * **ModelsDiscoverDetailsDownloadOptions** - GGUF download options
 *
 * Groups the main quants by bit depth and renders one download action chip per
 * quant, the sidecars it pulls as a checkbox and a type pick, and the command
 * preview of each row.
 */
export { default as ModelsDiscoverDetailsDownloadOptions } from './ModelsDiscoverDetailsDownloadOptions.svelte';

/**
 * **ModelsDiscoverDetailsDownloadOptionsDraftDownloadButton** - Add a missing draft
 *
 * A downloaded model that is missing the draft of its quant offers the draft as
 * its own chip: the click posts the model with that kind, and the server keeps
 * the files already on disk and pulls the draft.
 */
export { default as ModelsDiscoverDetailsDownloadOptionsDraftDownloadButton } from './ModelsDiscoverDetailsDownloadOptionsDraftDownloadButton.svelte';

/**
 * **ModelsDiscoverDetailsDownloadOptionsRow** - One bit-depth group of quants
 *
 * A single bit-depth row inside ModelsDiscoverDetailsDownloadOptions: the
 * depth label carries the memory the row needs and opens the row's command
 * preview, and the quant chips of that depth download beside it.
 */
export { default as ModelsDiscoverDetailsDownloadOptionsRow } from './ModelsDiscoverDetailsDownloadOptionsRow.svelte';

/**
 * **ModelsDiscoverDetailsDownloadOptionsQuantDownloadButton** - One quant chip
 *
 * A single GGUF file as an independent action chip: download / retry when
 * idle, pause / resume / cancel while in flight, delete when downloaded.
 */
export { default as ModelsDiscoverDetailsDownloadOptionsQuantDownloadButton } from './ModelsDiscoverDetailsDownloadOptionsQuantDownloadButton.svelte';

/**
 * **ModelsDiscoverDetailsDownloadOptionsDownloadCommand** - Terminal command
 *
 * The `llama serve -hf ...` command box of one row, with a draft type pick and
 * a copy button. The quant is the row's own, so the command pins it; the draft
 * type is the panel's own pick, shared in both directions, so the command only
 * names its `--spec-type`.
 */
export { default as ModelsDiscoverDetailsDownloadOptionsDownloadCommand } from './ModelsDiscoverDetailsDownloadOptionsDownloadCommand.svelte';
