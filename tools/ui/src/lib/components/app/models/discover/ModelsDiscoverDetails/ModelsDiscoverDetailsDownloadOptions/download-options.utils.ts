import { DRAFT_FILE_LABEL, MODEL_ID, PATH_SEPARATOR } from '$lib/constants';
import { ModelDraftSidecar, ModelSelectableFileKind } from '$lib/enums';
import { HuggingFaceService } from '$lib/services';
import { isAuxSidecar, isDraftSidecar } from '$lib/utils';

/**
 * Draft kinds in the order a main-model download pulls them along. DFlash
 * wins: on the models that ship both it drafts a block at a time and stays
 * accurate further out than MTP's few tokens. A repo carrying only the other
 * kind keeps it; a kind not listed here never rides along automatically.
 */
export const DRAFT_PREFERENCE_ORDER: readonly ModelDraftSidecar[] = [
	ModelDraftSidecar.DFLASH,
	ModelDraftSidecar.MTP
];

/** Kind of a file path: the main weights, a draft sidecar, or an aux sidecar (mmproj). */
export function classify(path: string): ModelSelectableFileKind {
	const sidecar = HuggingFaceService.extractQuantMeta(path)?.sidecar;

	if (!sidecar) return ModelSelectableFileKind.MAIN;

	return isAuxSidecar(sidecar) ? ModelSelectableFileKind.AUX : ModelSelectableFileKind.DRAFT;
}

/** Display label of a file: its quant, else the file name without the extension. */
export function labelFor(path: string): string {
	const meta = HuggingFaceService.extractQuantMeta(path);

	if (meta?.quant) return meta.quant;

	// Quantless sidecar files: the chip badge already carries the sidecar type,
	// so the label only marks draft files; aux sidecars badge alone.
	if (meta?.sidecar) return isDraftSidecar(meta.sidecar) ? DRAFT_FILE_LABEL : '';

	const basename = path.split(PATH_SEPARATOR).pop() ?? path;

	return basename.replace(MODEL_ID.WEIGHT_EXTENSION_REGEX, '');
}
