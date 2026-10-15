import { LOCAL_BACKEND_ID, type ModalityKey } from '$lib/constants';
import { ModelCapability, ModelGroupKind, ModelsTableGroupKind } from '$lib/enums';
import { HuggingFaceService, ModelsService } from '$lib/services';
import { modelsStore } from '$lib/stores';
import type {
	ModelDownloadEntry,
	ModelDownloadProgress,
	ModelOption,
	ModelSidecarBadge
} from '$lib/types/models';
import { detectThinkingSupport, detectToolUseSupport, isDraftSidecar, repoOf } from '$lib/utils';

/** One repo of the table, with the rows it ships as. */
export interface ModelQuantGroup {
	base: ModelOption;
	/** Draft downloads hanging off the row, in flight or paused. */
	draftEntries?: ModelDownloadEntry[];
	key: string;
	kind: ModelGroupKind;
	quants: ModelOption[];
	/** Draft sidecars already on disk, shown as checked sub-lines. */
	sidecarBadges?: ModelSidecarBadge[];
}

/** True when any of the manager's filters narrows the table. */
export function hasActiveFilters(
	contextLimit: number,
	modalities: ModalityKey[],
	capabilities: ModelCapability[]
): boolean {
	return contextLimit > 0 || modalities.length > 0 || capabilities.length > 0;
}

/** Byte counts of a tracked download: live while it runs, frozen while paused. */
export function downloadProgressFor(repoWithTag: string): ModelDownloadProgress | null {
	return (
		modelsStore.status.getDownloadProgress(repoWithTag) ??
		modelsStore.status.getPausedDownloadProgress(repoWithTag)
	);
}

/**
 * One table row per downloading repo, so the table lists them the way it lists
 * any other model: the main weights stand for the repo and its draft downloads
 * and on-disk sidecars hang off the row as sub-lines. A repo with no main
 * download in flight keeps one row per draft. A paused download is already in
 * the router's listing, so its own option is reused; a fresh one stands for
 * its tag alone.
 */
export function downloadGroups(
	entries: ModelDownloadEntry[],
	models: ModelOption[],
	sidecarsFor: (repo: string) => ModelSidecarBadge[]
): ModelQuantGroup[] {
	const clusters = new Map<string, { drafts: ModelDownloadEntry[]; mains: ModelDownloadEntry[] }>();

	for (const entry of entries) {
		const repo = repoOf(entry.repoWithTag);
		const cluster = clusters.get(repo) ?? { drafts: [], mains: [] };
		const sidecar = ModelsService.parseModelId(entry.repoWithTag).sidecar;

		if (sidecar && isDraftSidecar(sidecar)) cluster.drafts.push(entry);
		else cluster.mains.push(entry);

		clusters.set(repo, cluster);
	}

	return Array.from(clusters.entries()).flatMap(([repo, cluster]) => {
		const sidecars = sidecarsFor(repo);
		// the synthesized option carries backendId, which ModelOption drops on
		// assignment, so the callers spread it instead of typing this as one
		const groupOf = (repoWithTag: string) => {
			const listed = models.find((candidate) => candidate.model === repoWithTag);

			return (
				listed ?? {
					backendId: LOCAL_BACKEND_ID,
					capabilities: [],
					id: repoWithTag,
					model: repoWithTag,
					name: repoWithTag
				}
			);
		};

		if (cluster.mains.length > 0) {
			const main = cluster.mains[0];

			// the sub-lines carry the drafts, so the inline badges would repeat them
			return [
				{
					base: { ...groupOf(main.repoWithTag), draftSidecars: [] },
					draftEntries: cluster.drafts,
					key: `download::${main.repoWithTag}`,
					kind: ModelGroupKind.QUANTS,
					quants: [{ ...groupOf(main.repoWithTag), draftSidecars: [] }],
					sidecarBadges: sidecars
				}
			];
		}

		return cluster.drafts.map((entry) => ({
			base: { ...groupOf(entry.repoWithTag), draftSidecars: [] },
			key: `download::${entry.repoWithTag}`,
			kind: ModelGroupKind.QUANTS,
			quants: [{ ...groupOf(entry.repoWithTag), draftSidecars: [] }]
		}));
	});
}

/** One collapsible block of the manager's table. */
export interface ModelsTableGroup {
	/** One entry per repo, its quants hanging off it. */
	items: ModelQuantGroup[];
	key: string;
	kind: ModelsTableGroupKind;
	label: string;
}

/** Context the model runs with: what a loaded model reports. */
export function configuredContext(option: ModelOption): number | null {
	return modelsStore.isModelRunning(option.model)
		? modelsStore.props.getModelContextSize(option.model)
		: null;
}

/**
 * Capability a model reports: true or false once its listing or its chat template
 * answers, null while the Hub record that carries the template is not read yet.
 */
export function modelCapability(option: ModelOption, capability: ModelCapability): boolean | null {
	if (option.capabilities.includes(capability)) return true;

	const details = HuggingFaceService.cachedDetails(option.model);

	// only the chat template answers this, and it arrives with the Hub record
	if (details === undefined) return null;

	const template = details?.gguf?.chat_template ?? '';

	return capability === ModelCapability.TOOL_USE
		? detectToolUseSupport(template)
		: detectThinkingSupport(template);
}

/**
 * Context a model reports: the provider listing first, then the cached Hub record.
 * Null while neither of them has answered.
 */
export function modelContextLength(option: ModelOption): number | null {
	return (
		option.contextLength ??
		HuggingFaceService.cachedDetails(option.model)?.gguf?.context_length ??
		null
	);
}

/** Fold the quants of one repo into a single entry, so the table shows one row per model. */
export function groupModelQuants(models: ModelOption[]): ModelQuantGroup[] {
	const groups = new Map<string, ModelQuantGroup>();

	for (const option of models) {
		const key = repoOf(option.model);
		const group = groups.get(key);

		if (group) {
			group.quants.push(option);

			continue;
		}

		groups.set(key, { base: option, key, kind: ModelGroupKind.QUANTS, quants: [option] });
	}

	return Array.from(groups.values()).flatMap((group) => {
		const kind = groupKind(group.quants);
		const modelIds = group.quants.map((option) => option.model);

		// the very same id twice is not a quant set; keep those rows apart
		if (modelIds.length > 1 && modelIds.every((model) => model === modelIds[0])) {
			return group.quants.map((option) => ({
				...group,
				base: option,
				key: `${group.key}::${option.id}`,
				kind: ModelGroupKind.VARIANTS,
				quants: [option]
			}));
		}

		return [{ ...group, kind }];
	});
}

/** What a group folds: quants or variants of one repo. */
function groupKind(quants: ModelOption[]): ModelGroupKind {
	const isQuant = quants.every(
		(option) => (option.parsedId ?? ModelsService.parseModelId(option.model)).quantization
	);

	return isQuant ? ModelGroupKind.QUANTS : ModelGroupKind.VARIANTS;
}

/**
 * Split the repos of a section into the quants that are hidden and the ones that
 * are not, so a partly hidden repo lists in both blocks instead of dragging its
 * visible quants into the hidden one.
 */
export function splitHiddenQuants(
	entries: ModelQuantGroup[],
	isHidden: (option: ModelOption) => boolean
): { hidden: ModelQuantGroup[]; local: ModelQuantGroup[] } {
	const hidden: ModelQuantGroup[] = [];
	const local: ModelQuantGroup[] = [];

	for (const entry of entries) {
		const hiddenQuants = entry.quants.filter((quant) => isHidden(quant));
		const localQuants = entry.quants.filter((quant) => !isHidden(quant));

		if (hiddenQuants.length > 0) {
			hidden.push({
				...entry,
				base: hiddenQuants[0],
				key: `${entry.key}::hidden`,
				quants: hiddenQuants
			});
		}

		if (localQuants.length > 0) {
			local.push({ ...entry, base: localQuants[0], quants: localQuants });
		}
	}

	return { hidden, local };
}
