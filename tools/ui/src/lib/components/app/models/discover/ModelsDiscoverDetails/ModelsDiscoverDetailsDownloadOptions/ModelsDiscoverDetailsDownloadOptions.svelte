<script lang="ts">
	import { classify, draftBadgeOf, labelFor, sortDraftKinds } from './download-options.utils';
	import ModelsDiscoverDetailsDownloadOptionsDownloadCommand from './ModelsDiscoverDetailsDownloadOptionsDownloadCommand.svelte';
	import ModelsDiscoverDetailsDownloadOptionsRow from './ModelsDiscoverDetailsDownloadOptionsRow.svelte';
	import { ChevronDown, Terminal } from '@lucide/svelte';
	import { DialogConfirmDownload } from '$lib/components/app/dialogs';
	import * as Select from '$lib/components/ui/select';
	import {
		ModelDownloadConfirmAction,
		ModelDraftSidecar,
		ModelSelectableFileKind
	} from '$lib/enums';
	import { HuggingFaceService, ModelsService } from '$lib/services';
	import { modelsStore } from '$lib/stores';
	import type {
		ModelBitDepthRow,
		ModelCompanionDraft,
		ModelDownloadEntryState,
		ModelQuantOption,
		ModelSelectableFile
	} from '$lib/types';
	import { directoryOf, isServerSidecarFile } from '$lib/utils';
	import { cubicOut } from 'svelte/easing';
	import { slide } from 'svelte/transition';

	interface Props {
		/** Full HuggingFace repo id, e.g. `ggml-org/gemma-3-4b-it-GGUF`. */
		modelId: string;
		/** GGUF files grouped by bit depth. */
		bitDepthRows: ModelBitDepthRow[];
	}

	let { bitDepthRows, modelId }: Props = $props();

	// collapsed until asked for; the panel remounts per model
	let commandOpen = $state(false);

	// one shared dialog confirms the panel's destructive chip actions; the target
	// is kept after closing so its copy stays rendered through the close transition
	let pending: { action: ModelDownloadConfirmAction; repoWithTag: string } = $state({
		action: ModelDownloadConfirmAction.CANCEL,
		repoWithTag: ''
	});
	let confirmOpen = $state(false);

	function requestCancel(repoWithTag: string) {
		pending = { action: ModelDownloadConfirmAction.CANCEL, repoWithTag };
		confirmOpen = true;
	}

	function requestDelete(repoWithTag: string) {
		pending = { action: ModelDownloadConfirmAction.DELETE, repoWithTag };
		confirmOpen = true;
	}

	/** Download state of a main quant's tag, from the status feed. */
	function stateFor(repoWithTag: string): ModelDownloadEntryState {
		const isDownloading = modelsStore.status.isDownloadInProgress(repoWithTag);
		const isPaused = modelsStore.status.isDownloadPaused(repoWithTag);

		return {
			isDownloaded: !isDownloading && modelsStore.status.isModelDownloaded(repoWithTag),
			isDownloading,
			isFailed: modelsStore.status.hasFailedDownload(repoWithTag),
			isPaused,
			// live progress while downloading, else the frozen snapshot of the pause
			progress:
				modelsStore.status.getDownloadProgress(repoWithTag) ??
				modelsStore.status.getPausedDownloadProgress(repoWithTag),
			repoWithTag
		};
	}

	/**
	 * Every GGUF of the panel with its role, in row order.
	 * Single source of truth for the chip rows and the command selects.
	 */
	let selectableFiles = $derived.by(() =>
		bitDepthRows.flatMap((row) => row.files.map((file) => ({ ...file, kind: classify(file.path) })))
	);

	/** Sidecar kinds the repo reports on disk, so a chip tells a pulled draft from a missing one. */
	let onDiskDrafts = $derived(modelsStore.status.getDraftSidecars(modelId));

	/** Draft kinds the repo ships, best first and once each; one rides with every chip. */
	let draftKinds = $derived(
		sortDraftKinds(
			selectableFiles
				.flatMap((file) =>
					file.kind === ModelSelectableFileKind.DRAFT ? [draftBadgeOf(file)] : []
				)
				.filter((kind): kind is ModelDraftSidecar => kind !== null)
				.filter((kind, index, all) => all.indexOf(kind) === index)
		)
	);

	// the operator's draft pick; the best available kind rides along until they change it
	let draftPick = $state<ModelDraftSidecar | null>(null);
	let draftKind = $derived(draftPick ?? draftKinds[0] ?? null);

	/** Bits of a file's quant, 0 when its name carries none, as the server reads it. */
	function bitsOf(path: string): number {
		const quant = HuggingFaceService.extractQuantMeta(path)?.quant;

		return (quant ? HuggingFaceService.getBitDepth(quant) : null) ?? 0;
	}

	/**
	 * Draft that rides with a quant: the file of the picked kind the server would
	 * pull. Its own directory, then the exact quant, then the closest bit depth -
	 * a 4-bit model pairs with a 16-bit draft head when the repo ships no 4-bit
	 * one, the way -hf settles it.
	 */
	function companionFor(
		main: Pick<ModelSelectableFile, 'path'>,
		quant: string | null
	): ModelCompanionDraft | null {
		if (!draftKind) return null;

		const directory = directoryOf(main.path);
		const candidates = selectableFiles.filter(
			(candidate) =>
				candidate.kind === ModelSelectableFileKind.DRAFT &&
				draftBadgeOf(candidate) === draftKind &&
				// the two rules the server's own lookup applies: the token it scans
				// for, and the same directory as the weights
				isServerSidecarFile(candidate.path, draftKind) &&
				directoryOf(candidate.path) === directory
		);

		if (candidates.length === 0) return null;

		const quantOf = (candidate: ModelSelectableFile) =>
			HuggingFaceService.extractQuantMeta(candidate.path)?.quant ?? null;
		const mainBits = quant ? bitsOf(main.path) : 0;
		const exact = quant ? candidates.find((candidate) => quantOf(candidate) === quant) : undefined;
		// the closest bit depth wins when the exact quant is missing, ties on file order
		const closest = [...candidates].sort(
			(a, b) => Math.abs(bitsOf(a.path) - mainBits) - Math.abs(bitsOf(b.path) - mainBits)
		)[0];
		const picked = exact ?? closest;

		if (!picked) return null;

		const pickedQuant = quantOf(picked);

		return {
			kind: draftKind,
			// on disk when the model's own sidecar option reports it, or when a loaded
			// model's args name that same draft file
			onDisk:
				modelsStore.status.isModelDownloaded(
					ModelsService.buildDownloadTag(modelId, quant, draftKind)
				) || onDiskDrafts.some((badge) => badge.kind === draftKind && badge.quant === pickedQuant),
			path: picked.path,
			quant: pickedQuant,
			size: picked.size ?? 0
		};
	}

	/** Rows for the row component: one chip per main quant, its draft joined to it. */
	let rows = $derived.by(() =>
		bitDepthRows.map((row) => ({
			bitDepth: row.bitDepth,
			files: row.files.flatMap((file) => {
				if (classify(file.path) !== ModelSelectableFileKind.MAIN) return [];

				const quant = HuggingFaceService.extractQuantMeta(file.path)?.quant ?? null;
				const companion = companionFor(file, quant);

				return [
					{
						...file,
						companion,
						kind: ModelSelectableFileKind.MAIN,
						// the store adds the model's mmproj and the sidecars it already pulls
						sidecars: companion ? [companion.kind] : [],
						state: stateFor(ModelsService.buildDownloadTag(modelId, quant, null))
					}
				];
			})
		}))
	);

	function optionFor(file: ModelSelectableFile): ModelQuantOption {
		return {
			label: labelFor(file.path),
			path: file.path
		};
	}

	/** Non-draft quants for the command's base select, in row order. */
	let mainOptions = $derived(
		selectableFiles.filter((f) => f.kind === ModelSelectableFileKind.MAIN).map(optionFor)
	);

	/** Draft options for the command's draft select; a repo can ship several flavours. */
	let draftOptions = $derived(
		selectableFiles
			.filter((f) => f.kind === ModelSelectableFileKind.DRAFT)
			.map((f) => ({
				...optionFor(f),
				badge: draftBadgeOf(f)
			}))
	);
</script>

{#if bitDepthRows.length}
	<section class="rounded-3xl border border-border/30 bg-muted/60 shadow-sm dark:border-border/20">
		<!-- The repo ships more than one draft flavour: the pick in the corner decides
		     which one every chip on this panel pulls. -->
		{#if draftKinds.length > 1}
			<div
				class="flex items-center justify-between gap-3 border-b border-border/50 px-4 py-2.5 dark:border-border/35"
			>
				<span class="text-xs text-muted-foreground">Draft model</span>

				<Select.Root
					onValueChange={(value) => value && (draftPick = value as ModelDraftSidecar)}
					type="single"
					value={draftKind}
				>
					<Select.Trigger aria-label="Draft model type" size="xs">
						{draftKind}
					</Select.Trigger>

					<Select.Content class="font-mono text-xs">
						{#each draftKinds as kind (kind)}
							<Select.Item class="text-xs" label={kind} value={kind}>{kind}</Select.Item>
						{/each}
					</Select.Content>
				</Select.Root>
			</div>
		{/if}

		<!-- One chip per main quant, each an independent download action. The draft
			 of that quant rides with it and badges the chip; the server picks its file,
			 the way -hf does. -->
		<div class="flex w-full flex-col divide-y divide-border/50 px-4 pb-1 dark:divide-border/35">
			{#each rows as row (row.bitDepth)}
				<ModelsDiscoverDetailsDownloadOptionsRow
					bitDepth={row.bitDepth}
					files={row.files}
					onRequestCancel={requestCancel}
					onRequestDelete={requestDelete}
				/>
			{/each}
		</div>

		<!-- Terminal command, collapsed until asked for; its picks are not bound to the chips. -->
		<div class="border-t border-border/50 px-4 pt-2.5 pb-3 dark:border-border/35">
			<button
				aria-expanded={commandOpen}
				class="flex w-full cursor-pointer items-center gap-2 rounded-md py-1 text-left text-xs text-muted-foreground transition-colors hover:text-foreground"
				onclick={() => (commandOpen = !commandOpen)}
				type="button"
			>
				<Terminal class="h-3.5 w-3.5 shrink-0" />

				<span class="flex-1">
					{commandOpen
						? 'Hide the download command for terminal'
						: 'Show the download command for terminal'}
				</span>

				<ChevronDown
					class="h-3.5 w-3.5 shrink-0 transition-transform duration-200 ease-[cubic-bezier(0.23,1,0.32,1)] {commandOpen
						? 'rotate-180'
						: ''}"
				/>
			</button>

			{#if commandOpen}
				<div transition:slide={{ duration: 200, easing: cubicOut }} class="pt-2.5">
					<ModelsDiscoverDetailsDownloadOptionsDownloadCommand
						{draftOptions}
						{mainOptions}
						{modelId}
					/>
				</div>
			{/if}
		</div>
	</section>
{/if}

<DialogConfirmDownload
	action={pending.action}
	onClose={() => (confirmOpen = false)}
	open={confirmOpen}
	repoWithTag={pending.repoWithTag}
/>
