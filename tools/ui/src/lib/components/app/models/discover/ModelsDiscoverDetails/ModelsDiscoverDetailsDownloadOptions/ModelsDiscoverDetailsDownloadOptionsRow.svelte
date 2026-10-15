<script lang="ts">
	import ModelsDiscoverDetailsDownloadOptionsQuantDownloadButton from './ModelsDiscoverDetailsDownloadOptionsQuantDownloadButton.svelte';
	import {
		BIT_DEPTH_LABEL_SUFFIX,
		GIGABYTE_LABEL,
		OTHER_BIT_DEPTH,
		OTHER_BIT_DEPTH_LABEL
	} from '$lib/constants';
	import { ModelSelectableFileKind } from '$lib/enums';
	import type { ModelDownloadEntryState, ModelSelectableFile } from '$lib/types';
	import { minMemoryTierGb } from '$lib/utils';

	interface Props {
		/** Bit depth of the row; `99` renders as "Other". */
		bitDepth: number;
		/** The draft sidecar that rides along with a main download, per main file. */
		companionDraftFor?: (file: ModelSelectableFile) =>
			| (ModelSelectableFile & {
					state: ModelDownloadEntryState;
			  })
			| undefined;
		/** Every GGUF of this bit depth, with download state attached. */
		files: (ModelSelectableFile & { state: ModelDownloadEntryState })[];
		/** Forwarded to each chip: ask the parent to confirm a cancel. */
		onRequestCancel?: (repoWithTag: string) => void;
		/** Forwarded to each chip: ask the parent to confirm a delete. */
		onRequestDelete?: (repoWithTag: string) => void;
	}

	let { bitDepth, companionDraftFor, files, onRequestCancel, onRequestDelete }: Props = $props();

	let mainFile = $derived(files.find((f) => f.kind === ModelSelectableFileKind.MAIN) ?? null);
	let draftFile = $derived(files.find((f) => f.kind === ModelSelectableFileKind.DRAFT) ?? null);

	let mainMemGb = $derived(mainFile ? minMemoryTierGb(mainFile.size ?? 0) : null);
	let draftMemGb = $derived(draftFile ? minMemoryTierGb(draftFile.size ?? 0) : null);
</script>

<div class="grid grid-cols-[5rem_1fr] items-center gap-3 py-3 @max-lg:grid-cols-1 @max-lg:gap-1.5">
	<div
		class="pt-1 text-sm tabular-nums text-muted-foreground @max-lg:flex @max-lg:items-baseline @max-lg:gap-2 @max-lg:pt-0"
	>
		{#if bitDepth === OTHER_BIT_DEPTH}
			{OTHER_BIT_DEPTH_LABEL}
		{:else}
			{bitDepth}{BIT_DEPTH_LABEL_SUFFIX}
		{/if}

		{#if mainMemGb}
			<span
				class="block text-[10px] whitespace-nowrap text-muted-foreground/60 @max-lg:inline @max-lg:whitespace-normal"
			>
				needs at least {mainMemGb}{GIGABYTE_LABEL}{draftMemGb
					? ` + ${draftMemGb}${GIGABYTE_LABEL}`
					: ''} memory
			</span>
		{/if}
	</div>

	<div class="flex flex-wrap justify-end gap-1.5 @max-lg:justify-start">
		{#each files as file (file.path)}
			<ModelsDiscoverDetailsDownloadOptionsQuantDownloadButton
				companion={companionDraftFor && file.kind === ModelSelectableFileKind.MAIN
					? companionDraftFor(file)
					: undefined}
				entry={file.state}
				{file}
				{onRequestCancel}
				{onRequestDelete}
			/>
		{/each}
	</div>
</div>
