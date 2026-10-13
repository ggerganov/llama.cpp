<script lang="ts">
	import ModelDownloadProgressBar from '../ModelDownloadProgressBar.svelte';
	import ModelDraftSubline from '../ModelDraftSubline.svelte';
	import ModelOrgAvatar from '../ModelOrgAvatar.svelte';
	import { Loader2, Pause, Play, X } from '@lucide/svelte';
	import { ActionIcon, ModelId } from '$lib/components/app';
	import { DISCOVER_ICON } from '$lib/constants';
	import { HuggingFaceService, ModelsService } from '$lib/services';
	import { modelsStore, uiStore } from '$lib/stores';
	import type { ModelDownloadEntry, ModelSidecarBadge } from '$lib/types';
	import { repoOf } from '$lib/utils';

	interface Props {
		/** One entry from the status feed: an in-flight or paused download. */
		entry: ModelDownloadEntry;
		/** The repo's draft downloads, rendered as sub-lines inside the item. */
		drafts?: ModelDownloadEntry[];
		/** Draft sidecars already on disk, rendered as checked sub-lines. */
		sidecars?: ModelSidecarBadge[];
		/** Ask the list to confirm cancelling; the list owns the single dialog. */
		onRequestCancel?: (repoWithTag: string) => void;
		/** Show the organization name in the repo id, as the other selector rows do. */
		showOrgName?: boolean;
	}

	let { drafts = [], entry, onRequestCancel, showOrgName = true, sidecars = [] }: Props = $props();

	let percent = $derived(
		entry.progress && entry.progress.totalBytes > 0
			? Math.round((entry.progress.downloadedBytes / entry.progress.totalBytes) * 100)
			: null
	);

	let actionText = $derived(entry.isPaused ? 'Resume downloading' : 'Pause downloading');

	// the base model org resolves lazily via HF when unknown
	let orgName = $derived(ModelsService.parseModelId(entry.repoWithTag).orgName);
	let fetchedBaseModelOrg = $state<string | null>(null);

	$effect(() => {
		const repoWithTag = entry.repoWithTag;

		fetchedBaseModelOrg = null;

		// avatars hidden with the Hub metadata setting off: no lookups at all
		if (!HuggingFaceService.isEnabled()) return;

		let cancelled = false;

		void HuggingFaceService.getBaseModel(repoWithTag)
			.then((base) => {
				if (!cancelled && base?.org) fetchedBaseModelOrg = base.org;
			})
			// best-effort lookup: offline or unknown repos keep the repo org
			.catch(() => {});

		return () => {
			cancelled = true;
		};
	});
</script>

<!-- One in-flight download; same actions as the discover quant chips: the row
     itself pauses / resumes, the trailing X cancels (stops and discards the
     partial files). Both affordances fade in on hover, the slots are reserved
     so the list never reflows. The repo's draft downloads hang off the item as
     compact sub-lines instead of rows of their own. -->
<div class="group relative rounded-sm transition-colors hover:bg-accent/50">
	<div class="flex items-center gap-2 p-2 text-left text-sm">
		<button
			aria-label={actionText}
			class="flex min-w-0 flex-1 cursor-pointer items-center gap-2 text-left"
			onclick={() => void modelsStore.status.toggleDownloadWithDrafts(entry.repoWithTag)}
			type="button"
		>
			{#if orgName}
				<ModelOrgAvatar
					org={fetchedBaseModelOrg ?? orgName}
					quantOrg={orgName}
					quantPositionClass="-bottom-1 -right-1"
					quantSize="h-3 w-3"
					size="h-6 w-6"
				/>
			{/if}

			<ModelId
				class="flex-1"
				hideOrgName={!showOrgName}
				modelId={entry.repoWithTag}
				showRawTooltip
			/>

			{#if percent !== null}
				<span class="ml-auto shrink-0 font-mono text-xs tabular-nums text-muted-foreground">
					{percent}%
				</span>
			{:else if entry.isPaused}
				<span class="ml-auto shrink-0 text-xs text-muted-foreground">Paused</span>
			{/if}

			<!-- status action: spinner -> pause on hover while in flight, play on hover
				 when paused; opacity only for the spinner, the spin owns the transform -->
			<span class="relative inline-flex h-5 w-5 shrink-0 items-center justify-center">
				{#if entry.isPaused}
					<Play
						class="absolute h-4 w-4 scale-75 opacity-0 transition-[opacity,transform] duration-150 ease-[cubic-bezier(0.23,1,0.32,1)] group-hover:scale-100 group-hover:opacity-100 [@media(pointer:coarse)]:scale-100 [@media(pointer:coarse)]:opacity-100"
					/>
				{:else}
					<Loader2
						class="absolute h-4 w-4 animate-spin text-muted-foreground transition-opacity duration-150 ease-[cubic-bezier(0.23,1,0.32,1)] group-hover:opacity-0 [@media(pointer:coarse)]:hidden"
					/>

					<Pause
						class="absolute h-4 w-4 opacity-0 transition-opacity duration-150 ease-[cubic-bezier(0.23,1,0.32,1)] group-hover:opacity-100 [@media(pointer:coarse)]:opacity-100"
					/>
				{/if}
			</span>
		</button>

		<!-- the row actions keep the sizing and reveal of the other selector rows -->
		<ActionIcon
			ariaLabel="See in Discover"
			class="pointer-events-none h-5 w-5 shrink-0 opacity-0 transition-opacity group-hover:pointer-events-auto group-hover:opacity-100 hover:text-foreground [@media(pointer:coarse)]:pointer-events-auto [@media(pointer:coarse)]:opacity-100"
			icon={DISCOVER_ICON}
			iconSize="h-4 w-4"
			onclick={() => uiStore.openModelsDiscover(repoOf(entry.repoWithTag))}
		/>

		<ActionIcon
			ariaLabel="Cancel downloading"
			class="pointer-events-none h-5 w-5 shrink-0 opacity-0 transition-opacity group-hover:pointer-events-auto group-hover:opacity-100 hover:text-destructive [@media(pointer:coarse)]:pointer-events-auto [@media(pointer:coarse)]:opacity-100"
			icon={X}
			iconSize="h-4 w-4"
			onclick={() => onRequestCancel?.(entry.repoWithTag)}
		/>
	</div>

	<div class="flex flex-col px-2 pb-1.5">
		{#each drafts as draft (draft.repoWithTag)}
			{@const parsedDraft = ModelsService.parseModelId(draft.repoWithTag)}

			<ModelDraftSubline
				connectorClass="ml-4"
				kind={parsedDraft.sidecar ?? ''}
				percent={draft.progress && draft.progress.totalBytes > 0
					? Math.round((draft.progress.downloadedBytes / draft.progress.totalBytes) * 100)
					: null}
				quant={parsedDraft.quantization}
				state={draft.isPaused ? 'paused' : 'downloading'}
			/>
		{/each}

		{#each sidecars as badge (badge.repo + ':' + badge.kind + ':' + (badge.quant ?? ''))}
			<ModelDraftSubline
				connectorClass="ml-4"
				kind={badge.kind}
				quant={badge.quant}
				state="downloaded"
			/>
		{/each}
	</div>

	{#if entry.progress && entry.progress.totalBytes > 0}
		<ModelDownloadProgressBar
			downloadedBytes={entry.progress.downloadedBytes}
			overlay
			totalBytes={entry.progress.totalBytes}
		/>
	{/if}
</div>
