<script lang="ts">
	import ModelsSelectorDownloadItem from './ModelsSelectorDownloadItem.svelte';
	import { Check, Heart, Power } from '@lucide/svelte';
	import {
		GroupedList,
		ModelAvatar,
		ModelDraftSubline,
		ModelId,
		ModelOrgAvatar,
		ModelsSection,
		ModelsSelectorOption
	} from '$lib/components/app';
	import { DialogConfirmDownload } from '$lib/components/app/dialogs';
	import Logo from '$lib/components/app/misc/Logo.svelte';
	import {
		LOCAL_BACKEND_ID,
		MODEL_ROW_WINDOW,
		SELECTOR_GROUP_STICKY_OFFSET,
		SETTINGS_KEYS
	} from '$lib/constants';
	import { ModelDownloadConfirmAction, ServerModelStatus } from '$lib/enums';
	import { ModelsService } from '$lib/services';
	import { modelsStore, settingsStore } from '$lib/stores';
	import type { ModelDownloadEntry, ModelSidecarBadge } from '$lib/types';
	import {
		type GroupedModelOptions,
		groupModelFamilies,
		type ModelFamilyGroup,
		type ModelItem,
		orgOf,
		windowLocalGroups
	} from '$lib/utils';

	interface Props {
		groups: GroupedModelOptions;
		currentModel: string | null;
		activeId: string | null;
		sectionHeaderClass?: string;
		onSelect: (modelId: string) => void;
		renderOption?: import('svelte').Snippet<[ModelItem, boolean]>;
		/** Favorite models of every backend, listed in their own section. */
		favorites?: ModelItem[];
		/** Loaded models of every llama-compat backend, leading the list. */
		loaded?: ModelItem[];
		/** Show the organization name in every model id of the list. */
		showOrgName?: boolean;
	}

	let {
		activeId,
		currentModel,
		favorites = [],
		groups,
		loaded = [],
		onSelect,
		renderOption,
		sectionHeaderClass = 'm-0 px-2 py-2 text-[13px] font-semibold text-muted-foreground/70 select-none',
		showOrgName = true
	}: Props = $props();
	let render = $derived(renderOption ?? defaultOption);
	// a large catalog mounts a window at a time: the sentinel at the end of the list
	// grows the window when it scrolls into view
	let visibleCount = $state(MODEL_ROW_WINDOW);
	let sentinelEl = $state<HTMLElement | null>(null);
	const localGroups = $derived(windowLocalGroups(groups, visibleCount));
	const localRowCount = $derived(
		groups.available.reduce((count, group) => count + group.items.length, 0)
	);
	const hasMoreLocal = $derived(localGroups.shown < localRowCount);

	$effect(() => {
		const sentinel = sentinelEl;

		if (!sentinel || !hasMoreLocal) return;

		// scroll does not bubble, so listen in the capture phase and ask the
		// sentinel where it is instead of guessing which ancestor scrolls
		const onScroll = () => {
			const rect = sentinel.getBoundingClientRect();

			// grow only when the end of the list is on screen: otherwise every scroll
			// event of the page would mount the whole catalog
			if (rect.top > window.innerHeight || rect.bottom < 0) return;

			visibleCount += MODEL_ROW_WINDOW;
		};

		document.addEventListener('scroll', onScroll, { capture: true, passive: true });
		onScroll();

		return () => document.removeEventListener('scroll', onScroll, { capture: true });
	});
	// section headers stick below the dropdown's search block: `--dropdown-sticky-height`
	// comes from DropdownMenuSearchable and falls back to 0 in the mobile sheet
	let headerClass = $derived(`${sectionHeaderClass} sticky z-10 bg-popover`);
	const headerStyle = 'top: var(--dropdown-sticky-height, 0px)';

	/** In-flight / paused downloads, tracked by the status feed. */
	// only a running download belongs in the selector; a paused one waits in the
	// manager table, where its partial files are resumed or dropped
	let downloadEntries = $derived(
		modelsStore.status.getDownloadEntries().filter((entry) => !entry.isPaused)
	);

	// one item per repo: its live download when one runs, the listing state
	// otherwise, with the repo's draft sub-lines carried inside the item - the
	// ones on disk with a check, the ones in flight live
	type DownloadRow =
		| {
				drafts: ModelDownloadEntry[];
				entry: ModelDownloadEntry;
				kind: 'main';
				sidecars: ModelSidecarBadge[];
		  }
		| {
				drafts: ModelDownloadEntry[];
				kind: 'parent';
				repo: string;
				sidecars: ModelSidecarBadge[];
				status: ServerModelStatus | null;
		  };

	let downloadRows = $derived.by(() => {
		const rows: DownloadRow[] = [];

		for (const cluster of modelsStore.status.getDownloadClusters()) {
			const mains = cluster.mains.filter((entry) => !entry.isPaused);
			const drafts = cluster.drafts.filter((entry) => !entry.isPaused);
			// sidecars already on disk show as downloaded sub-lines
			const sidecars = modelsStore.status.getDraftSidecars(cluster.repo);

			if (mains.length > 0) {
				rows.push({ drafts, entry: mains[0], kind: 'main', sidecars });
			} else {
				rows.push({
					drafts,
					kind: 'parent',
					repo: cluster.repo,
					sidecars,
					status: modelsStore.status.mainStatus(cluster.repo)
				});
			}
		}

		return rows;
	});

	// cancel is confirmed once for the whole list, so one dialog serves every row
	// The target is kept while the dialog closes so its copy stays rendered.
	let pendingCancel = $state('');
	let cancelOpen = $state(false);

	function requestCancel(repoWithTag: string) {
		pendingCancel = repoWithTag;
		cancelOpen = true;
	}
</script>

{#snippet familyHeading({ group: family }: { group: ModelFamilyGroup<ModelItem> })}
	<div class="flex items-center gap-2 px-2 py-1.5 select-none max-md:py-2.5">
		<ModelAvatar
			option={family.entries[0].option}
			showBaseModelAvatar
			showQuantBadge={false}
			size="size-6"
		/>

		<span class="truncate text-[13px] font-semibold text-muted-foreground/70">
			{family.label}
		</span>

		<span class="text-xs text-muted-foreground">
			{family.entries.length} model{family.entries.length === 1 ? '' : 's'}
		</span>
	</div>
{/snippet}

{#snippet listItem({ depth, entry }: { depth: number; entry: ModelItem })}
	<div style="padding-left: {depth * 16}px">{@render render(entry, !showOrgName)}</div>
{/snippet}

{#snippet listRows(items: ModelItem[], prefix: string)}
	{#if settingsStore.config[SETTINGS_KEYS.GROUP_MODELS_BY_FAMILY]}
		<GroupedList
			group={familyHeading}
			groups={groupModelFamilies(items, (row) => row.option.model).map((family) => ({
				entries: family.entries,
				group: family,
				key: `${prefix}-${family.key}`
			}))}
			initialCollapsed={modelsStore.collapsedGroupsUnder(prefix)}
			item={listItem}
			keyOf={(row) => `${prefix}-${row.option.id}`}
			onCollapsedChange={(key, collapsed) => modelsStore.setGroupCollapsed(prefix, key, collapsed)}
			stickyStyle={SELECTOR_GROUP_STICKY_OFFSET}
		/>
	{:else}
		<GroupedList item={listItem} {items} keyOf={(row) => `${prefix}-${row.option.id}`} />
	{/if}
{/snippet}

{#snippet defaultOption(item: ModelItem, hideOrgName: boolean)}
	{@const { option } = item}
	{@const isSelected = currentModel === option.model || activeId === option.id}
	{@const isFav = modelsStore.favoriteModelIds.has(option.model)}

	<ModelsSelectorOption
		{hideOrgName}
		{isFav}
		isHighlighted={false}
		{isSelected}
		onKeyDown={() => {}}
		onMouseEnter={() => {}}
		{onSelect}
		{option}
	/>
{/snippet}

{#if loaded.length > 0}
	<ModelsSection
		count={loaded.length}
		label="Loaded models"
		persistKey="loaded"
		revealChevronOnHover
		sticky
	>
		{#snippet icon()}
			<Power class="h-3.5 w-3.5 shrink-0" />
		{/snippet}

		{#each loaded as item (item.option.id)}
			{@render render(item, !showOrgName)}
		{/each}
	</ModelsSection>
{/if}

{#if favorites.length > 0}
	<!-- Favorites come first; the sections below skip them -->
	<ModelsSection label="Favorites" persistKey="favorites" revealChevronOnHover sticky>
		{#snippet icon()}
			<Heart class="h-3.5 w-3.5 shrink-0" />
		{/snippet}

		{#each favorites as item (`fav-${item.option.id}`)}
			{@render render(item, !showOrgName)}
		{/each}
	</ModelsSection>
{/if}

{#if downloadEntries.length > 0}
	<p class={headerClass} style={headerStyle}>Downloading</p>

	{#each downloadRows as row (row.kind === 'main' ? row.entry.repoWithTag : `parent:${row.repo}`)}
		{#if row.kind === 'main'}
			<ModelsSelectorDownloadItem
				drafts={row.drafts}
				entry={row.entry}
				onRequestCancel={requestCancel}
				{showOrgName}
				sidecars={row.sidecars}
			/>
		{:else}
			<!-- no main download runs for the repo; the row anchors its draft sub-lines -->
			<div class="rounded-sm transition-colors hover:bg-accent/50">
				<div class="flex items-center gap-2 p-2 text-sm">
					<ModelOrgAvatar org={orgOf(row.repo)} size="h-6 w-6" />

					<ModelId class="flex-1" hideOrgName={!showOrgName} modelId={row.repo} showRawTooltip />

					{#if row.status === ServerModelStatus.LOADED}
						<span class="h-2 w-2 shrink-0 rounded-full bg-green-500"></span>
					{:else if row.status === ServerModelStatus.DOWNLOADED}
						<Check class="h-4 w-4 shrink-0 text-green-500" />
					{/if}
				</div>

				<div class="flex flex-col px-2 pb-1.5">
					{#each row.drafts as draft (draft.repoWithTag)}
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

					{#each row.sidecars as badge (badge.repo + ':' + badge.kind + ':' + (badge.quant ?? ''))}
						<ModelDraftSubline
							connectorClass="ml-4"
							kind={badge.kind}
							quant={badge.quant}
							state="downloaded"
						/>
					{/each}
				</div>
			</div>
		{/if}
	{/each}
{/if}

{#snippet localRows()}
	{@render listRows(
		localGroups.available.flatMap((group) => group.items),
		'local'
	)}

	{#if hasMoreLocal}
		<div bind:this={sentinelEl} aria-hidden="true" class="h-px"></div>
	{/if}
{/snippet}

{#if groups.available.length > 0}
	<ModelsSection label="Local models" persistKey={LOCAL_BACKEND_ID} revealChevronOnHover sticky>
		{#snippet icon()}
			<Logo class="shrink-0" style="--size: 0.875rem" />
		{/snippet}

		{@render localRows()}
	</ModelsSection>
{/if}

<DialogConfirmDownload
	action={ModelDownloadConfirmAction.CANCEL}
	onClose={() => (cancelOpen = false)}
	onConfirm={() => void modelsStore.status.cancelDownloadWithDrafts(pendingCancel)}
	open={cancelOpen}
	repoWithTag={pendingCancel}
/>
