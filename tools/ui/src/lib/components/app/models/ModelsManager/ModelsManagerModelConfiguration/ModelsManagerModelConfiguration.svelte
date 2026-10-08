<script lang="ts">
	import ModelDownloadProgressBar from '../../ModelDownloadProgressBar.svelte';
	import { downloadProgressFor, type ModelOverride } from '../utils';
	import ModelsManagerModelConfigurationActions from './ModelsManagerModelConfigurationActions.svelte';
	import ModelsManagerModelConfigurationHeader from './ModelsManagerModelConfigurationHeader.svelte';
	import ModelsManagerModelConfigurationInference from './ModelsManagerModelConfigurationInference.svelte';
	import ModelsManagerModelConfigurationInformation from './ModelsManagerModelConfigurationInformation.svelte';
	import ModelsManagerModelConfigurationLoad from './ModelsManagerModelConfigurationLoad.svelte';
	import { Trash2 } from '@lucide/svelte';
	import { DialogConfirmDownload } from '$lib/components/app/dialogs';
	import { Button } from '$lib/components/ui/button';
	import * as Tabs from '$lib/components/ui/tabs';
	import { LOAD_DEFAULTS, MODEL_ID } from '$lib/constants';
	import { ModelDownloadConfirmAction, ServerModelStatus } from '$lib/enums';
	import { HuggingFaceService } from '$lib/services';
	import { deviceStore, modelsStore, uiStore } from '$lib/stores';
	import type { HfModelDetailInfo } from '$lib/types/huggingface';
	import type { ModelOption } from '$lib/types/models';
	import { repoOf } from '$lib/utils';
	import { getBackend } from '$lib/utils/api-base';
	import { getBackendCapabilities } from '$lib/utils/backend';

	interface Props {
		isCustomized: boolean;
		onClose: () => void;
		onSave: (override: ModelOverride) => void;
		onToggleLoad: () => void;
		onUseInChat: () => void;
		onUseInNewChat: () => void;
		option: ModelOption;
		override?: ModelOverride;
	}

	let {
		isCustomized,
		onClose,
		onSave,
		onToggleLoad,
		onUseInChat,
		onUseInNewChat,
		option,
		override
	}: Props = $props();

	let tab = $state('information');
	// edits in this pane, cleared on reset; the stored override is the baseline
	let edits = $state<ModelOverride | null>(null);
	let draft = $derived(edits ?? override ?? {});

	// the same removal the table row offers, confirmed by the same dialog
	let deleteOpen = $state(false);

	// the props cache is a plain Map, so the read has to name its version to stay reactive
	let serverProps = $derived.by(() => {
		void modelsStore.props.cacheVersion;

		return modelsStore.props.getModelProps(option.model);
	});
	let status = $derived(modelsStore.getModelStatus(option.model));
	let isOperationInProgress = $derived(modelsStore.status.isOperationInProgress(option.model));
	let isLoaded = $derived(modelsStore.isModelRunning(option.model));
	let loadProgress = $derived(
		isOperationInProgress ? modelsStore.status.getLoadProgress(option.model) : null
	);
	// a provider that cannot load or unload has no button to offer
	let canToggleLoad = $derived(getBackendCapabilities(getBackend(option.backendId)).loadUnload);

	// an in-flight download has no configuration: the pane shows its progress and
	// the download actions, everything a loaded or downloaded model offers waits
	let isDownload = $derived(
		modelsStore.status.isDownloadInProgress(option.model) ||
			modelsStore.status.isDownloadPaused(option.model)
	);
	let isDownloadPaused = $derived(modelsStore.status.isDownloadPaused(option.model));
	let downloadProgress = $derived(downloadProgressFor(option.model));
	let downloadPercent = $derived(
		downloadProgress && downloadProgress.totalBytes > 0
			? Math.round((downloadProgress.downloadedBytes / downloadProgress.totalBytes) * 100)
			: null
	);

	// the same cancellation the table's rows confirm, owned by this pane
	let cancelDownloadOpen = $state(false);

	// The server reports the full metadata only once a model is loaded, which loads it.
	// With the Hub enabled, it fills those gaps instead.

	// The server only reports the full metadata once a model is loaded, which loads
	// it. When discovery is on, the Hub fills those gaps instead.
	let hubDetails = $state<HfModelDetailInfo | null>(null);
	// the window the model can take, from the listing or the Hub; a loaded model's runtime
	// context only bounds it when nothing else says otherwise
	let contextMax = $derived(
		option.contextLength ??
			hubDetails?.gguf?.context_length ??
			serverProps?.default_generation_settings?.n_ctx ??
			LOAD_DEFAULTS.contextLength
	);

	$effect(() => {
		const repo = repoOf(option.model);

		let cancelled = false;

		hubDetails = null;

		if (!HuggingFaceService.isEnabled() || !repo.includes(MODEL_ID.ORG_SEPARATOR)) {
			return;
		}

		void HuggingFaceService.getDetails(repo)
			.then((details) => {
				if (!cancelled) hubDetails = details;
			})
			.catch(() => {});

		return () => {
			cancelled = true;
		};
	});

	function resetDraft(): void {
		edits = null;
	}
</script>

<div class="flex h-full min-h-0 flex-col">
	<ModelsManagerModelConfigurationHeader
		{isDownload}
		{isDownloadPaused}
		{canToggleLoad}
		{isCustomized}
		{isLoaded}
		{onClose}
		onRequestCancelDownload={() => (cancelDownloadOpen = true)}
		onSeeInDiscover={isDownload
			? () => uiStore.openModelsDiscover(repoOf(option.model))
			: undefined}
		onToggleDownload={() => void modelsStore.status.toggleDownloadWithDrafts(option.model)}
		{onToggleLoad}
		{onUseInChat}
		{onUseInNewChat}
		{option}
		{status}
	/>

	{#if isDownload}
		<!-- nothing is on disk yet, so the pane keeps to the download itself -->
		<div class="min-h-0 flex-1 overflow-y-auto py-4 pl-4 max-md:pr-4">
			<div class="space-y-2 py-1">
				<div class="flex items-baseline justify-between gap-3 text-sm">
					<span class="text-muted-foreground">
						{isDownloadPaused ? 'Paused at' : 'Downloaded'}
					</span>

					<span class="font-mono text-xs tabular-nums text-muted-foreground">
						{#if downloadPercent !== null}
							{HuggingFaceService.formatFileSize(downloadProgress?.downloadedBytes ?? 0)}
							/
							{HuggingFaceService.formatFileSize(downloadProgress?.totalBytes ?? 0)}
							({downloadPercent}%)
						{:else}
							Waiting for progress...
						{/if}
					</span>
				</div>

				{#if downloadProgress && downloadProgress.totalBytes > 0}
					<ModelDownloadProgressBar
						downloadedBytes={downloadProgress.downloadedBytes}
						totalBytes={downloadProgress.totalBytes}
					/>
				{/if}
			</div>
		</div>
	{:else}
		<Tabs.Root
			class="mt-3 min-h-0 flex-1 gap-0"
			onValueChange={(value) => (tab = value)}
			value={tab}
		>
			<div class="pl-4 max-md:pr-4">
				<Tabs.List class="w-full">
					<Tabs.Trigger value="information">Information</Tabs.Trigger>

					<Tabs.Trigger value="load">Load</Tabs.Trigger>

					<Tabs.Trigger value="inference">Inference</Tabs.Trigger>
				</Tabs.List>
			</div>

			<div class="min-h-0 flex-1 overflow-y-auto py-4 pl-4 max-md:pr-4">
				<Tabs.Content value="information">
					<ModelsManagerModelConfigurationInformation
						draftSetting={override?.load?.speculativeDecoding ?? null}
						hub={hubDetails}
						{option}
						{serverProps}
					/>
				</Tabs.Content>

				<Tabs.Content value="load">
					<ModelsManagerModelConfigurationLoad
						{contextMax}
						{draft}
						{loadProgress}
						onChange={(next) => (edits = next)}
						onReset={resetDraft}
						onSave={() => onSave(draft)}
					/>
				</Tabs.Content>

				<Tabs.Content value="inference">
					<ModelsManagerModelConfigurationInference
						{draft}
						onChange={(next) => (edits = next)}
						onReset={resetDraft}
						onSave={() => onSave(draft)}
					/>
				</Tabs.Content>
			</div>
		</Tabs.Root>

		<div class="border-t border-border/30 px-4 pt-3 pb-4">
			<Button
				class="w-full justify-start gap-2 text-destructive hover:bg-destructive/10 hover:text-destructive dark:hover:bg-destructive/20"
				onclick={() => (deleteOpen = true)}
				size="sm"
				variant="ghost"
			>
				<Trash2 class="h-4 w-4" />
				Delete this model from disk
			</Button>
		</div>
	{/if}

	<!-- a phone keeps the actions at the bottom of its screen, in reach of the thumb -->
	{#if deviceStore.isMobile}
		<div class="shrink-0 border-t border-border/40 px-4 py-3">
			<ModelsManagerModelConfigurationActions
				{isDownload}
				{isDownloadPaused}
				{canToggleLoad}
				{isLoaded}
				isLoading={status === ServerModelStatus.LOADING}
				onRequestCancelDownload={() => (cancelDownloadOpen = true)}
				onSeeInDiscover={isDownload
					? () => uiStore.openModelsDiscover(repoOf(option.model))
					: undefined}
				onToggleDownload={() => void modelsStore.status.toggleDownloadWithDrafts(option.model)}
				{onToggleLoad}
				{onUseInChat}
				{onUseInNewChat}
			/>
		</div>
	{/if}

	<DialogConfirmDownload
		action={ModelDownloadConfirmAction.DELETE}
		onClose={() => (deleteOpen = false)}
		open={deleteOpen}
		repoWithTag={option.model}
	/>

	<DialogConfirmDownload
		action={ModelDownloadConfirmAction.CANCEL}
		onClose={() => (cancelDownloadOpen = false)}
		onConfirm={() => void modelsStore.status.cancelDownloadWithDrafts(option.model)}
		open={cancelDownloadOpen}
		repoWithTag={option.model}
	/>
</div>
