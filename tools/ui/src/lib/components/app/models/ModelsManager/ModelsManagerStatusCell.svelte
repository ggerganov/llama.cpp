<script lang="ts">
	import ModelLoadControl from '../ModelLoadControl.svelte';
	import ModelsManagerDownloadControl from './ModelsManagerDownloadControl.svelte';
	import { LOCAL_BACKEND_ID } from '$lib/constants';
	import type { ModelRowDownloadState } from '$lib/enums';
	import { ServerModelStatus } from '$lib/enums';
	import { modelsStore, serverStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';

	interface Props {
		class?: string;
		/** Download state, when the row stands for a tracked download. */
		download?: ModelRowDownloadState | null;
		option: ModelOption;
	}

	let { class: className = '', download = null, option }: Props = $props();

	let status = $derived(modelsStore.getModelStatus(option.model));
	let isOperationInProgress = $derived(modelsStore.status.isOperationInProgress(option.model));
	let isLoaded = $derived(modelsStore.isModelRunning(option.model));
	// a row served by this UI loads through it in router mode only; a row of
	// another server always loads through that server
	let canToggleLoad = $derived(
		!option.backendId || option.backendId === LOCAL_BACKEND_ID ? serverStore.localIsRouter : true
	);
</script>

{#if download}
	<ModelsManagerDownloadControl class="justify-self-center {className}" {option} state={download} />
{:else if canToggleLoad}
	<ModelLoadControl
		class="justify-self-center {className}"
		isFailed={status === ServerModelStatus.FAILED}
		{isLoaded}
		isLoading={status === ServerModelStatus.LOADING || isOperationInProgress}
		isSleeping={status === ServerModelStatus.SLEEPING}
		{option}
	/>
{:else}
	<span
		class="justify-self-center text-sm text-muted-foreground {className}"
		title="Served by this server"
	>
		-
	</span>
{/if}
