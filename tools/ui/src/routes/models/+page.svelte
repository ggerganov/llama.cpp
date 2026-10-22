<script lang="ts">
	import ModelsManager from '$lib/components/app/models/ModelsManager/ModelsManager.svelte';
	import { Button } from '$lib/components/ui/button';
	import { APP_NAME, DISCOVER_ICON, MODEL_ICON, SETTINGS_KEYS } from '$lib/constants';
	import { modelsStore, settingsStore, uiStore } from '$lib/stores';
	import { onMount } from 'svelte';

	let discoverEnabled = $derived(
		Boolean(settingsStore.getConfig(SETTINGS_KEYS.ENABLE_DISCOVER_MODELS))
	);

	// sync point: the table opens on cached state and corrects itself when the list
	// fetch lands; a no-op outside router mode
	onMount(() => {
		void modelsStore.fetchRouterModels();
	});
</script>

<svelte:head>
	<title>Models - {APP_NAME}</title>
</svelte:head>

<div class="flex h-[100dvh] flex-col p-4 pb-0 max-md:pt-16">
	<h1 class="flex min-h-7 items-center gap-2 p-2 text-lg font-semibold leading-none">
		<MODEL_ICON class="h-5 w-5" />

		<span>Models</span>
	</h1>

	<ModelsManager class="min-h-0">
		{#snippet toolbarEnd()}
			{#if discoverEnabled}
				<Button
					class="gap-1.5"
					onclick={() => uiStore.openModelsDiscover()}
					size="sm"
					variant="tertiary"
				>
					<DISCOVER_ICON class="h-3.5 w-3.5" />

					Discover Models
				</Button>
			{/if}
		{/snippet}
	</ModelsManager>
</div>
