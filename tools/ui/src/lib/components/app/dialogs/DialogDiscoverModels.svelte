<script lang="ts">
	import ModelsDiscover from '$lib/components/app/models/discover/ModelsDiscover.svelte';
	import * as Dialog from '$lib/components/ui/dialog';
	import { DISCOVER_ICON, SETTINGS_KEYS } from '$lib/constants';
	import { settingsStore } from '$lib/stores';
	import { untrack } from 'svelte';

	interface Props {
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
	}

	let { onOpenChange, open = $bindable(false) }: Props = $props();

	// the dialog exists only while its setting is on
	let discoverEnabled = $derived(
		Boolean(settingsStore.getConfig(SETTINGS_KEYS.ENABLE_DISCOVER_MODELS))
	);

	$effect(() => {
		if (open && !discoverEnabled) {
			untrack(() => handleOpenChange(false));
		}
	});

	function handleOpenChange(value: boolean) {
		open = value;
		onOpenChange?.(value);
	}
</script>

<Dialog.Root onOpenChange={handleOpenChange} {open}>
	<Dialog.Content
		class="max-md:h-[100dvh]! max-md:w-screen! max-md:max-w-none! max-md:rounded-none! max-md:border-0! md:h-[calc(100vh-4rem)]! md:max-h-240! md:w-[calc(100vw-4rem)]! md:max-w-380! flex flex-col p-4 pb-0"
		onOpenAutoFocus={(event) => event.preventDefault()}
	>
		<Dialog.Header class="flex flex-row items-center p-2 pr-8">
			<Dialog.Title class="flex min-h-7 items-center gap-2">
				<DISCOVER_ICON class="h-5 w-5" />

				<span>Discover</span>
			</Dialog.Title>

			<Dialog.Description class="sr-only">
				Find models on Hugging Face and download them to the server.
			</Dialog.Description>
		</Dialog.Header>

		<div class="grid min-h-0 flex-1 overflow-hidden" style="grid-template-columns: auto 1fr;">
			<ModelsDiscover />
		</div>
	</Dialog.Content>
</Dialog.Root>
