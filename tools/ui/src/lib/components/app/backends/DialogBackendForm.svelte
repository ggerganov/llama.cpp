<script lang="ts">
	import BackendForm from './BackendForm.svelte';
	import { CheckCircle2, Loader2, PlugZap, XCircle } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Dialog from '$lib/components/ui/dialog';
	import { BACKEND_ID_PREFIX } from '$lib/constants';
	import { BackendsService } from '$lib/services';
	import type { BackendTestResult } from '$lib/services/backends.service';
	import { backendsStore } from '$lib/stores';
	import type { Backend } from '$lib/types';
	import { uuid } from '$lib/utils';

	interface Props {
		backend?: Backend | null;
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
		onSaved?: (backend: Backend) => void;
	}

	let { backend = null, onOpenChange, onSaved, open = $bindable(false) }: Props = $props();

	let draft = $state<Backend>(createBackend());
	let testResult = $state<BackendTestResult | null>(null);
	let testing = $state(false);
	let apiKeyRequired = $state(false);

	let isEdit = $derived(backend !== null);
	let urlError = $derived.by(() => {
		const url = draft.baseUrl.trim();

		if (!url) return 'Base URL is required';

		try {
			new URL(url);

			return null;
		} catch {
			return 'Invalid URL format';
		}
	});
	let canSave = $derived(
		!urlError && draft.name.trim().length > 0 && (!apiKeyRequired || Boolean(draft.apiKey?.trim()))
	);

	// reset the draft each time the dialog opens
	$effect(() => {
		if (!open) return;

		draft = backend ? { ...backend } : createBackend();
		testResult = null;
		testing = false;
		apiKeyRequired = false;
	});

	// Once the URL settles, ask the endpoint whether it guards /props with a key, the
	// way the MCP dialog previews a server. A refused probe is retried once a key is
	// typed, which is the answer the API-key field waits for.
	$effect(() => {
		const url = draft.baseUrl.trim();
		const apiKey = draft.apiKey?.trim();

		if (!open || isEdit || urlError || !url) {
			apiKeyRequired = false;

			return;
		}

		const timer = setTimeout(async () => {
			const probe = await BackendsService.probe({ ...draft, apiKey, baseUrl: url });

			apiKeyRequired = probe.authRequired;
		}, 600);

		return () => clearTimeout(timer);
	});

	function createBackend(): Backend {
		return {
			baseUrl: '',
			enabled: true,
			id: uuid() || `${BACKEND_ID_PREFIX}-${Date.now()}`,
			name: ''
		};
	}

	function handleChange(patch: Partial<Backend>) {
		draft = { ...draft, ...patch };
		testResult = null;
	}

	function handleOpenChange(value: boolean) {
		open = value;
		onOpenChange?.(value);
	}

	async function handleTest() {
		if (urlError) return;

		testing = true;
		testResult = null;

		try {
			testResult = await BackendsService.test(draft);
		} finally {
			testing = false;
		}
	}

	function handleSave() {
		if (!canSave) return;

		const next: Backend = { ...draft, name: draft.name.trim() || hostOf(draft.baseUrl) };

		if (isEdit && backend) {
			backendsStore.updateBackend(backend.id, next);
		} else {
			backendsStore.addBackend(next);
		}

		onSaved?.(next);
		handleOpenChange(false);
	}

	function handleSubmit(event: SubmitEvent) {
		event.preventDefault();
		handleSave();
	}

	function hostOf(url: string): string {
		try {
			return new URL(url).host;
		} catch {
			return url;
		}
	}
</script>

<Dialog.Root onOpenChange={handleOpenChange} {open}>
	<Dialog.Content class="max-w-2xl!">
		<Dialog.Header>
			<Dialog.Title class="select-none">
				{isEdit ? 'Edit Provider' : 'Add New Provider'}
			</Dialog.Title>

			<Dialog.Description>Point at another llama-server.</Dialog.Description>
		</Dialog.Header>

		<form class="contents" onsubmit={handleSubmit}>
			<div class="space-y-4 py-4">
				<BackendForm
					{apiKeyRequired}
					backend={draft}
					id="backend"
					onChange={handleChange}
					{urlError}
				/>
			</div>

			<!-- the last thing before the actions: what the endpoint said when asked -->
			<div
				aria-live="polite"
				class="mb-3 flex items-start gap-2.5 rounded-lg border border-border/60 bg-muted/30 p-3"
			>
				{#if testing}
					<Loader2 class="mt-0.5 h-4 w-4 shrink-0 animate-spin text-muted-foreground" />

					<div class="min-w-0">
						<p class="text-sm font-medium">Testing connection</p>

						<p class="text-xs text-muted-foreground">
							Asking {hostOf(draft.baseUrl)} for the models it serves.
						</p>
					</div>
				{:else if testResult?.ok}
					<CheckCircle2 class="mt-0.5 h-4 w-4 shrink-0 text-emerald-500" />

					<div class="min-w-0">
						<p class="text-sm font-medium">Connected</p>

						<p class="text-xs text-muted-foreground">
							{testResult.modelCount ?? 0}
							model{(testResult.modelCount ?? 0) === 1 ? '' : 's'} available.
						</p>
					</div>
				{:else if testResult}
					<XCircle class="mt-0.5 h-4 w-4 shrink-0 text-destructive" />

					<div class="min-w-0">
						<p class="text-sm font-medium text-destructive">Connection failed</p>

						<p class="text-xs break-words text-muted-foreground">
							{testResult.error ?? 'The endpoint did not answer.'}{testResult.status
								? ` (HTTP ${testResult.status})`
								: ''}
						</p>
					</div>
				{:else}
					<PlugZap class="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />

					<div class="min-w-0">
						<p class="text-sm font-medium">Connection not tested</p>

						<p class="text-xs text-muted-foreground">
							Test it to list the models this provider serves.
						</p>
					</div>
				{/if}
			</div>

			<Dialog.Footer>
				<Button onclick={() => handleOpenChange(false)} size="sm" variant="secondary">
					Cancel
				</Button>

				<Button disabled={testing} onclick={handleTest} size="sm" type="button" variant="outline">
					Test connection
				</Button>

				<Button disabled={!canSave} size="sm" type="submit">
					{isEdit ? 'Save' : 'Add'}
				</Button>
			</Dialog.Footer>
		</form>
	</Dialog.Content>
</Dialog.Root>
