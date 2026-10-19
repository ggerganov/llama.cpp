<script lang="ts">
	import { Input } from '$lib/components/ui/input';
	import type { Backend } from '$lib/types';

	interface Props {
		backend: Backend;
		/** The endpoint refused the probe for want of a key. */
		apiKeyRequired?: boolean;
		id: string;
		onChange: (patch: Partial<Backend>) => void;
		urlError?: string | null;
	}

	let { apiKeyRequired = false, backend, id, onChange, urlError = null }: Props = $props();
</script>

<div class="grid gap-5">
	<div>
		<label class="mb-2 block text-xs font-medium select-none" for="backend-url-{id}">
			Base URL <span class="text-destructive">*</span>
		</label>

		<Input
			class={urlError ? 'border-destructive' : ''}
			id="backend-url-{id}"
			oninput={(e) => onChange({ baseUrl: e.currentTarget.value })}
			placeholder="https://api.example.com"
			type="url"
			value={backend.baseUrl}
		/>

		{#if urlError}
			<p class="mt-1.5 text-xs text-destructive">{urlError}</p>
		{/if}
	</div>

	<div class="grid gap-5 sm:grid-cols-2">
		<div>
			<label class="mb-2 block text-xs font-medium select-none" for="backend-name-{id}">
				Display name
			</label>

			<Input
				id="backend-name-{id}"
				oninput={(e) => onChange({ name: e.currentTarget.value })}
				placeholder="Name shown in the model selector"
				type="text"
				value={backend.name}
			/>
		</div>

		<div>
			<label class="mb-2 block text-xs font-medium select-none" for="backend-key-{id}">
				API key{#if apiKeyRequired}<span class="text-destructive"> *</span>{/if}
			</label>

			<Input
				autocomplete="off"
				id="backend-key-{id}"
				oninput={(e) => onChange({ apiKey: e.currentTarget.value || undefined })}
				placeholder="Optional"
				type="password"
				value={backend.apiKey ?? ''}
			/>

			<p class="mt-1.5 text-xs text-muted-foreground">
				{#if apiKeyRequired}
					Required by this endpoint.
				{:else}
					Sent as a Bearer token.
				{/if}
			</p>
		</div>
	</div>
</div>
