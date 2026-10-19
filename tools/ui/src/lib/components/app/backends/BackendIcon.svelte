<script lang="ts">
	import type { Backend } from '$lib/types';
	import { backendFaviconUrl } from '$lib/utils';
	import type { Snippet } from 'svelte';
	import { SvelteSet } from 'svelte/reactivity';

	/** Favicons that failed to load, so a row remount does not ask again. */
	const failedFavicons = new SvelteSet<string>();

	interface Props {
		backend?: Backend;
		class?: string;
		/** Rendered when the backend serves no favicon. */
		fallback?: Snippet;
	}

	let { backend, class: className = 'size-4', fallback }: Props = $props();

	let faviconUrl = $derived(backend ? backendFaviconUrl(backend.baseUrl) : null);
</script>

{#if faviconUrl && !failedFavicons.has(faviconUrl)}
	<img
		alt=""
		class={['shrink-0 rounded-sm object-contain', className]}
		decoding="async"
		loading="lazy"
		onerror={() => failedFavicons.add(faviconUrl)}
		src={faviconUrl}
	/>
{:else}
	{@render fallback?.()}
{/if}
