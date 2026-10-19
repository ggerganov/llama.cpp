/**
 * Backend list parsing, defaults and endpoint URLs.
 *
 * External backends are persisted in settings as a JSON list. Malformed
 * entries are dropped instead of throwing so a corrupted settings value can
 * never break URL resolution.
 */

import {
	BACKEND_ID_PREFIX,
	DEFAULT_BACKEND_CHAT_PATH,
	DEFAULT_BACKEND_MODELS_PATH,
	FAVICON_SERVICE_URL,
	LOCAL_BACKEND_ID
} from '$lib/constants';
import type { Backend } from '$lib/types';

/** Absolute chat completions URL for a backend. */
export function backendChatUrl(backend: Backend): string {
	return joinBackendUrl(backend.baseUrl, DEFAULT_BACKEND_CHAT_PATH);
}

/** Absolute models listing URL for a backend. */
export function backendModelsUrl(backend: Backend): string {
	return joinBackendUrl(backend.baseUrl, DEFAULT_BACKEND_MODELS_PATH);
}

/** Icon size requested from the favicon service, shown at 16px. */
const FAVICON_SIZE = 64;

/**
 * Favicon of a backend's root domain, used when no bundled mark matches. Hosts
 * rarely serve a favicon themselves, so the subdomain is dropped
 * (`api.z.ai` -> `z.ai`) and the icon is requested through a favicon service.
 * Returns null when the URL carries no usable host.
 */
export function backendFaviconUrl(baseUrl: string): string | null {
	try {
		const labels = new URL(baseUrl).hostname.split('.');
		const root = labels.length > 2 ? labels.slice(-2).join('.') : labels.join('.');

		return root ? `${FAVICON_SERVICE_URL}${root}&sz=${FAVICON_SIZE}` : null;
	} catch {
		return null;
	}
}

/** The built-in backend pointing at the server that serves this UI. */
export function createLocalBackend(apiKey?: string, enabled = true): Backend {
	return {
		apiKey,
		baseUrl: '',
		enabled,
		id: LOCAL_BACKEND_ID,
		name: 'Local'
	};
}

/**
 * Parse the persisted backends JSON into backend entries.
 */
export function parseBackendsSettings(rawBackends: unknown): Backend[] {
	if (!rawBackends) return [];

	let parsed: unknown;

	if (typeof rawBackends === 'string') {
		const trimmed = rawBackends.trim();

		if (!trimmed) return [];

		try {
			parsed = JSON.parse(trimmed);
		} catch (error) {
			console.warn('[backends] Failed to parse backends JSON, ignoring value:', error);

			return [];
		}
	} else {
		parsed = rawBackends;
	}

	if (!Array.isArray(parsed)) return [];

	return parsed.flatMap((entry, index) => {
		const backend = parseBackendEntry(entry, index);

		return backend ? [backend] : [];
	});
}

function joinBackendUrl(baseUrl: string, path: string): string {
	const base = baseUrl.replace(/\/+$/, '');
	const suffix = path.startsWith('/') ? path : `/${path}`;

	return `${base}${suffix}`;
}

function parseBackendEntry(entry: unknown, index: number): Backend | null {
	if (!entry || typeof entry !== 'object') return null;

	const raw = entry as Record<string, unknown>;
	const baseUrl = typeof raw.baseUrl === 'string' ? raw.baseUrl.trim() : '';

	// the local backend is built in and never persisted
	if (!baseUrl || raw.id === LOCAL_BACKEND_ID) return null;

	const id =
		typeof raw.id === 'string' && raw.id.trim()
			? raw.id.trim()
			: `${BACKEND_ID_PREFIX}-${index + 1}`;
	const name = typeof raw.name === 'string' && raw.name.trim() ? raw.name.trim() : baseUrl;
	const apiKey =
		typeof raw.apiKey === 'string' && raw.apiKey.trim() ? raw.apiKey.trim() : undefined;

	return {
		apiKey,
		baseUrl,
		enabled: raw.enabled !== false,
		headers: parseBackendHeaders(raw.headers),
		id,
		name
	};
}

function parseBackendHeaders(raw: unknown): Record<string, string> | undefined {
	if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return undefined;

	const entries = Object.entries(raw as Record<string, unknown>)
		.filter(([, value]) => typeof value === 'string' && value.trim() !== '')
		.map(([key, value]) => [key.trim(), (value as string).trim()] as const);

	return entries.length > 0 ? Object.fromEntries(entries) : undefined;
}

/**
 * Read a model's context size out of one `/v1/models` entry. A server leaves it
 * unknown when it has not loaded the model, and a model-mode listing reports the
 * trained size under `meta.n_ctx_train` instead.
 */
export function readModelContextLength(value: unknown): number | undefined {
	if (!value || typeof value !== 'object') return undefined;

	const entry = value as Record<string, unknown>;

	if (typeof entry.context_length === 'number' && entry.context_length > 0) {
		return entry.context_length;
	}

	const meta = entry.meta;

	if (meta && typeof meta === 'object') {
		const trained = (meta as Record<string, unknown>).n_ctx_train;

		if (typeof trained === 'number' && trained > 0) return trained;
	}

	return undefined;
}
