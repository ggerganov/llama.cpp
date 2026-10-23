import type { BackendStatus } from '$lib/enums';

/**
 * A llama.cpp server the UI talks to: the one serving this app, or one the
 * user added. Every backend speaks the llama.cpp API, so nothing is negotiated
 * at runtime.
 */
export interface Backend {
	/** Sent as `Authorization: Bearer <apiKey>` when set. */
	apiKey?: string;
	/** Endpoint origin, no trailing slash; empty for the built-in backend. */
	baseUrl: string;
	enabled: boolean;
	/** Extra request headers sent verbatim, e.g. an authenticating proxy. */
	headers?: Record<string, string>;
	id: string;
	modelCount?: number;
	name: string;
	/** Result of the last connectivity check; undefined until it is tested. */
	status?: BackendStatus;
	/** Last connectivity or model-list failure. */
	error?: string;
}
