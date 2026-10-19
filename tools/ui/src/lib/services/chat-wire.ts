/**
 * llama.cpp chat wire format: turns the streamed chunks of a chat completion
 * into canonical events, and reads back the model a payload reports.
 *
 * llama-server answers with `timings` and `prompt_progress` of its own, so the
 * client keeps no timing fallback and asks for no usage chunk.
 */

import type {
	ApiChatCompletionStreamChunk,
	ApiChatCompletionToolCallDelta,
	ApiChatCompletionUsage
} from '$lib/types/api';
import type { ChatMessagePromptProgress, ChatMessageTimings } from '$lib/types/chat';

/** One canonical delta decoded from a backend's stream payload. */
export type ChatStreamEvent =
	| { type: 'done' }
	| { type: 'error'; message: string }
	| { type: 'id'; id: string }
	| { type: 'model'; model: string }
	| { type: 'prompt_progress'; progress: ChatMessagePromptProgress }
	| { type: 'text'; text: string }
	| { type: 'thinking'; text: string }
	| { type: 'timings'; timings: ChatMessageTimings; promptProgress?: ChatMessagePromptProgress }
	| { type: 'tool_calls'; deltas: ApiChatCompletionToolCallDelta[] }
	| { type: 'usage'; usage: ApiChatCompletionUsage };

/** Decodes one stream's payloads. Create one per request. */
export interface ChatStreamReader {
	/** Canonical deltas for one parsed SSE payload. */
	readChunk(payload: unknown): ChatStreamEvent[];
}

/**
 * Model name a payload reports. Streaming chunks carry it on the delta, final
 * responses on the message, and some gateways on metadata or the choice itself.
 */
export function extractModelName(data: unknown): string | undefined {
	const asRecord = (value: unknown): Record<string, unknown> | undefined =>
		typeof value === 'object' && value !== null ? (value as Record<string, unknown>) : undefined;
	const getTrimmedString = (value: unknown): string | undefined =>
		typeof value === 'string' && value.trim() ? value.trim() : undefined;
	const root = asRecord(data);

	if (!root) return undefined;

	const rootModel = getTrimmedString(root.model);

	if (rootModel) return rootModel;

	const firstChoice = Array.isArray(root.choices) ? asRecord(root.choices[0]) : undefined;

	if (!firstChoice) return undefined;

	const metadataModel = getTrimmedString(asRecord(firstChoice.metadata)?.model);

	if (metadataModel) return metadataModel;

	const deltaModel = getTrimmedString(asRecord(firstChoice.delta)?.model);

	if (deltaModel) return deltaModel;

	const messageModel = getTrimmedString(asRecord(firstChoice.message)?.model);

	if (messageModel) return messageModel;

	return getTrimmedString(firstChoice.model);
}

function readChunk(payload: unknown): ChatStreamEvent[] {
	if (!payload || typeof payload !== 'object') return [];

	const chunk = payload as ApiChatCompletionStreamChunk;
	const events: ChatStreamEvent[] = [];
	const model = extractModelName(chunk);

	if (chunk.id) events.push({ id: chunk.id, type: 'id' });

	if (model) events.push({ model, type: 'model' });

	if (chunk.prompt_progress) {
		events.push({ progress: chunk.prompt_progress, type: 'prompt_progress' });
	}

	if (chunk.timings) {
		events.push({
			promptProgress: chunk.prompt_progress,
			timings: chunk.timings,
			type: 'timings'
		});
	}

	if (chunk.usage) events.push({ type: 'usage', usage: chunk.usage });

	const choice = chunk.choices?.[0];
	const delta = choice?.delta;

	if (!delta) return events;

	if (delta.content) events.push({ text: delta.content, type: 'text' });

	// endpoints disagree on the reasoning field name; take the first one set
	const reasoning = delta.reasoning_content ?? delta.reasoning ?? delta.reasoning_text;

	if (reasoning) events.push({ text: reasoning, type: 'thinking' });

	if (delta.tool_calls) events.push({ deltas: delta.tool_calls, type: 'tool_calls' });

	return events;
}

/** New streaming reader for one request. */
export function createChatStreamReader(): ChatStreamReader {
	return { readChunk };
}
