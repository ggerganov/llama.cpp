<script lang="ts">
	import ChatScreenActionScrollDown from './ChatScreenActionScrollDown.svelte';
	import ChatScreenDialogsAndAlerts from './ChatScreenDialogsAndAlerts.svelte';
	import ChatScreenGreeting from './ChatScreenGreeting.svelte';
	import { page } from '$app/state';
	import {
		ChatMessages,
		ChatScreenDragOverlay,
		ChatScreenForm,
		ChatScreenServerError,
		ChatScreenStreamResumeStatus,
		ServerLoadingSplash
	} from '$lib/components/app';
	import { ROUTES } from '$lib/constants';
	import { useChatScreenActiveModel } from '$lib/hooks/use-chat-screen-active-model.svelte';
	import { useChatScreenDragAndDrop } from '$lib/hooks/use-chat-screen-drag-and-drop.svelte';
	import { useChatScreenFileUpload } from '$lib/hooks/use-chat-screen-file-upload.svelte';
	import { useKeyboardShortcuts } from '$lib/hooks/use-keyboard-shortcuts.svelte';
	import { createStickToBottom } from '$lib/hooks/use-stick-to-bottom.svelte';
	import {
		chatStore,
		conversationsStore,
		deviceStore,
		serverStore,
		settingsStore
	} from '$lib/stores';
	import { parseFilesToMessageExtras } from '$lib/utils/browser-only';
	import { onMount, tick } from 'svelte';

	let { showCenteredEmpty = false } = $props();

	let emptyFileNames = $state<string[]>([]);
	let initialMessage = $state('');
	let showDeleteDialog = $state(false);
	let showEmptyFileDialog = $state(false);
	let isEmpty = $derived(
		showCenteredEmpty && conversationsStore.activeMessages.length === 0 && !chatStore.isLoading
	);
	let activeErrorDialog = $derived(chatStore.errorDialogState);
	let isServerLoading = $derived(serverStore.loading);
	let hasPropsError = $derived(!!serverStore.error);
	let isCurrentConversationLoading = $derived(chatStore.isLoading || chatStore.isStreaming());
	let chatFormBottomPosition = $derived.by(() => {
		if (!deviceStore.isMobile) return '1rem';

		if (deviceStore.isStandalone) return '1.5rem';

		if (deviceStore.isIOSSafari) return '0.25rem';

		return '0.5rem';
	});

	// The page is the scroller: it follows new content while the reader sits at
	// the bottom, and otherwise leaves their position alone.
	const stick = createStickToBottom();
	const activeModel = useChatScreenActiveModel();
	const fileUpload = useChatScreenFileUpload({
		activeModelId: () => activeModel.activeModelId,
		capabilities: () => ({
			hasAudio: activeModel.hasAudioModality,
			hasVideo: activeModel.hasVideoModality,
			hasVision: activeModel.hasVisionModality
		})
	});
	const dragAndDrop = useChatScreenDragAndDrop({
		onDrop: fileUpload.handleFileUpload
	});
	const { handleKeydown } = useKeyboardShortcuts({
		deleteActiveConversation: () => {
			if (conversationsStore.activeConversation) {
				showDeleteDialog = true;
			}
		}
	});

	async function handleDeleteConfirm() {
		const conversation = conversationsStore.activeConversation;

		if (conversation) {
			await conversationsStore.deleteConversation(conversation.id);
		}

		showDeleteDialog = false;
	}

	async function handleSendMessage(message: string, files?: ChatUploadedFile[]): Promise<boolean> {
		const plainFiles = files ? $state.snapshot(files) : undefined;
		const result = plainFiles
			? await parseFilesToMessageExtras(plainFiles, activeModel.activeModelId ?? undefined)
			: undefined;

		if (result?.emptyFiles && result.emptyFiles.length > 0) {
			emptyFileNames = result.emptyFiles;
			showEmptyFileDialog = true;

			if (files) {
				const emptyFileNamesSet = new Set(result.emptyFiles);

				fileUpload.uploadedFiles = fileUpload.uploadedFiles.filter(
					(file) => !emptyFileNamesSet.has(file.name)
				);
			}

			return false;
		}

		// Sending re-arms following: the reply streams in under the message.
		stick.scrollToBottom();

		await chatStore.sendMessage(message, result?.extras);

		return true;
	}

	let lastScrolledConversationId: string | null = null;

	// Lands at the bottom of a conversation the first time its messages render.
	// The resize observer keeps it there while the list settles (lazy rows,
	// syntax highlighting, images) without polling for a stable height.
	async function handleMessagesReady(messageCount: number) {
		if (messageCount === 0) return;

		const id = conversationsStore.activeConversation?.id ?? null;

		if (!id || id === lastScrolledConversationId) return;

		lastScrolledConversationId = id;
		await tick();
		stick.scrollToBottom();
	}

	function handleErrorDialogOpenChange(open: boolean) {
		if (!open) {
			chatStore.dismissErrorDialog();
		}
	}

	async function handleSystemPromptAdd(draft: { message: string; files: ChatUploadedFile[] }) {
		if (draft.message || draft.files.length > 0) {
			chatStore.savePendingDraft(draft.message, draft.files);
		}

		await chatStore.addSystemPrompt();
	}

	$effect(() => {
		stick.setEnabled(!settingsStore.config.disableAutoScroll);
	});

	$effect(() => {
		stick.attach(document.documentElement, document.body);

		return () => stick.detach();
	});

	onMount(() => {
		const pendingDraft = chatStore.consumePendingDraft();

		if (pendingDraft) {
			initialMessage = pendingDraft.message;
			fileUpload.uploadedFiles = pendingDraft.files;
		}
	});
</script>

{#if dragAndDrop.isDragOver}
	<ChatScreenDragOverlay />
{/if}

<svelte:window onkeydown={handleKeydown} onscroll={() => stick.handleScroll()} />

{#if isServerLoading}
	<ServerLoadingSplash />
{:else}
	<div
		style:--chat-form-bottom-position={chatFormBottomPosition}
		class="chat-screen flex grow flex-col min-h-[calc(100dvh-1rem)] md:min-h-[calc(100dvh-1rem-var(--chat-tabs-offset,0px))] px-4 md:py-0 pt-12 pb-48 md:pb-4"
		ondragenter={dragAndDrop.dragHandlers.dragenter}
		ondragleave={dragAndDrop.dragHandlers.dragleave}
		ondragover={dragAndDrop.dragHandlers.dragover}
		ondrop={dragAndDrop.dragHandlers.drop}
		role="main"
	>
		{#if !isEmpty}
			<ChatMessages
				messages={conversationsStore.activeMessages}
				onMessagesReady={handleMessagesReady}
			/>
		{/if}

		<div
			style:padding-top={!isEmpty ? 'var(--chat-form-padding-top)' : undefined}
			class={[
				// animate the centered->bottomed move with transform, not bottom:
				// layout-property transitions need the main thread every frame and
				// stutter while a long conversation loads; transform transitions
				// run on the compositor and stay smooth
				'pointer-events-none md:sticky fixed  mt-auto transition-transform duration-200',
				deviceStore.isStandalone
					? 'bottom-6 right-4 left-4'
					: deviceStore.isIOSSafari
						? 'bottom-1 left-2 right-2'
						: 'bottom-2 right-2 left-2',
				'md:bottom-4',
				isEmpty ? 'md:translate-y-[calc(-50dvh+8rem)] 2xl:translate-y-[calc(-50dvh+5rem)]' : ''
			]}
		>
			<ChatScreenGreeting {isEmpty} />

			<ChatScreenServerError />

			{#if page.params.id}
				<ChatScreenStreamResumeStatus />
			{/if}

			<div class="pointer-events-none flex flex-col gap-6 items-center w-full">
				{#if !stick.atBottom && page.url.hash.includes(ROUTES.CHAT) && page.params.id}
					<ChatScreenActionScrollDown onclick={() => stick.scrollToBottom('smooth')} />
				{/if}
			</div>

			<ChatScreenForm
				bind:uploadedFiles={fileUpload.uploadedFiles}
				class="pointer-events-auto conversation-chat-form"
				disabled={hasPropsError || chatStore.isEditing()}
				{initialMessage}
				isLoading={isCurrentConversationLoading}
				onFileRemove={fileUpload.handleFileRemove}
				onFileUpload={fileUpload.handleFileUpload}
				onSend={handleSendMessage}
				onStop={() => chatStore.stopGeneration()}
				onSystemPromptAdd={handleSystemPromptAdd}
			/>
		</div>
	</div>
{/if}

<ChatScreenDialogsAndAlerts
	{activeErrorDialog}
	{emptyFileNames}
	{fileUpload}
	{handleDeleteConfirm}
	{handleErrorDialogOpenChange}
	{showDeleteDialog}
	{showEmptyFileDialog}
/>
