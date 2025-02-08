import React, { createContext, useContext, useEffect, useState } from 'react';
import {
  APIMessage,
  CanvasData,
  Conversation,
  Message,
  PendingMessage,
} from './types';
import StorageUtils from './storage';
import {
  filterThoughtFromMsgs,
  normalizeMsgsForAPI,
  getSSEStreamAsync,
} from './misc';
import { BASE_URL, CONFIG_DEFAULT, isDev } from '../Config';
import { matchPath, useLocation } from 'react-router';

interface AppContextValue {
  // conversations and messages
  viewingConversation: Conversation | null;
  pendingMessages: Record<Conversation['id'], PendingMessage>;
  isGenerating: (convId: string) => boolean;
  sendMessage: (
    convId: string,
    content: string,
    onChunk?: CallbackGeneratedChunk
  ) => Promise<boolean>;
  stopGenerating: (convId: string) => void;
  replaceMessageAndGenerate: (
    convId: string,
    origMsgId: Message['id'],
    content?: string,
    onChunk?: CallbackGeneratedChunk
  ) => Promise<void>;

  // canvas
  canvasData: CanvasData | null;
  setCanvasData: (data: CanvasData | null) => void;

  // config
  config: typeof CONFIG_DEFAULT;
  saveConfig: (config: typeof CONFIG_DEFAULT) => void;
  showSettings: boolean;
  setShowSettings: (show: boolean) => void;
}

// for now, this callback is only used for scrolling to the bottom of the chat
type CallbackGeneratedChunk = () => void;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const AppContext = createContext<AppContextValue>({} as any);

export const AppContextProvider = ({
  children,
}: {
  children: React.ReactElement;
}) => {
  const { pathname } = useLocation();
  const params = matchPath('/chat/:convId', pathname);
  const convId = params?.params?.convId;

  const [viewingConversation, setViewingConversation] =
    useState<Conversation | null>(null);
  const [pendingMessages, setPendingMessages] = useState<
    Record<Conversation['id'], PendingMessage>
  >({});
  const [aborts, setAborts] = useState<
    Record<Conversation['id'], AbortController>
  >({});
  const [config, setConfig] = useState(StorageUtils.getConfig());
  const [canvasData, setCanvasData] = useState<CanvasData | null>(null);
  const [showSettings, setShowSettings] = useState(false);

  // handle change when the convId from URL is changed
  useEffect(() => {
    // also reset the canvas data
    setCanvasData(null);
    const handleConversationChange = (changedConvId: string) => {
      if (changedConvId !== convId) return;
      setViewingConversation(StorageUtils.getOneConversation(convId));
    };
    StorageUtils.onConversationChanged(handleConversationChange);
    setViewingConversation(StorageUtils.getOneConversation(convId ?? ''));
    return () => {
      StorageUtils.offConversationChanged(handleConversationChange);
    };
  }, [convId]);

  const setPending = (convId: string, pendingMsg: PendingMessage | null) => {
    // if pendingMsg is null, remove the key from the object
    if (!pendingMsg) {
      setPendingMessages((prev) => {
        const newState = { ...prev };
        delete newState[convId];
        return newState;
      });
    } else {
      setPendingMessages((prev) => ({ ...prev, [convId]: pendingMsg }));
    }
  };

  const setAbort = (convId: string, controller: AbortController | null) => {
    if (!controller) {
      setAborts((prev) => {
        const newState = { ...prev };
        delete newState[convId];
        return newState;
      });
    } else {
      setAborts((prev) => ({ ...prev, [convId]: controller }));
    }
  };

  ////////////////////////////////////////////////////////////////////////
  // public functions

  const isGenerating = (convId: string) => !!pendingMessages[convId];

  const generateMessage = async (
    convId: string,
    onChunk?: CallbackGeneratedChunk
  ) => {
    if (isGenerating(convId)) return;

    const config = StorageUtils.getConfig();
    const currConversation = StorageUtils.getOneConversation(convId);
    if (!currConversation) {
      throw new Error('Current conversation is not found');
    }

    const abortController = new AbortController();
    setAbort(convId, abortController);

    let pendingMsg: PendingMessage = {
      id: Date.now() + 1,
      role: 'assistant',
      content: null,
    };
    setPending(convId, pendingMsg);

    try {
      // prepare messages for API
      let messages: APIMessage[] = [
        ...(config.systemMessage.length === 0
          ? []
          : [{ role: 'system', content: config.systemMessage } as APIMessage]),
        ...normalizeMsgsForAPI(currConversation?.messages ?? []),
      ];
      if (config.excludeThoughtOnReq) {
        messages = filterThoughtFromMsgs(messages);
      }
      if (isDev) console.log({ messages });

      // prepare params
      const params = {
        messages,
        stream: true,
        cache_prompt: true,
        samplers: config.samplers,
        temperature: config.temperature,
        dynatemp_range: config.dynatemp_range,
        dynatemp_exponent: config.dynatemp_exponent,
        top_k: config.top_k,
        top_p: config.top_p,
        min_p: config.min_p,
        typical_p: config.typical_p,
        xtc_probability: config.xtc_probability,
        xtc_threshold: config.xtc_threshold,
        repeat_last_n: config.repeat_last_n,
        repeat_penalty: config.repeat_penalty,
        presence_penalty: config.presence_penalty,
        frequency_penalty: config.frequency_penalty,
        dry_multiplier: config.dry_multiplier,
        dry_base: config.dry_base,
        dry_allowed_length: config.dry_allowed_length,
        dry_penalty_last_n: config.dry_penalty_last_n,
        max_tokens: config.max_tokens,
        timings_per_token: !!config.showTokensPerSecond,
        ...(config.custom.length ? JSON.parse(config.custom) : {}),
      };

      // send request
      const fetchResponse = await fetch(`${BASE_URL}/v1/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(config.apiKey
            ? { Authorization: `Bearer ${config.apiKey}` }
            : {}),
        },
        body: JSON.stringify(params),
        signal: abortController.signal,
      });
      if (fetchResponse.status !== 200) {
        const body = await fetchResponse.json();
        throw new Error(body?.error?.message || 'Unknown error');
      }
      const chunks = getSSEStreamAsync(fetchResponse);
      for await (const chunk of chunks) {
        // const stop = chunk.stop;
        if (chunk.error) {
          throw new Error(chunk.error?.message || 'Unknown error');
        }
        const addedContent = chunk.choices[0].delta.content;
        const lastContent = pendingMsg.content || '';
        if (addedContent) {
          pendingMsg = {
            id: pendingMsg.id,
            role: 'assistant',
            content: lastContent + addedContent,
          };
        }
        const timings = chunk.timings;
        if (timings && config.showTokensPerSecond) {
          // only extract what's really needed, to save some space
          pendingMsg.timings = {
            prompt_n: timings.prompt_n,
            prompt_ms: timings.prompt_ms,
            predicted_n: timings.predicted_n,
            predicted_ms: timings.predicted_ms,
          };
        }
        setPending(convId, pendingMsg);
        onChunk?.();
      }
    } catch (err) {
      setPending(convId, null);
      if ((err as Error).name === 'AbortError') {
        // user stopped the generation via stopGeneration() function
        // we can safely ignore this error
      } else {
        console.error(err);
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        alert((err as any)?.message ?? 'Unknown error');
        throw err; // rethrow
      }
    }

    if (pendingMsg.content) {
      StorageUtils.appendMsg(currConversation.id, {
        id: pendingMsg.id,
        content: pendingMsg.content,
        role: pendingMsg.role,
        timings: pendingMsg.timings,
      });
    }
    setPending(convId, null);
    onChunk?.(); // trigger scroll to bottom
  };

  const sendMessage = async (
    convId: string,
    content: string,
    onChunk?: CallbackGeneratedChunk
  ): Promise<boolean> => {
    if (isGenerating(convId) || content.trim().length === 0) return false;

    StorageUtils.appendMsg(convId, {
      id: Date.now(),
      role: 'user',
      content,
    });

    try {
      await generateMessage(convId, onChunk);
      return true;
    } catch (_) {
      // rollback
      StorageUtils.popMsg(convId);
    }
    return false;
  };

  const stopGenerating = (convId: string) => {
    setPending(convId, null);
    aborts[convId]?.abort();
  };

  // if content is undefined, we remove last assistant message
  const replaceMessageAndGenerate = async (
    convId: string,
    origMsgId: Message['id'],
    content?: string,
    onChunk?: CallbackGeneratedChunk
  ) => {
    if (isGenerating(convId)) return;

    StorageUtils.filterAndKeepMsgs(convId, (msg) => msg.id < origMsgId);
    if (content) {
      StorageUtils.appendMsg(convId, {
        id: Date.now(),
        role: 'user',
        content,
      });
    }

    await generateMessage(convId, onChunk);
  };

  const saveConfig = (config: typeof CONFIG_DEFAULT) => {
    StorageUtils.setConfig(config);
    setConfig(config);
  };

  return (
    <AppContext.Provider
      value={{
        isGenerating,
        viewingConversation,
        pendingMessages,
        sendMessage,
        stopGenerating,
        replaceMessageAndGenerate,
        canvasData,
        setCanvasData,
        config,
        saveConfig,
        showSettings,
        setShowSettings,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

export const useAppContext = () => useContext(AppContext);
