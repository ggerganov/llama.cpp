import React, { createContext, useContext, useEffect, useState } from 'react';
import { APIMessage, Conversation, Message, PendingMessage } from './types';
import StorageUtils from './storage';
import {
  filterThoughtFromMsgs,
  normalizeMsgsForAPI,
  getSSEStreamAsync,
} from './misc';
import { BASE_URL, CONFIG_DEFAULT, isDev } from '../Config';
import { matchPath, useLocation } from 'react-router';

interface AppContextValue {
  isGenerating: boolean;
  viewingConversation: Conversation | null;
  pendingMessage: PendingMessage | null;
  sendMessage: (
    convId: string,
    content: string,
    onChunk?: CallbackGeneratedChunk
  ) => Promise<boolean>;
  stopGenerating: () => void;
  replaceMessageAndGenerate: (
    convId: string,
    origMsgId: Message['id'],
    content?: string,
    onChunk?: CallbackGeneratedChunk
  ) => Promise<void>;

  config: typeof CONFIG_DEFAULT;
  saveConfig: (config: typeof CONFIG_DEFAULT) => void;
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

  const [isGenerating, setIsGenerating] = useState(false);
  const [viewingConversation, setViewingConversation] =
    useState<Conversation | null>(null);
  const [pendingMessage, setPendingMessage] = useState<PendingMessage | null>(
    null
  );
  const [abortController, setAbortController] = useState(new AbortController());
  const [config, setConfig] = useState(StorageUtils.getConfig());

  useEffect(() => {
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

  const generateMessage = async (
    convId: string,
    onChunk?: CallbackGeneratedChunk
  ) => {
    if (isGenerating) return;

    const config = StorageUtils.getConfig();
    const currConversation = StorageUtils.getOneConversation(convId);
    if (!currConversation) {
      throw new Error('Current conversation is not found');
    }

    const abortController = new AbortController();
    setIsGenerating(true);
    setAbortController(abortController);

    let pendingMsg: PendingMessage = {
      convId,
      id: Date.now() + 1,
      role: 'assistant',
      content: null,
    };
    setPendingMessage(pendingMsg);

    try {
      // prepare messages for API
      let messages: APIMessage[] = [
        { role: 'system', content: config.systemMessage },
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
            convId,
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
        setPendingMessage(pendingMsg);
        onChunk?.();
      }
    } catch (err) {
      console.error(err);
      setPendingMessage(null);
      setIsGenerating(false);
      if ((err as Error).name === 'AbortError') {
        // user stopped the generation via stopGeneration() function
        // we can safely ignore this error
      } else {
        setIsGenerating(false);
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
    setPendingMessage(null);
    setIsGenerating(false);
    onChunk?.(); // trigger scroll to bottom
  };

  const sendMessage = async (
    convId: string,
    content: string,
    onChunk?: CallbackGeneratedChunk
  ): Promise<boolean> => {
    if (isGenerating || content.trim().length === 0) return false;

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

  const stopGenerating = () => {
    setIsGenerating(false);
    setPendingMessage(null);
    abortController.abort();
  };

  // if content is undefined, we remove last assistant message
  const replaceMessageAndGenerate = async (
    convId: string,
    origMsgId: Message['id'],
    content?: string,
    onChunk?: CallbackGeneratedChunk
  ) => {
    if (isGenerating) return;

    StorageUtils.filterAndKeepMsgs(convId, (msg) => msg.id < origMsgId);
    if (content) {
      // case: replace user message then generate assistant message
      await sendMessage(convId, content, onChunk);
    } else {
      // case: generate last assistant message
      await generateMessage(convId, onChunk);
    }
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
        pendingMessage,
        sendMessage,
        stopGenerating,
        replaceMessageAndGenerate,
        config,
        saveConfig,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

export const useAppContext = () => useContext(AppContext);
