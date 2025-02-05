import React, { createContext, useContext, useEffect, useState } from 'react';
import { APIMessage, Conversation, Message, PendingMessage } from './types';
import StorageUtils from './storage';
import {
  filterThoughtFromMsgs,
  normalizeMsgsForAPI,
  sendSSEPostRequest,
} from './misc';
import { BASE_URL, CONFIG_DEFAULT, isDev } from '../Config';
import { matchPath, useLocation, useParams } from 'react-router';

interface AppContextValue {
  isGenerating: boolean;
  viewingConversation: Conversation | null;
  pendingMessage: PendingMessage | null;
  sendMessage: (convId: string, content: string) => Promise<void>;
  stopGenerating: () => void;
  replaceMessageAndGenerate: (
    convId: string,
    origMsgId: Message['id'],
    content?: string
  ) => Promise<void>;

  config: typeof CONFIG_DEFAULT;
  saveConfig: (config: typeof CONFIG_DEFAULT) => void;
}

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

  const generateMessage = async (convId: string) => {
    if (isGenerating) return;

    const config = StorageUtils.getConfig();
    const currConversation = StorageUtils.getOneConversation(convId);
    if (!currConversation) {
      throw new Error('Current conversation is not found');
    }

    setIsGenerating(true);
    setAbortController(new AbortController());

    let pendingMsg: PendingMessage = {
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
      const chunks = sendSSEPostRequest(`${BASE_URL}/v1/chat/completions`, {
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
      for await (const chunk of chunks) {
        // const stop = chunk.stop;
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
        setPendingMessage(pendingMsg);
      }
    } catch (err) {
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
        ...pendingMsg,
        content: pendingMsg.content,
      });
    }
    setPendingMessage(null);
    setIsGenerating(false);
  };

  const sendMessage = async (convId: string, content: string) => {
    if (isGenerating || content.trim().length === 0) return;

    StorageUtils.appendMsg(convId, {
      id: Date.now(),
      role: 'user',
      content,
    });
    try {
      await generateMessage(convId);
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
    } catch (_) {
      // rollback
      StorageUtils.popMsg(convId);
    }
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
    content?: string
  ) => {
    if (isGenerating) return;

    if (content) {
      StorageUtils.filterAndKeepMsgs(convId, (msg) => msg.id < origMsgId);
      await sendMessage(convId, content);
    } else {
      StorageUtils.filterAndKeepMsgs(convId, (msg) => msg.id < origMsgId);
      await generateMessage(convId);
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

// eslint-disable-next-line
export const useAppContext = () => useContext(AppContext);
