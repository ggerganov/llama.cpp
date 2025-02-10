// coversations is stored in localStorage
// format: { [convId]: { id: string, lastModified: number, messages: [...] } }

import { CONFIG_DEFAULT } from '../Config';
import { Conversation, Message } from './types';

const event = new EventTarget();

type CallbackConversationChanged = (convId: string) => void;
let onConversationChangedHandlers: [
  CallbackConversationChanged,
  EventListener,
][] = [];
const dispatchConversationChange = (convId: string) => {
  event.dispatchEvent(
    new CustomEvent('conversationChange', { detail: { convId } })
  );
};

// convId is a string prefixed with 'conv-'
const StorageUtils = {
  /**
   * manage conversations
   */
  getAllConversations(): Conversation[] {
    const res = [];
    for (const key in localStorage) {
      if (key.startsWith('conv-')) {
        res.push(JSON.parse(localStorage.getItem(key) ?? '{}'));
      }
    }
    res.sort((a, b) => b.lastModified - a.lastModified);
    return res;
  },
  /**
   * can return null if convId does not exist
   */
  getOneConversation(convId: string): Conversation | null {
    return JSON.parse(localStorage.getItem(convId) || 'null');
  },
  /**
   * if convId does not exist, create one
   */
  appendMsg(convId: string, msg: Message): void {
    if (msg.content === null) return;
    const conv = StorageUtils.getOneConversation(convId) || {
      id: convId,
      lastModified: Date.now(),
      messages: [],
    };
    conv.messages.push(msg);
    conv.lastModified = Date.now();
    localStorage.setItem(convId, JSON.stringify(conv));
    dispatchConversationChange(convId);
  },
  /**
   * Get new conversation id
   */
  getNewConvId(): string {
    return `conv-${Date.now()}`;
  },
  /**
   * remove conversation by id
   */
  remove(convId: string): void {
    localStorage.removeItem(convId);
    dispatchConversationChange(convId);
  },
  /**
   * remove all conversations
   */
  filterAndKeepMsgs(
    convId: string,
    predicate: (msg: Message) => boolean
  ): void {
    const conv = StorageUtils.getOneConversation(convId);
    if (!conv) return;
    conv.messages = conv.messages.filter(predicate);
    conv.lastModified = Date.now();
    localStorage.setItem(convId, JSON.stringify(conv));
    dispatchConversationChange(convId);
  },
  /**
   * remove last message from conversation
   */
  popMsg(convId: string): Message | undefined {
    const conv = StorageUtils.getOneConversation(convId);
    if (!conv) return;
    const msg = conv.messages.pop();
    conv.lastModified = Date.now();
    if (conv.messages.length === 0) {
      StorageUtils.remove(convId);
    } else {
      localStorage.setItem(convId, JSON.stringify(conv));
    }
    dispatchConversationChange(convId);
    return msg;
  },

  // event listeners
  onConversationChanged(callback: CallbackConversationChanged) {
    const fn = (e: Event) => callback((e as CustomEvent).detail.convId);
    onConversationChangedHandlers.push([callback, fn]);
    event.addEventListener('conversationChange', fn);
  },
  offConversationChanged(callback: CallbackConversationChanged) {
    const fn = onConversationChangedHandlers.find(([cb, _]) => cb === callback);
    if (fn) {
      event.removeEventListener('conversationChange', fn[1]);
    }
    onConversationChangedHandlers = [];
  },

  // manage config
  getConfig(): typeof CONFIG_DEFAULT {
    const savedVal = JSON.parse(localStorage.getItem('config') || '{}');
    // to prevent breaking changes in the future, we always provide default value for missing keys
    return {
      ...CONFIG_DEFAULT,
      ...savedVal,
    };
  },
  setConfig(config: typeof CONFIG_DEFAULT) {
    localStorage.setItem('config', JSON.stringify(config));
  },
  getTheme(): string {
    return localStorage.getItem('theme') || 'auto';
  },
  setTheme(theme: string) {
    if (theme === 'auto') {
      localStorage.removeItem('theme');
    } else {
      localStorage.setItem('theme', theme);
    }
  },
};

export default StorageUtils;
