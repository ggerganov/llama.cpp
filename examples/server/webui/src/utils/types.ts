export interface TimingReport {
  prompt_n: number;
  prompt_ms: number;
  predicted_n: number;
  predicted_ms: number;
}

/**
 * What is conversation "branching"? It is a feature that allows the user to edit an old message in the history, while still keeping the conversation flow.
 * Inspired by ChatGPT UI where you edit a message, a new branch of the conversation is created, and the old message is still visible.
 *
 * We use the same node based structure as ChatGPT, where each message has a parent and children. A "root" message is the first message in a conversation, which will not be displayed in the UI.
 */

export interface Message {
  id: number;
  convId: string;
  type: 'text' | 'root';
  timestamp: number; // timestamp from Date.now()
  role: 'user' | 'assistant' | 'system';
  content: string;
  timings?: TimingReport;
  // node based system for branching
  parent: Message['id'];
  children: Message['id'][];
}

export type APIMessage = Pick<Message, 'role' | 'content'>;

export interface Conversation {
  id: string; // format: `conv-{timestamp}`
  lastModified: number; // timestamp from Date.now()
  currNode: Message['id']; // the current message node being viewed
  name: string;
}

export interface ViewingChat {
  conv: Readonly<Conversation>;
  messages: Readonly<Message[]>;
}

export type PendingMessage = Omit<Message, 'content'> & {
  content: string | null;
};

export enum CanvasType {
  PY_INTERPRETER,
}

export interface CanvasPyInterpreter {
  type: CanvasType.PY_INTERPRETER;
  content: string;
}

export type CanvasData = CanvasPyInterpreter;
