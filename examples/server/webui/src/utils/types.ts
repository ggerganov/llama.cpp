export interface TimingReport {
  prompt_n: number;
  prompt_ms: number;
  predicted_n: number;
  predicted_ms: number;
}

export interface Message {
  id: number;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timings?: TimingReport;
}

export type APIMessage = Pick<Message, 'role' | 'content'>;

export interface Conversation {
  id: string; // format: `conv-{timestamp}`
  lastModified: number; // timestamp from Date.now()
  messages: Message[];
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
