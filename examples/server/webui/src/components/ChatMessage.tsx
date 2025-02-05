import { useMemo, useState } from 'react';
import { useAppContext } from '../utils/app.context';
import { Message, PendingMessage } from '../utils/types';
import { classNames } from '../utils/misc';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHightlight from 'rehype-highlight';

export default function ChatMessage({
  msg,
  id,
  scrollToBottom,
}: {
  msg: Message | PendingMessage;
  id?: string;
  scrollToBottom: (requiresNearBottom: boolean) => void;
}) {
  const { viewingConversation, replaceMessageAndGenerate, config } =
    useAppContext();
  const [editingContent, setEditingContent] = useState<string | null>(null);
  const timings = useMemo(
    () =>
      msg.timings
        ? {
            ...msg.timings,
            prompt_per_second:
              (msg.timings.prompt_n / msg.timings.prompt_ms) * 1000,
            predicted_per_second:
              (msg.timings.predicted_n / msg.timings.predicted_ms) * 1000,
          }
        : null,
    [msg.timings]
  );

  if (!viewingConversation) return null;

  const regenerate = async () => {
    replaceMessageAndGenerate(viewingConversation.id, msg.id, undefined, () =>
      scrollToBottom(true)
    );
  };

  return (
    <div className="group" id={id}>
      <div
        className={classNames({
          chat: true,
          'chat-start': msg.role !== 'user',
          'chat-end': msg.role === 'user',
        })}
      >
        <div
          className={classNames({
            'chat-bubble markdown': true,
            'chat-bubble-base-300': msg.role !== 'user',
          })}
        >
          {/* textarea for editing message */}
          {editingContent !== null && (
            <>
              <textarea
                dir="auto"
                className="textarea textarea-bordered bg-base-100 text-base-content w-[calc(90vw-8em)] lg:w-96"
                value={editingContent}
                onChange={(e) => setEditingContent(e.target.value)}
              ></textarea>
              <br />
              <button
                className="btn btn-ghost mt-2 mr-2"
                onClick={() => setEditingContent(null)}
              >
                Cancel
              </button>
              <button
                className="btn mt-2"
                onClick={() =>
                  replaceMessageAndGenerate(
                    viewingConversation.id,
                    msg.id,
                    editingContent
                  )
                }
              >
                Submit
              </button>
            </>
          )}
          {editingContent === null && (
            <>
              {msg.content === null ? (
                <>
                  {/* show loading dots for pending message */}
                  <span className="loading loading-dots loading-md"></span>
                </>
              ) : (
                <>
                  {/* render message as markdown */}
                  <div dir="auto">
                    <MarkdownDisplay content={msg.content} />
                  </div>
                </>
              )}
              {/* render timings if enabled */}
              {timings && config.showTokensPerSecond && (
                <div className="dropdown dropdown-hover dropdown-top mt-2">
                  <div
                    tabIndex={0}
                    role="button"
                    className="cursor-pointer font-semibold text-sm opacity-60"
                  >
                    Speed: {timings.predicted_per_second.toFixed(1)} t/s
                  </div>
                  <div className="dropdown-content bg-base-100 z-10 w-64 p-2 shadow mt-4">
                    <b>Prompt</b>
                    <br />- Tokens: {timings.prompt_n}
                    <br />- Time: {timings.prompt_ms} ms
                    <br />- Speed: {timings.prompt_per_second.toFixed(1)} t/s
                    <br />
                    <b>Generation</b>
                    <br />- Tokens: {timings.predicted_n}
                    <br />- Time: {timings.predicted_ms} ms
                    <br />- Speed: {timings.predicted_per_second.toFixed(1)} t/s
                    <br />
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* actions for each message */}
      {msg.content !== null && (
        <div
          className={classNames({
            'mx-4 mt-2 mb-2': true,
            'text-right': msg.role === 'user',
          })}
        >
          {/* user message */}
          {msg.role === 'user' && (
            <button
              className="badge btn-mini show-on-hover"
              onClick={() => setEditingContent(msg.content)}
              disabled={msg.content === null}
            >
              ✍️ Edit
            </button>
          )}
          {/* assistant message */}
          {msg.role === 'assistant' && (
            <>
              <button
                className="badge btn-mini show-on-hover mr-2"
                onClick={regenerate}
                disabled={msg.content === null}
              >
                🔄 Regenerate
              </button>
              <button
                className="badge btn-mini show-on-hover mr-2"
                onClick={() => navigator.clipboard.writeText(msg.content || '')}
                disabled={msg.content === null}
              >
                📋 Copy
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function MarkdownDisplay({ content }: { content: string }) {
  return (
    <Markdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHightlight]}>
      {content}
    </Markdown>
  );
}
