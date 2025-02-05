import React, { useMemo, useState } from 'react';
import { useAppContext } from '../utils/app.context';
import { Message, PendingMessage } from '../utils/types';
import { classNames, copyStr } from '../utils/misc';
import Markdown, { ExtraProps } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHightlight from 'rehype-highlight';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';
import remarkBreaks from 'remark-breaks';
import 'katex/dist/katex.min.css';

interface SplitMessage {
  content: PendingMessage['content'];
  thought?: string;
  isThinking?: boolean;
}

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

  const isPending: boolean = !!(msg as PendingMessage).convId;

  // for reasoning model, we split the message into content and thought
  // TODO: implement this as remark/rehype plugin in the future
  const { content, thought, isThinking }: SplitMessage = useMemo(() => {
    if (msg.content === null || msg.role !== 'assistant') {
      return { content: msg.content };
    }
    let actualContent = '';
    let thought = '';
    let isThinking = false;
    let thinkSplit = msg.content.split('<think>', 2);
    actualContent += thinkSplit[0];
    while (thinkSplit[1] !== undefined) {
      // <think> tag found
      thinkSplit = thinkSplit[1].split('</think>', 2);
      thought += thinkSplit[0];
      isThinking = true;
      if (thinkSplit[1] !== undefined) {
        // </think> closing tag found
        isThinking = false;
        thinkSplit = thinkSplit[1].split('<think>', 2);
        actualContent += thinkSplit[0];
      }
    }
    return { content: actualContent, thought, isThinking };
  }, [msg]);

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
          {/* not editing content, render message */}
          {editingContent === null && (
            <>
              {content === null ? (
                <>
                  {/* show loading dots for pending message */}
                  <span className="loading loading-dots loading-md"></span>
                </>
              ) : (
                <>
                  {/* render message as markdown */}
                  <div dir="auto">
                    {thought && (
                      <details
                        className="collapse bg-base-200 collapse-arrow mb-4"
                        open={isThinking && config.showThoughtInProgress}
                      >
                        <summary className="collapse-title">
                          {isPending && isThinking ? (
                            <span>
                              <span
                                v-if="isGenerating"
                                className="loading loading-spinner loading-md mr-2"
                                style={{ verticalAlign: 'middle' }}
                              ></span>
                              <b>Thinking</b>
                            </span>
                          ) : (
                            <b>Thought Process</b>
                          )}
                        </summary>
                        <div className="collapse-content">
                          <MarkdownDisplay content={thought} />
                        </div>
                      </details>
                    )}
                    <MarkdownDisplay content={content} />
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

const Pre: React.ElementType<
  React.ClassAttributes<HTMLPreElement> &
    React.HTMLAttributes<HTMLPreElement> &
    ExtraProps & { origContent: string }
> = ({ node, origContent, ...props }) => {
  const startOffset = node?.position?.start.offset ?? 0;
  const endOffset = node?.position?.end.offset ?? 0;

  const [copied, setCopied] = useState(false);
  const copiedContent = useMemo(
    () =>
      origContent
        .substring(startOffset, endOffset)
        .replace(/^```[^\n]+\n/g, '')
        .replace(/```$/g, ''),
    [origContent, startOffset, endOffset]
  );

  if (!node?.position) {
    return <pre {...props} />;
  }

  return (
    <div className="relative my-4">
      <div
        className="text-right sticky top-4 mb-2 mr-2 h-0"
        onClick={() => {
          copyStr(copiedContent);
          setCopied(true);
        }}
        onMouseLeave={() => setCopied(false)}
      >
        <button className="badge btn-mini">
          {copied ? 'Copied!' : '📋 Copy'}
        </button>
      </div>
      <pre {...props} />
    </div>
  );
};

function MarkdownDisplay({ content }: { content: string }) {
  const preprocessedContent = useMemo(
    () => preprocessLaTeX(content),
    [content]
  );
  return (
    <Markdown
      remarkPlugins={[remarkGfm, remarkMath, remarkBreaks]}
      rehypePlugins={[rehypeHightlight, rehypeKatex]}
      components={{
        pre: (props) => <Pre {...props} origContent={preprocessedContent} />,
      }}
    >
      {preprocessedContent}
    </Markdown>
  );
}

/**
 * Preprocesses LaTeX content by replacing delimiters and escaping certain characters.
 * Ref: https://github.com/remarkjs/react-markdown/issues/785
 */
export function preprocessLaTeX(content: string): string {
  // Step 1: Protect code blocks
  const codeBlocks: string[] = [];
  content = content.replace(/(```[\s\S]*?```|`[^`\n]+`)/g, (_, code) => {
    codeBlocks.push(code);
    return `<<CODE_BLOCK_${codeBlocks.length - 1}>>`;
  });

  // Step 2: Protect existing LaTeX expressions
  const latexExpressions: string[] = [];
  content = content.replace(
    /(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\\(.*?\\\))/g,
    (match) => {
      latexExpressions.push(match);
      return `<<LATEX_${latexExpressions.length - 1}>>`;
    }
  );

  // Step 3: Escape dollar signs that are likely currency indicators
  content = content.replace(/\$(?=\d)/g, '\\$');

  // Step 4: Restore LaTeX expressions
  content = content.replace(
    /<<LATEX_(\d+)>>/g,
    (_, index) => latexExpressions[parseInt(index)]
  );

  // Step 5: Restore code blocks
  content = content.replace(
    /<<CODE_BLOCK_(\d+)>>/g,
    (_, index) => codeBlocks[parseInt(index)]
  );

  // Step 6: Apply additional escaping functions
  content = escapeBrackets(content);
  content = escapeMhchem(content);

  return content;
}

function escapeBrackets(text: string) {
  const pattern =
    /(```[\S\s]*?```|`.*?`)|\\\[([\S\s]*?[^\\])\\]|\\\((.*?)\\\)/g;
  return text.replaceAll(
    pattern,
    (match, codeBlock, squareBracket, roundBracket) => {
      if (codeBlock) {
        return codeBlock;
      } else if (squareBracket) {
        return `$$${squareBracket}$$`;
      } else if (roundBracket) {
        return `$${roundBracket}$`;
      }
      return match;
    }
  );
}

function escapeMhchem(text: string) {
  return text.replaceAll('$\\ce{', '$\\\\ce{').replaceAll('$\\pu{', '$\\\\pu{');
}
