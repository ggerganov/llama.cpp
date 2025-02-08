import { useEffect, useState } from 'react';
import { useAppContext } from '../utils/app.context';
import StorageUtils from '../utils/storage';
import { useNavigate } from 'react-router';
import ChatMessage from './ChatMessage';
import { CanvasType, PendingMessage } from '../utils/types';
import { classNames } from '../utils/misc';
import CanvasPyInterpreter from './CanvasPyInterpreter';

export default function ChatScreen() {
  const {
    viewingConversation,
    sendMessage,
    isGenerating,
    stopGenerating,
    pendingMessages,
    canvasData,
  } = useAppContext();
  const [inputMsg, setInputMsg] = useState('');
  const navigate = useNavigate();

  const currConvId = viewingConversation?.id ?? '';
  const pendingMsg: PendingMessage | undefined = pendingMessages[currConvId];

  const scrollToBottom = (requiresNearBottom: boolean) => {
    const mainScrollElem = document.getElementById('main-scroll');
    if (!mainScrollElem) return;
    const spaceToBottom =
      mainScrollElem.scrollHeight -
      mainScrollElem.scrollTop -
      mainScrollElem.clientHeight;
    if (!requiresNearBottom || spaceToBottom < 50) {
      setTimeout(
        () => mainScrollElem.scrollTo({ top: mainScrollElem.scrollHeight }),
        1
      );
    }
  };

  // scroll to bottom when conversation changes
  useEffect(() => {
    scrollToBottom(false);
  }, [viewingConversation?.id]);

  const sendNewMessage = async () => {
    if (inputMsg.trim().length === 0 || isGenerating(currConvId)) return;
    const convId = viewingConversation?.id ?? StorageUtils.getNewConvId();
    const lastInpMsg = inputMsg;
    setInputMsg('');
    if (!viewingConversation) {
      // if user is creating a new conversation, redirect to the new conversation
      navigate(`/chat/${convId}`);
    }
    scrollToBottom(false);
    // auto scroll as message is being generated
    const onChunk = () => scrollToBottom(true);
    if (!(await sendMessage(convId, inputMsg, onChunk))) {
      // restore the input message if failed
      setInputMsg(lastInpMsg);
    }
  };

  const hasCanvas = !!canvasData;

  return (
    <div
      className={classNames({
        'grid lg:gap-8 grow transition-[300ms]': true,
        'grid-cols-[1fr_0fr] lg:grid-cols-[1fr_1fr]': hasCanvas, // adapted for mobile
        'grid-cols-[1fr_0fr]': !hasCanvas,
      })}
    >
      <div
        className={classNames({
          'flex flex-col w-full max-w-[900px] mx-auto': true,
          'hidden lg:flex': hasCanvas, // adapted for mobile
          flex: !hasCanvas,
        })}
      >
        {/* chat messages */}
        <div id="messages-list" className="grow">
          <div className="mt-auto flex justify-center">
            {/* placeholder to shift the message to the bottom */}
            {viewingConversation ? '' : 'Send a message to start'}
          </div>
          {viewingConversation?.messages.map((msg) => (
            <ChatMessage
              key={msg.id}
              msg={msg}
              scrollToBottom={scrollToBottom}
            />
          ))}

          {pendingMsg && (
            <ChatMessage
              msg={pendingMsg}
              scrollToBottom={scrollToBottom}
              isPending
              id="pending-msg"
            />
          )}
        </div>

        {/* chat input */}
        <div className="flex flex-row items-center pt-8 pb-6 sticky bottom-0 bg-base-100">
          <textarea
            className="textarea textarea-bordered w-full"
            placeholder="Type a message (Shift+Enter to add a new line)"
            value={inputMsg}
            onChange={(e) => setInputMsg(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && e.shiftKey) return;
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendNewMessage();
              }
            }}
            id="msg-input"
            dir="auto"
          ></textarea>
          {isGenerating(currConvId) ? (
            <button
              className="btn btn-neutral ml-2"
              onClick={() => stopGenerating(currConvId)}
            >
              Stop
            </button>
          ) : (
            <button
              className="btn btn-primary ml-2"
              onClick={sendNewMessage}
              disabled={inputMsg.trim().length === 0}
            >
              Send
            </button>
          )}
        </div>
      </div>
      <div className="w-full sticky top-[7em] h-[calc(100vh-9em)]">
        {canvasData?.type === CanvasType.PY_INTERPRETER && (
          <CanvasPyInterpreter />
        )}
      </div>
    </div>
  );
}
