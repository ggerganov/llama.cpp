import { useEffect, useRef, useState } from 'react';
import { useAppContext } from '../utils/app.context';
import StorageUtils from '../utils/storage';
import { useNavigate } from 'react-router';
import ChatMessage from './ChatMessage';
import { PendingMessage } from '../utils/types';

export default function ChatScreen() {
  const {
    viewingConversation,
    sendMessage,
    isGenerating,
    stopGenerating,
    pendingMessages,
  } = useAppContext();
  const [inputMsg, setInputMsg] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  const currConvId = viewingConversation?.id ?? '';
  const pendingMsg: PendingMessage | undefined = pendingMessages[currConvId];

  const scrollToBottom = (requiresNearBottom: boolean) => {
    if (!containerRef.current) return;
    const msgListElem = containerRef.current;
    const spaceToBottom =
      msgListElem.scrollHeight -
      msgListElem.scrollTop -
      msgListElem.clientHeight;
    if (!requiresNearBottom || spaceToBottom < 50) {
      setTimeout(
        () => msgListElem.scrollTo({ top: msgListElem.scrollHeight }),
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

  return (
    <>
      {/* chat messages */}
      <div
        id="messages-list"
        className="flex flex-col grow overflow-y-auto"
        ref={containerRef}
      >
        <div className="mt-auto flex justify-center">
          {/* placeholder to shift the message to the bottom */}
          {viewingConversation ? '' : 'Send a message to start'}
        </div>
        {viewingConversation?.messages.map((msg) => (
          <ChatMessage key={msg.id} msg={msg} scrollToBottom={scrollToBottom} />
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
      <div className="flex flex-row items-center mt-8 mb-6">
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
    </>
  );
}
