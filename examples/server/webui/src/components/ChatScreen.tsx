import { useEffect, useRef, useState } from 'react';
import { useAppContext } from '../utils/app.context';
import StorageUtils from '../utils/storage';
import { useNavigate } from 'react-router';
import ChatMessage from './ChatMessage';

export default function ChatScreen() {
  const {
    viewingConversation,
    sendMessage,
    isGenerating,
    stopGenerating,
    pendingMessage,
  } = useAppContext();
  const [inputMsg, setInputMsg] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  const scrollToBottom = (requiresNearBottom: boolean) => {
    if (!containerRef.current) return;
    const msgListElem = containerRef.current;
    const spaceToBottom =
      msgListElem.scrollHeight -
      msgListElem.scrollTop -
      msgListElem.clientHeight;
    if (!requiresNearBottom || spaceToBottom < 100) {
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
    if (inputMsg.trim().length === 0) return;
    const convId = viewingConversation?.id ?? StorageUtils.getNewConvId();
    const lastInpMsg = inputMsg;
    setInputMsg('');
    if (!viewingConversation) {
      // if user is creating a new conversation, redirect to the new conversation
      navigate(`/chat/${convId}`);
    }
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

        {pendingMessage !== null &&
          pendingMessage.convId === viewingConversation?.id && (
            <ChatMessage
              msg={pendingMessage}
              scrollToBottom={scrollToBottom}
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
        {isGenerating ? (
          <button className="btn btn-neutral ml-2" onClick={stopGenerating}>
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
