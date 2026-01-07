
import React, { useEffect, useRef } from 'react';
import { Message } from '../types';

interface ConversationLogProps {
  messages: Message[];
  isTyping: boolean;
}

const ConversationLog: React.FC<ConversationLogProps> = ({ messages, isTyping }) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  return (
    <div 
      ref={scrollRef}
      className="flex-1 overflow-y-auto p-4 space-y-4 bg-white/50 rounded-2xl shadow-inner border border-orange-100"
      style={{ minHeight: '300px' }}
    >
      {messages.length === 0 && !isTyping && (
        <div className="flex items-center justify-center h-full text-orange-300 italic">
          Tap "Start Chatting" to speak with Zippy Assistant
        </div>
      )}
      
      {messages.map((msg) => (
        <div 
          key={msg.id}
          className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
        >
          <div 
            className={`max-w-[80%] px-4 py-2 rounded-2xl text-sm ${
              msg.sender === 'user' 
                ? 'bg-orange-400 text-white rounded-tr-none' 
                : 'bg-white text-gray-700 border border-orange-100 rounded-tl-none shadow-sm'
            }`}
          >
            {msg.text}
          </div>
        </div>
      ))}

      {isTyping && (
        <div className="flex justify-start">
          <div className="bg-white px-4 py-2 rounded-2xl text-sm border border-orange-100 rounded-tl-none shadow-sm flex items-center space-x-1">
            <div className="w-1.5 h-1.5 bg-orange-300 rounded-full animate-bounce"></div>
            <div className="w-1.5 h-1.5 bg-orange-300 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
            <div className="w-1.5 h-1.5 bg-orange-300 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ConversationLog;
