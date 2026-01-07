
import React, { useState, useRef, useCallback } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality } from '@google/genai';
import { SYSTEM_PROMPT, ZIPPY_COLORS } from './constants';
import { SessionState, Message } from './types';
import { encode, decode, decodeAudioData, float32ToInt16, getRMS } from './utils/audio';
import Visualizer from './components/Visualizer';
import ConversationLog from './components/ConversationLog';

const App: React.FC = () => {
  const [sessionState, setSessionState] = useState<SessionState>(SessionState.DISCONNECTED);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [latency, setLatency] = useState<number | null>(null);

  // Live transcriptions for the UI HUD
  const [userPartialTranscript, setUserPartialTranscript] = useState('');
  const [aiPartialTranscript, setAiPartialTranscript] = useState('');

  // Hardware & Session Refs
  const audioContextIn = useRef<AudioContext | null>(null);
  const audioContextOut = useRef<AudioContext | null>(null);
  const analyserIn = useRef<AnalyserNode | null>(null);
  const analyserOut = useRef<AnalyserNode | null>(null);
  const sessionRef = useRef<any>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const activeSources = useRef<Set<AudioBufferSourceNode>>(new Set());
  const nextStartTime = useRef<number>(0);
  
  // Data State Refs
  const currentInputTranscription = useRef('');
  const currentOutputTranscription = useRef('');
  const isSessionActive = useRef(false);
  
  // Performance Metrics
  const lastUserAudioTime = useRef<number>(0);
  const firstAiAudioReceived = useRef(false);

  const cleanupSession = useCallback(() => {
    isSessionActive.current = false;
    
    // Stop and clear any playing audio
    activeSources.current.forEach(s => {
      try { s.stop(); } catch(e) {}
    });
    activeSources.current.clear();

    // Close session gracefully
    if (sessionRef.current) {
      try { sessionRef.current.close(); } catch (e) {}
      sessionRef.current = null;
    }
    
    // Kill the microphone processor
    if (scriptProcessorRef.current) {
      scriptProcessorRef.current.disconnect();
      scriptProcessorRef.current = null;
    }

    // Close audio hardware
    if (audioContextIn.current) {
      try { audioContextIn.current.close(); } catch (e) {}
      audioContextIn.current = null;
    }
    if (audioContextOut.current) {
      try { audioContextOut.current.close(); } catch (e) {}
      audioContextOut.current = null;
    }
    
    setSessionState(SessionState.DISCONNECTED);
    setIsTyping(false);
    setUserPartialTranscript('');
    setAiPartialTranscript('');
    nextStartTime.current = 0;
  }, []);

  const startConversation = async () => {
    try {
      setErrorMessage(null);
      setSessionState(SessionState.CONNECTING);

      // 1. Initialize Audio Hardware
      // Use 16kHz for input as required by PCM rate and 24kHz for output
      const ctxIn = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      const ctxOut = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      
      await ctxIn.resume();
      await ctxOut.resume();

      audioContextIn.current = ctxIn;
      audioContextOut.current = ctxOut;
      analyserIn.current = ctxIn.createAnalyser();
      analyserOut.current = ctxOut.createAnalyser();
      
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const micSource = ctxIn.createMediaStreamSource(stream);
      micSource.connect(analyserIn.current);

      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      
      // 2. Establish Connection
      const sessionPromise = ai.live.connect({
        model: 'gemini-2.5-flash-native-audio-preview-12-2025',
        callbacks: {
          onopen: () => {
            console.log("Zippy Live Connected");
            setSessionState(SessionState.CONNECTED);
            isSessionActive.current = true;
            
            // 8192 buffer provides more stability for network jitter
            const scriptProcessor = ctxIn.createScriptProcessor(8192, 1, 1);
            scriptProcessorRef.current = scriptProcessor;

            scriptProcessor.onaudioprocess = (e) => {
              if (!isSessionActive.current || !sessionRef.current) return;
              
              const inputData = e.inputBuffer.getChannelData(0);
              
              // NOISE GATE: Only send if there is actual sound (0.01 threshold)
              // This prevents "Internal Error" from excessive silence processing
              const volume = getRMS(inputData);
              if (volume < 0.005) return; 

              const pcmData = float32ToInt16(inputData);
              const pcmBlob = {
                data: encode(new Uint8Array(pcmData.buffer)),
                mimeType: 'audio/pcm;rate=16000',
              };
              
              try {
                sessionRef.current.sendRealtimeInput({ media: pcmBlob });
              } catch (err) {
                console.warn("Failed to stream audio chunk, session may be unstable.");
              }
            };
            
            micSource.connect(scriptProcessor);
            scriptProcessor.connect(ctxIn.destination);
          },
          onmessage: async (message: LiveServerMessage) => {
            if (!isSessionActive.current) return;

            // Handle Incoming Audio
            const audioChunk = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
            if (audioChunk && audioContextOut.current) {
              // Track First-Byte-Latency
              if (!firstAiAudioReceived.current && lastUserAudioTime.current > 0) {
                const now = performance.now();
                setLatency(parseFloat(((now - lastUserAudioTime.current) / 1000).toFixed(2)));
                firstAiAudioReceived.current = true;
              }

              setIsTyping(true);
              const ctx = audioContextOut.current;
              nextStartTime.current = Math.max(nextStartTime.current, ctx.currentTime);
              
              const audioBuffer = await decodeAudioData(decode(audioChunk), ctx, 24000, 1);
              const sourceNode = ctx.createBufferSource();
              sourceNode.buffer = audioBuffer;
              sourceNode.connect(analyserOut.current!);
              analyserOut.current!.connect(ctx.destination);
              
              sourceNode.onended = () => {
                activeSources.current.delete(sourceNode);
              };
              
              sourceNode.start(nextStartTime.current);
              nextStartTime.current += audioBuffer.duration;
              activeSources.current.add(sourceNode);
            }

            // Handle Input/Output Transcriptions
            if (message.serverContent?.inputTranscription) {
              const text = message.serverContent.inputTranscription.text;
              currentInputTranscription.current += text;
              setUserPartialTranscript(currentInputTranscription.current);
              lastUserAudioTime.current = performance.now();
              firstAiAudioReceived.current = false;
            }
            if (message.serverContent?.outputTranscription) {
              const text = message.serverContent.outputTranscription.text;
              currentOutputTranscription.current += text;
              setAiPartialTranscript(currentOutputTranscription.current);
            }

            // End of a back-and-forth turn
            if (message.serverContent?.turnComplete) {
              const uText = currentInputTranscription.current;
              const aText = currentOutputTranscription.current;

              if (uText || aText) {
                setMessages(prev => [
                  ...prev,
                  ...(uText ? [{ id: `u-${Date.now()}`, sender: 'user', text: uText, timestamp: Date.now() } as Message] : []),
                  ...(aText ? [{ id: `a-${Date.now()}`, sender: 'assistant', text: aText, timestamp: Date.now() } as Message] : [])
                ]);
              }

              // Clear turn-based state
              currentInputTranscription.current = '';
              currentOutputTranscription.current = '';
              setUserPartialTranscript('');
              setAiPartialTranscript('');
              setIsTyping(false);
            }

            // Handle Barge-in / Interruptions
            if (message.serverContent?.interrupted) {
              activeSources.current.forEach(s => { try { s.stop(); } catch {} });
              activeSources.current.clear();
              nextStartTime.current = 0;
              setIsTyping(false);
              currentOutputTranscription.current = '';
              setAiPartialTranscript('');
              firstAiAudioReceived.current = false;
            }
          },
          onerror: (e) => {
            console.error('Session Critical Error:', e);
            const msg = (e as any)?.message || '';
            if (msg.includes('Internal error') || msg.includes('Network')) {
              setErrorMessage('Connection error. Zippy is reconnecting...');
            } else {
              setErrorMessage('Communication error. Please try starting the chat again.');
            }
            cleanupSession();
          },
          onclose: () => {
            console.log('Session closed by server.');
            setSessionState(SessionState.DISCONNECTED);
            setIsTyping(false);
            isSessionActive.current = false;
          }
        },
        config: {
          responseModalities: [Modality.AUDIO],
          systemInstruction: SYSTEM_PROMPT.trim(),
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } }
          },
          inputAudioTranscription: {},
          outputAudioTranscription: {}
        }
      });

      // Secure the resolved session object for direct access
      sessionRef.current = await sessionPromise;

    } catch (err) {
      console.error('Failed to start Zippy:', err);
      setErrorMessage('Could not access microphone or network. Please check your settings.');
      setSessionState(SessionState.DISCONNECTED);
    }
  };

  return (
    <div className="min-h-screen flex flex-col p-4 md:p-8 max-w-2xl mx-auto space-y-6">
      {/* Brand Header */}
      <header className="text-center space-y-2 py-4">
        <div className="w-24 h-24 zippy-gradient rounded-full mx-auto flex items-center justify-center shadow-lg mb-4 ring-4 ring-orange-100 ring-offset-4 animate-in fade-in zoom-in duration-500">
          <svg className="w-14 h-14 text-white" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
          </svg>
        </div>
        <h1 className="text-3xl font-bold text-orange-600 tracking-tight">Zippy Assistant</h1>
        <p className="text-orange-400 font-medium">Your screen-free product companion</p>
      </header>

      {/* Main Conversation Log */}
      <main className="flex-1 flex flex-col space-y-4 min-h-0 relative">
        <ConversationLog messages={messages} isTyping={isTyping || !!aiPartialTranscript} />
        
        {/* Transcription HUD */}
        {(userPartialTranscript || aiPartialTranscript) && (
           <div className="absolute bottom-6 left-6 right-6 flex flex-col items-stretch space-y-3 animate-in fade-in slide-in-from-bottom-4 duration-300 pointer-events-none">
              {userPartialTranscript && (
                <div className="bg-orange-50/95 backdrop-blur-md border border-orange-200 p-4 rounded-2xl text-sm text-orange-900 shadow-xl border-l-4 border-l-orange-500">
                  <span className="font-bold opacity-40 block text-[10px] uppercase mb-1 tracking-widest">Listening</span>
                  "{userPartialTranscript}"
                </div>
              )}
              {aiPartialTranscript && (
                <div className="bg-blue-50/95 backdrop-blur-md border border-blue-200 p-4 rounded-2xl text-sm text-blue-900 shadow-xl border-l-4 border-l-blue-500">
                  <span className="font-bold opacity-40 block text-[10px] uppercase mb-1 tracking-widest">Zippy</span>
                  "{aiPartialTranscript}"
                </div>
              )}
           </div>
        )}
      </main>
      
      {/* Control Station */}
      <div className="bg-white p-6 rounded-[2.5rem] shadow-2xl border border-orange-50 space-y-5">
        <div className="flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <div className={`w-2 h-2 rounded-full transition-colors duration-500 ${sessionState === SessionState.CONNECTED ? 'bg-green-500 animate-pulse' : 'bg-gray-200'}`}></div>
            <span className="text-[10px] font-black text-gray-400 uppercase tracking-widest">{sessionState}</span>
            {latency !== null && sessionState === SessionState.CONNECTED && (
              <span className="text-[10px] bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full font-black animate-in zoom-in duration-300">
                ⚡ {latency}s RESPONSE
              </span>
            )}
          </div>
        </div>

        <div className="relative h-24 flex items-center justify-center">
          <Visualizer 
            analyser={isTyping ? analyserOut.current : analyserIn.current} 
            isActive={sessionState === SessionState.CONNECTED}
            color={isTyping ? ZIPPY_COLORS.secondary : ZIPPY_COLORS.primary}
          />
        </div>

        {errorMessage && (
          <div className="bg-red-50 text-red-600 p-3 rounded-xl text-xs text-center border border-red-100 font-semibold animate-bounce">
            {errorMessage}
          </div>
        )}

        <div className="pt-2">
          {sessionState === SessionState.DISCONNECTED ? (
            <button
              onClick={startConversation}
              className="w-full zippy-gradient text-white font-bold py-5 px-6 rounded-2xl shadow-xl hover:scale-[1.01] active:scale-[0.98] transition-all flex items-center justify-center space-x-4 text-xl"
            >
              <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>
              <span>Start Speaking</span>
            </button>
          ) : (
            <button
              onClick={cleanupSession}
              disabled={sessionState === SessionState.CONNECTING}
              className="w-full bg-red-400 text-white font-bold py-5 px-6 rounded-2xl shadow-xl hover:bg-red-500 active:scale-[0.98] transition-all flex items-center justify-center space-x-4 text-xl disabled:opacity-50"
            >
              <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M6 18L18 6M6 6l12 12" />
              </svg>
              <span>{sessionState === SessionState.CONNECTING ? 'Connecting...' : 'End Conversation'}</span>
            </button>
          )}
        </div>
      </div>

      <footer className="text-center text-orange-300 text-[10px] py-2 uppercase tracking-[0.3em] font-black opacity-60">
        Screen-Free • Montessori • Family Safe
      </footer>
    </div>
  );
};

export default App;
