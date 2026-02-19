import { useState, useEffect, useRef } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { Send, Mic, MicOff, Volume2, Loader2 } from 'lucide-react';
import { toast } from 'sonner';
import axios from 'axios';

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

export default function ChatPage() {
  const { session } = useAuth();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [pendingReminder, setPendingReminder] = useState(null);
  const [listening, setListening] = useState(false);
  const [speechSupported, setSpeechSupported] = useState(false);
  const messagesEndRef = useRef(null);
  const recognitionRef = useRef(null);

  const headers = { Authorization: `Bearer ${session?.access_token}` };

  useEffect(() => {
    loadMessages();
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    setSpeechSupported(!!SR);
    // eslint-disable-next-line
  }, []);

  const loadMessages = async () => {
    try {
      const { data } = await axios.get(`${API}/messages`, { headers });
      setMessages(data || []);
      scrollToBottom();
    } catch {
      // First-time user, no messages yet
    }
  };

  const scrollToBottom = () => {
    setTimeout(() => messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }), 100);
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    const msg = input.trim();
    if (!msg || sending) return;

    setInput('');
    setSending(true);

    setMessages(prev => [...prev, { role: 'user', content: msg, created_at: new Date().toISOString() }]);
    scrollToBottom();

    try {
      const { data } = await axios.post(`${API}/chat`, {
        message: msg,
        pending_reminder: pendingReminder,
      }, { headers });

      const assistantMsg = {
        role: 'assistant',
        content: data.response,
        created_at: new Date().toISOString(),
        _type: data.type,
        _kb: data.kb_entry,
        _reminder: data.reminder,
      };
      setMessages(prev => [...prev, assistantMsg]);

      if (data.type === 'clarify' && data.updated_pending) {
        setPendingReminder(data.updated_pending);
      } else {
        setPendingReminder(null);
      }

      if (data.type === 'kb') toast.success('Saved to Knowledge Base');
      if (data.type === 'reminder') toast.success('Reminder created');
      if (data.type === 'both') {
        toast.success('Saved to KB + Reminder created');
      }

      const autoSpeak = localStorage.getItem('autoSpeak') === 'true';
      if (autoSpeak && data.response && data.response.length < 200) {
        speak(data.response);
      }

      scrollToBottom();
    } catch (err) {
      toast.error('Failed to process message');
      console.error(err);
    } finally {
      setSending(false);
    }
  };

  const toggleListening = () => {
    // Must detect support synchronously in the click handler
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      toast.error('Voice input not supported in this browser. Use typing.');
      setSpeechSupported(false);
      return;
    }

    // If already listening, stop
    if (recognitionRef.current && listening) {
      console.log('[SR] stopping');
      recognitionRef.current.stop();
      setListening(false);
      return;
    }

    // Prevent double-start
    if (recognitionRef.current) {
      try { recognitionRef.current.abort(); } catch { /* ok */ }
      recognitionRef.current = null;
    }

    // Barge-in: cancel any ongoing TTS
    window.speechSynthesis?.cancel();

    // Create and start recognition synchronously inside user gesture
    const recognition = new SR();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      console.log('[SR] start');
      setListening(true);
    };

    recognition.onresult = (event) => {
      let finalTranscript = '';
      let interimTranscript = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const t = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += t;
        } else {
          interimTranscript += t;
        }
      }
      console.log('[SR] result:', finalTranscript || interimTranscript);
      if (finalTranscript) {
        setInput(prev => prev ? prev + ' ' + finalTranscript : finalTranscript);
      }
    };

    recognition.onerror = (event) => {
      console.error('[SR] error:', event.error);
      setListening(false);
      recognitionRef.current = null;
      if (event.error === 'not-allowed' || event.error === 'permission-denied') {
        toast.error('Microphone blocked. Enable mic permissions in browser site settings and refresh.');
      } else if (event.error === 'no-speech') {
        toast.info('No speech detected. Try again.');
      } else if (event.error !== 'aborted') {
        toast.error(`Voice error: ${event.error}`);
      }
    };

    recognition.onend = () => {
      console.log('[SR] end');
      setListening(false);
      recognitionRef.current = null;
    };

    recognitionRef.current = recognition;
    try {
      recognition.start();
    } catch (err) {
      console.error('[SR] start failed:', err);
      setListening(false);
      recognitionRef.current = null;
      toast.error('Failed to start voice input. Try again.');
    }
  };

  const speak = (text) => {
    if (!window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1;
    utterance.onerror = () => {};
    try {
      window.speechSynthesis.speak(utterance);
    } catch {
      // iOS may block auto-play
    }
  };

  return (
    <div className="chat-page" data-testid="chat-page">
      <div className="chat-messages" data-testid="chat-messages">
        {messages.length === 0 && !sending && (
          <div className="empty-state" style={{ paddingTop: '25vh' }}>
            <p className="text-lg font-semibold" style={{ fontFamily: 'Manrope, sans-serif' }}>
              Welcome to SecondBrain
            </p>
            <p className="text-sm text-muted-foreground max-w-[260px]">
              Tell me anything. I'll save facts to your Knowledge Base and create reminders automatically.
            </p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`msg ${msg.role}`} data-testid={`msg-${i}`}>
            <div className={`msg-bubble ${msg.role}`}>
              {msg.content}
              {msg._type === 'kb' && <div><span className="badge-kb">Saved to KB</span></div>}
              {msg._type === 'reminder' && <div><span className="badge-reminder">Reminder Set</span></div>}
              {msg._type === 'both' && <div><span className="badge-kb">Saved to KB</span>{' '}<span className="badge-reminder">Reminder Set</span></div>}
            </div>
            {msg.role === 'assistant' && (
              <button
                className="speak-btn"
                onClick={() => speak(msg.content)}
                title="Speak this message"
                data-testid={`speak-${i}`}
              >
                <Volume2 size={14} />
              </button>
            )}
          </div>
        ))}

        {sending && (
          <div className="msg assistant">
            <div className="msg-bubble assistant">
              <div className="typing-indicator">
                <span /><span /><span />
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form className="chat-input-form" onSubmit={sendMessage} data-testid="chat-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={listening ? 'Listening...' : 'Type a message...'}
          disabled={sending}
          data-testid="chat-input"
          autoComplete="off"
        />
        <button
          type="button"
          onClick={toggleListening}
          className={listening ? 'mic-active' : ''}
          title={listening ? 'Stop listening' : 'Voice input'}
          data-testid="mic-button"
        >
          {listening ? <MicOff size={20} /> : <Mic size={20} />}
        </button>
        <button
          type="submit"
          disabled={!input.trim() || sending}
          data-testid="send-button"
        >
          {sending ? <Loader2 size={20} className="animate-spin" /> : <Send size={20} />}
        </button>
      </form>
    </div>
  );
}
