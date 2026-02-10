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
    if (!speechSupported) {
      toast.error('Voice input not supported on this browser. Use typing.');
      return;
    }
    if (listening) {
      recognitionRef.current?.stop();
      setListening(false);
    } else {
      startListening();
    }
  };

  const startListening = () => {
    window.speechSynthesis?.cancel();
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SR();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      setInput(prev => prev ? prev + ' ' + transcript : transcript);
    };
    recognition.onerror = () => setListening(false);
    recognition.onend = () => setListening(false);

    recognitionRef.current = recognition;
    recognition.start();
    setListening(true);
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
          disabled={!speechSupported}
          title={speechSupported ? (listening ? 'Stop listening' : 'Voice input') : 'Voice not supported'}
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
