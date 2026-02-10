import { useState, useEffect, useMemo } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { Check, Clock, AlertTriangle, Calendar } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

export default function RemindersPage() {
  const { session } = useAuth();
  const [reminders, setReminders] = useState([]);
  const [loading, setLoading] = useState(true);

  const headers = { Authorization: `Bearer ${session?.access_token}` };

  useEffect(() => {
    loadReminders();
    // eslint-disable-next-line
  }, []);

  const loadReminders = async () => {
    setLoading(true);
    try {
      const { data } = await axios.get(`${API}/reminders`, { headers, params: { status: 'open' } });
      setReminders(data);
    } catch {
      toast.error('Failed to load reminders');
    } finally {
      setLoading(false);
    }
  };

  const markDone = async (id) => {
    try {
      await axios.patch(`${API}/reminders/${id}`, { status: 'done' }, { headers });
      setReminders(prev => prev.filter(r => r.id !== id));
      toast.success('Reminder completed');
    } catch {
      toast.error('Failed to update');
    }
  };

  const { overdue, today, upcoming } = useMemo(() => {
    const now = new Date();
    const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const todayEnd = new Date(todayStart);
    todayEnd.setDate(todayEnd.getDate() + 1);

    return {
      overdue: reminders.filter(r => new Date(r.due_datetime) < todayStart),
      today: reminders.filter(r => {
        const d = new Date(r.due_datetime);
        return d >= todayStart && d < todayEnd;
      }),
      upcoming: reminders.filter(r => new Date(r.due_datetime) >= todayEnd),
    };
  }, [reminders]);

  const formatDue = (dt, tz) => {
    try {
      return new Date(dt).toLocaleString('en-US', {
        timeZone: tz || 'UTC',
        month: 'short', day: 'numeric',
        hour: 'numeric', minute: '2-digit',
      });
    } catch {
      return new Date(dt).toLocaleString();
    }
  };

  const ReminderCard = ({ reminder, i }) => (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, x: -100 }}
      transition={{ delay: i * 0.04 }}
      className="reminder-card"
      data-testid={`reminder-${reminder.id}`}
    >
      <div className="reminder-content">
        <h3 className="reminder-title">{reminder.title}</h3>
        <p className="reminder-due">
          <Clock size={12} />
          {formatDue(reminder.due_datetime, reminder.timezone)}
        </p>
        <div className="flex gap-2 flex-wrap">
          {reminder.related_party && <span className="reminder-party">{reminder.related_party}</span>}
          {reminder.related_order_ref && <span className="reminder-ref">#{reminder.related_order_ref}</span>}
        </div>
      </div>
      <Button
        variant="outline"
        size="sm"
        onClick={() => markDone(reminder.id)}
        className="mark-done-btn shrink-0"
        data-testid={`done-${reminder.id}`}
      >
        <Check size={16} />
      </Button>
    </motion.div>
  );

  const Section = ({ title, icon: Icon, items, color }) => {
    if (items.length === 0) return null;
    return (
      <div className="reminder-section">
        <h2 className={`section-title ${color}`}>
          <Icon size={16} />
          {title} ({items.length})
        </h2>
        <AnimatePresence>
          {items.map((r, i) => <ReminderCard key={r.id} reminder={r} i={i} />)}
        </AnimatePresence>
      </div>
    );
  };

  return (
    <div className="page" data-testid="reminders-page">
      <div className="page-header">
        <h1 className="page-title">Reminders</h1>
      </div>

      <Section title="Overdue" icon={AlertTriangle} items={overdue} color="text-red-500" />
      <Section title="Today" icon={Calendar} items={today} color="text-amber-500" />
      <Section title="Upcoming" icon={Clock} items={upcoming} color="text-indigo-500" />

      {!loading && reminders.length === 0 && (
        <div className="empty-state">
          <Calendar size={48} className="text-muted-foreground/20" />
          <p className="text-muted-foreground text-sm">
            No open reminders. Chat to create one.
          </p>
        </div>
      )}
    </div>
  );
}
