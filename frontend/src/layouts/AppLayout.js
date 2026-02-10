import { useState, useEffect } from 'react';
import { Outlet, NavLink } from 'react-router-dom';
import { MessageSquare, Brain, Bell, Settings } from 'lucide-react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useAuth } from '@/contexts/AuthContext';
import { toast } from 'sonner';
import axios from 'axios';

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

const TIMEZONES = [
  'America/New_York', 'America/Chicago', 'America/Denver', 'America/Los_Angeles',
  'America/Anchorage', 'Pacific/Honolulu', 'America/Sao_Paulo',
  'Europe/London', 'Europe/Berlin', 'Europe/Paris', 'Europe/Moscow',
  'Asia/Dubai', 'Asia/Kolkata', 'Asia/Shanghai', 'Asia/Tokyo', 'Asia/Singapore',
  'Australia/Sydney', 'Pacific/Auckland', 'UTC',
];

const navItems = [
  { to: '/', icon: MessageSquare, label: 'Chat', testId: 'nav-chat' },
  { to: '/kb', icon: Brain, label: 'Knowledge', testId: 'nav-kb' },
  { to: '/reminders', icon: Bell, label: 'Reminders', testId: 'nav-reminders' },
  { to: '/settings', icon: Settings, label: 'Settings', testId: 'nav-settings' },
];

export default function AppLayout() {
  const { session } = useAuth();
  const [showTzDialog, setShowTzDialog] = useState(false);
  const [detectedTz, setDetectedTz] = useState('UTC');
  const [selectedTz, setSelectedTz] = useState('UTC');

  const headers = { Authorization: `Bearer ${session?.access_token}` };

  useEffect(() => {
    checkProfile();
    // eslint-disable-next-line
  }, []);

  const checkProfile = async () => {
    try {
      const { data } = await axios.get(`${API}/profile`, { headers });
      if (!data || !data.timezone) {
        promptTimezone();
      }
    } catch {
      promptTimezone();
    }
  };

  const promptTimezone = () => {
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
    setDetectedTz(tz);
    setSelectedTz(tz);
    setShowTzDialog(true);
  };

  const saveTimezone = async () => {
    try {
      await axios.post(`${API}/profile`, { timezone: selectedTz }, { headers });
      setShowTzDialog(false);
      toast.success('Timezone saved');
    } catch {
      toast.error('Failed to save timezone');
    }
  };

  return (
    <div className="app-shell">
      <main className="app-content">
        <Outlet />
      </main>

      <nav className="bottom-nav" data-testid="bottom-nav">
        {navItems.map(({ to, icon: Icon, label, testId }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            data-testid={testId}
            className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
          >
            <Icon size={20} strokeWidth={1.5} />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>

      <Dialog open={showTzDialog} onOpenChange={setShowTzDialog}>
        <DialogContent className="max-w-sm">
          <DialogHeader>
            <DialogTitle>Confirm Your Timezone</DialogTitle>
          </DialogHeader>
          <p className="text-sm text-muted-foreground">
            Detected: <strong>{detectedTz}</strong>. Confirm or change below.
          </p>
          <Select value={selectedTz} onValueChange={setSelectedTz}>
            <SelectTrigger data-testid="tz-dialog-select">
              <SelectValue placeholder="Select timezone" />
            </SelectTrigger>
            <SelectContent>
              {TIMEZONES.map(tz => (
                <SelectItem key={tz} value={tz}>{tz.replace(/_/g, ' ')}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <DialogFooter>
            <Button onClick={saveTimezone} data-testid="confirm-timezone-btn">Confirm</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
