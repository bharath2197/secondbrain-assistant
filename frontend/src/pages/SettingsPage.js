import { useState, useEffect } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { LogOut } from 'lucide-react';
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

export default function SettingsPage() {
  const { session, signOut, user } = useAuth();
  const [timezone, setTimezone] = useState('UTC');
  const [autoSpeak, setAutoSpeak] = useState(false);
  const [saving, setSaving] = useState(false);

  const headers = { Authorization: `Bearer ${session?.access_token}` };

  useEffect(() => {
    loadProfile();
    setAutoSpeak(localStorage.getItem('autoSpeak') === 'true');
    // eslint-disable-next-line
  }, []);

  const loadProfile = async () => {
    try {
      const { data } = await axios.get(`${API}/profile`, { headers });
      if (data?.timezone) setTimezone(data.timezone);
    } catch {
      // Profile may not exist yet
    }
  };

  const saveTimezone = async (tz) => {
    setTimezone(tz);
    setSaving(true);
    try {
      await axios.post(`${API}/profile`, { timezone: tz }, { headers });
      toast.success('Timezone updated');
    } catch {
      toast.error('Failed to save');
    } finally {
      setSaving(false);
    }
  };

  const toggleAutoSpeak = (checked) => {
    setAutoSpeak(checked);
    localStorage.setItem('autoSpeak', checked.toString());
    toast.success(checked ? 'Auto-speak enabled' : 'Auto-speak disabled');
  };

  const handleSignOut = async () => {
    await signOut();
  };

  return (
    <div className="page" data-testid="settings-page">
      <div className="page-header">
        <h1 className="page-title">Settings</h1>
      </div>

      <div className="settings-section">
        <Label className="settings-label">Timezone</Label>
        <Select value={timezone} onValueChange={saveTimezone} disabled={saving}>
          <SelectTrigger data-testid="timezone-select">
            <SelectValue placeholder="Select timezone" />
          </SelectTrigger>
          <SelectContent>
            {TIMEZONES.map(tz => (
              <SelectItem key={tz} value={tz}>{tz.replace(/_/g, ' ')}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="settings-section">
        <div className="settings-row">
          <div>
            <Label className="settings-label">Auto-speak replies</Label>
            <p className="text-xs text-muted-foreground mt-0.5">
              Automatically speak short assistant confirmations
            </p>
          </div>
          <Switch
            checked={autoSpeak}
            onCheckedChange={toggleAutoSpeak}
            data-testid="auto-speak-toggle"
          />
        </div>
      </div>

      <div className="settings-section">
        <p className="text-sm text-muted-foreground mb-3">
          Signed in as <strong>{user?.email}</strong>
        </p>
        <Button variant="destructive" onClick={handleSignOut} data-testid="sign-out-btn">
          <LogOut size={16} className="mr-2" />
          Sign Out
        </Button>
      </div>
    </div>
  );
}
