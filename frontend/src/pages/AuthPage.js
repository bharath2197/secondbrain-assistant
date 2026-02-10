import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { toast } from 'sonner';

export default function AuthPage() {
  const { signUp, signIn, session } = useAuth();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  useEffect(() => {
    if (session) navigate('/', { replace: true });
  }, [session, navigate]);

  const handleSubmit = async (action) => {
    if (!email || !password) {
      toast.error('Please fill in all fields');
      return;
    }
    if (action === 'signup' && password.length < 6) {
      toast.error('Password must be at least 6 characters');
      return;
    }
    setLoading(true);
    try {
      if (action === 'signup') {
        const data = await signUp(email, password);
        if (data.session) {
          toast.success('Account created!');
          navigate('/');
        } else {
          toast.success('Account created! Check your email to confirm, then log in.');
        }
      } else {
        await signIn(email, password);
        toast.success('Welcome back!');
        navigate('/');
      }
    } catch (e) {
      toast.error(e.message || 'Authentication failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page" data-testid="auth-page">
      <div className="auth-card">
        <h1 className="auth-title">SecondBrain</h1>
        <p className="auth-subtitle">Your personal knowledge assistant</p>

        <Tabs defaultValue="login">
          <TabsList className="w-full">
            <TabsTrigger value="login" className="flex-1" data-testid="login-tab">Log In</TabsTrigger>
            <TabsTrigger value="signup" className="flex-1" data-testid="signup-tab">Sign Up</TabsTrigger>
          </TabsList>

          <TabsContent value="login">
            <div className="space-y-4 mt-4">
              <div>
                <Label htmlFor="login-email">Email</Label>
                <Input
                  id="login-email"
                  type="email"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  data-testid="login-email-input"
                />
              </div>
              <div>
                <Label htmlFor="login-password">Password</Label>
                <Input
                  id="login-password"
                  type="password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  placeholder="Your password"
                  data-testid="login-password-input"
                  onKeyDown={e => e.key === 'Enter' && handleSubmit('login')}
                />
              </div>
              <Button
                className="w-full"
                onClick={() => handleSubmit('login')}
                disabled={loading}
                data-testid="login-submit-btn"
              >
                {loading ? 'Logging in...' : 'Log In'}
              </Button>
            </div>
          </TabsContent>

          <TabsContent value="signup">
            <div className="space-y-4 mt-4">
              <div>
                <Label htmlFor="signup-email">Email</Label>
                <Input
                  id="signup-email"
                  type="email"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  data-testid="signup-email-input"
                />
              </div>
              <div>
                <Label htmlFor="signup-password">Password</Label>
                <Input
                  id="signup-password"
                  type="password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  placeholder="Min 6 characters"
                  data-testid="signup-password-input"
                  onKeyDown={e => e.key === 'Enter' && handleSubmit('signup')}
                />
              </div>
              <Button
                className="w-full"
                onClick={() => handleSubmit('signup')}
                disabled={loading}
                data-testid="signup-submit-btn"
              >
                {loading ? 'Creating account...' : 'Sign Up'}
              </Button>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
