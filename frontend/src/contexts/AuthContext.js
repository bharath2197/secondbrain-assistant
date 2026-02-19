import { createContext, useContext, useState, useEffect } from 'react';
import { supabase } from '@/lib/supabase';

const SUPABASE_URL = process.env.REACT_APP_SUPABASE_URL;
const SUPABASE_ANON_KEY = process.env.REACT_APP_SUPABASE_ANON_KEY;

const AuthContext = createContext({});

// XHR-based auth request — bypasses any window.fetch interceptor that
// consumes the response body (analytics scripts, Supabase GoTrue, etc.).
function parseResponse(status, responseText) {
  const ok = status >= 200 && status < 300;
  const raw = responseText || '';
  let data = null;
  try {
    data = raw ? JSON.parse(raw) : null;
  } catch {
    // not json
  }
  return { ok, status, raw, data };
}

function authFetch(endpoint, body) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${SUPABASE_URL}/auth/v1${endpoint}`);
    xhr.setRequestHeader('apikey', SUPABASE_ANON_KEY);
    xhr.setRequestHeader('Content-Type', 'application/json');

    xhr.onload = function () {
      const { ok, raw, data } = parseResponse(xhr.status, xhr.responseText);
      if (!ok) {
        reject(
          new Error(
            data?.error_description ||
              data?.msg ||
              data?.error ||
              raw ||
              'Request failed'
          )
        );
      } else {
        resolve(data);
      }
    };

    xhr.onerror = function () {
      reject(new Error('Network error'));
    };

    xhr.send(JSON.stringify(body));
  });
}

export function AuthProvider({ children }) {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;

    // Load initial session (persisted login)
    supabase.auth.getSession().then(({ data: { session: s } }) => {
      if (!mounted) return;
      setSession(s);
      setLoading(false);
    });

    // React to login/logout/refresh token events
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (_event, s) => {
        setSession(s);
        setLoading(false);
      }
    );

    return () => {
      mounted = false;
      subscription.unsubscribe();
    };
  }, []);

  const signUp = async (email, password) => {
    const data = await authFetch('/signup', { email, password });

    // If GoTrue returns tokens immediately, persist them into supabase client
    if (data?.access_token) {
      const { error } = await supabase.auth.setSession({
        access_token: data.access_token,
        refresh_token: data.refresh_token,
      });
      if (error) throw error;
    }

    return data;
  };

  const signIn = async (email, password) => {
    const data = await authFetch('/token?grant_type=password', { email, password });

    if (data?.access_token) {
      const { error } = await supabase.auth.setSession({
        access_token: data.access_token,
        refresh_token: data.refresh_token,
      });
      if (error) throw error;
    }

    return data;
  };

  const signOut = async () => {
    await supabase.auth.signOut();
    setSession(null);
    setLoading(false);
  };

  return (
    <AuthContext.Provider
      value={{
        session,
        user: session?.user,
        loading,
        signUp,
        signIn,
        signOut,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
