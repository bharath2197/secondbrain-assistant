import { useState, useEffect } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { Search, Trash2, FileText, User, Package, Lightbulb } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

const typeIcons = { note: FileText, contact: User, order: Package, fact: Lightbulb };
const typeColors = {
  note: 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300',
  contact: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300',
  order: 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300',
  fact: 'bg-violet-100 text-violet-700 dark:bg-violet-900/40 dark:text-violet-300',
};

export default function KBPage() {
  const { session } = useAuth();
  const [entries, setEntries] = useState([]);
  const [search, setSearch] = useState('');
  const [filterType, setFilterType] = useState('');
  const [loading, setLoading] = useState(true);

  const headers = { Authorization: `Bearer ${session?.access_token}` };

  useEffect(() => {
    loadEntries();
    // eslint-disable-next-line
  }, [filterType]);

  const loadEntries = async () => {
    setLoading(true);
    try {
      const params = {};
      if (filterType) params.entity_type = filterType;
      if (search) params.search = search;
      const { data } = await axios.get(`${API}/kb`, { headers, params });
      setEntries(data);
    } catch {
      toast.error('Failed to load entries');
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = (e) => {
    e.preventDefault();
    loadEntries();
  };

  const deleteEntry = async (id) => {
    try {
      await axios.delete(`${API}/kb/${id}`, { headers });
      setEntries(prev => prev.filter(e => e.id !== id));
      toast.success('Entry deleted');
    } catch {
      toast.error('Failed to delete');
    }
  };

  return (
    <div className="page" data-testid="kb-page">
      <div className="page-header">
        <h1 className="page-title">Knowledge Base</h1>
      </div>

      <form onSubmit={handleSearch} className="search-bar">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" size={16} />
          <Input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search entries..."
            className="pl-9"
            data-testid="kb-search-input"
          />
        </div>
      </form>

      <div className="filter-chips">
        {['', 'note', 'contact', 'order', 'fact'].map(type => (
          <button
            key={type}
            onClick={() => setFilterType(type)}
            className={`chip ${filterType === type ? 'active' : ''}`}
            data-testid={`filter-${type || 'all'}`}
          >
            {type || 'All'}
          </button>
        ))}
      </div>

      <div className="entries-list">
        <AnimatePresence>
          {entries.map((entry, i) => {
            const Icon = typeIcons[entry.entity_type] || FileText;
            return (
              <motion.div
                key={entry.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ delay: i * 0.04 }}
                className="entry-card"
                data-testid={`kb-entry-${entry.id}`}
              >
                <div className="entry-header">
                  <span className={`entry-type ${typeColors[entry.entity_type] || ''}`}>
                    <Icon size={12} />
                    {entry.entity_type}
                  </span>
                  {entry.entity_name && <span className="entry-name">{entry.entity_name}</span>}
                  {entry.order_ref && <span className="entry-ref">#{entry.order_ref}</span>}
                </div>
                <p className="entry-details">{entry.details}</p>
                <div className="entry-footer">
                  <span className="entry-date">
                    {new Date(entry.created_at).toLocaleDateString()}
                  </span>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => deleteEntry(entry.id)}
                    data-testid={`delete-kb-${entry.id}`}
                    className="h-8 w-8 p-0 text-muted-foreground hover:text-destructive"
                  >
                    <Trash2 size={14} />
                  </Button>
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>
        {!loading && entries.length === 0 && (
          <div className="empty-state">
            <FileText size={48} className="text-muted-foreground/20" />
            <p className="text-muted-foreground text-sm">
              No entries yet. Start chatting to build your knowledge base.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
