import React, { useState, useEffect, useCallback, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import {
  checkHealth, indexRepository, queryCodebaseStream, listRepos, deleteRepo,
  RepoInfo, SourceChunk,
} from './api';

type Theme = 'dark' | 'light';

type ChatMessage = {
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceChunk[];
  streaming?: boolean;
};

const EXAMPLE_QUERIES = [
  'What does this codebase do, at a high level?',
  'Where is the application entry point?',
  'How is authentication handled?',
  'Explain the main data flow.',
];

const statusMeta: Record<string, { color: string; icon: string }> = {
  ready: { color: 'text-ok', icon: 'database' },
  indexing: { color: 'text-warn', icon: 'progress_activity' },
  error: { color: 'text-danger', icon: 'error' },
};

const baseName = (p: string) => p.split('/').pop() || p;

export default function App() {
  const [theme, setTheme] = useState<Theme>(() =>
    typeof document !== 'undefined' && document.documentElement.classList.contains('light') ? 'light' : 'dark'
  );
  const [apiOnline, setApiOnline] = useState(false);
  const [inputType, setInputType] = useState<'github' | 'local'>('github');
  const [repoInput, setRepoInput] = useState('');
  const [isIndexing, setIsIndexing] = useState(false);
  const [repos, setRepos] = useState<RepoInfo[]>([]);
  const [activeRepoId, setActiveRepoId] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [isQuerying, setIsQuerying] = useState(false);
  // Conversations are kept per-repo so switching repos preserves each chat.
  const [chats, setChats] = useState<Record<string, ChatMessage[]>>({});
  const scrollRef = useRef<HTMLDivElement>(null);

  const activeRepo = repos.find(r => r.repo_id === activeRepoId) || null;
  const messages = activeRepoId ? (chats[activeRepoId] || []) : [];
  const canChat = !!activeRepo && activeRepo.status === 'ready' && apiOnline;
  const anyIndexing = repos.some(r => r.status === 'indexing');

  // Sources of the most recent assistant turn drive the retrieval panel.
  const lastAssistant = [...messages].reverse().find(m => m.role === 'assistant');
  const panelSources = lastAssistant?.sources || [];

  const toggleTheme = () => {
    setTheme(prev => {
      const next: Theme = prev === 'dark' ? 'light' : 'dark';
      const root = document.documentElement;
      root.classList.remove('dark', 'light');
      root.classList.add(next);
      try { localStorage.setItem('codelens-theme', next); } catch { /* ignore */ }
      return next;
    });
  };

  const refreshRepos = useCallback(async () => {
    const list = await listRepos();
    setRepos(list);
    setActiveRepoId(prev => {
      if (prev && list.some(r => r.repo_id === prev)) return prev;
      const firstReady = list.find(r => r.status === 'ready');
      return firstReady ? firstReady.repo_id : null;
    });
  }, []);

  useEffect(() => {
    const tick = async () => {
      const online = await checkHealth();
      setApiOnline(online);
      if (online) refreshRepos();
    };
    tick();
    const interval = setInterval(tick, anyIndexing ? 1500 : 5000);
    return () => clearInterval(interval);
  }, [refreshRepos, anyIndexing]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages.length, isQuerying]);

  const handleIndex = async () => {
    if (!repoInput.trim()) return;
    setIsIndexing(true);
    try {
      const res = await indexRepository(repoInput.trim());
      setRepoInput('');
      await refreshRepos();
      setActiveRepoId(res.repo_id);
    } catch (e) {
      alert('Failed to index: ' + (e as Error).message);
    } finally {
      setIsIndexing(false);
    }
  };

  const handleDelete = async (repoId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm('Delete this indexed repository?')) return;
    try {
      await deleteRepo(repoId);
      setChats(prev => { const next = { ...prev }; delete next[repoId]; return next; });
      await refreshRepos();
    } catch (err) {
      alert('Failed to delete: ' + (err as Error).message);
    }
  };

  const runQuery = async (text: string) => {
    const q = text.trim();
    if (!q || !canChat || isQuerying || !activeRepoId) return;

    const repoId = activeRepoId;
    setQuery('');
    const prevMessages = chats[repoId] || [];
    const contextHistory = prevMessages.slice(-4);
    const assistantIndex = prevMessages.length + 1;
    setChats(prev => ({
      ...prev,
      [repoId]: [
        ...prevMessages,
        { role: 'user', content: q },
        { role: 'assistant', content: '', sources: [], streaming: true },
      ],
    }));
    setIsQuerying(true);

    const updateAssistant = (updater: (m: ChatMessage) => ChatMessage) => setChats(prev => {
      const list = [...(prev[repoId] || [])];
      if (list[assistantIndex]) list[assistantIndex] = updater(list[assistantIndex]);
      return { ...prev, [repoId]: list };
    });

    try {
      await queryCodebaseStream(repoId, q, contextHistory, {
        onSources: (sources) => updateAssistant(m => ({ ...m, sources })),
        onToken: (text) => updateAssistant(m => ({ ...m, content: m.content + text })),
      });
      updateAssistant(m => ({ ...m, streaming: false }));
    } catch (err) {
      updateAssistant(m => ({ ...m, streaming: false, content: m.content || ('**Error:** ' + (err as Error).message) }));
    } finally {
      setIsQuerying(false);
    }
  };

  const codeStyle = theme === 'dark' ? oneDark : oneLight;
  const awaitingSources = isQuerying && panelSources.length === 0;

  return (
    <div className="flex h-screen w-full bg-canvas text-txt overflow-hidden font-inter">
      {/* ───────────────── SIDEBAR ───────────────── */}
      <aside className="flex flex-col h-full w-64 shrink-0 bg-panel border-r border-line">
        <div className="flex items-center gap-2.5 px-4 h-14 border-b border-line">
          <Logo />
          <div className="flex flex-col leading-none">
            <span className="text-[15px] font-semibold tracking-tight">
              code<span className="text-accent">lens</span>
            </span>
            <span className="text-[9px] font-mono text-faint tracking-widest uppercase mt-0.5">hybrid code RAG</span>
          </div>
        </div>

        <div className="p-3 space-y-3">
          <div className="bg-canvas border border-line p-0.5 rounded-md flex gap-0.5">
            {(['github', 'local'] as const).map(t => (
              <button
                key={t}
                onClick={() => setInputType(t)}
                className={`flex-1 text-[11px] font-medium py-1.5 rounded transition-colors ${
                  inputType === t ? 'bg-panel2 text-txt shadow-sm' : 'text-muted hover:text-txt'
                }`}
              >{t === 'github' ? 'GitHub URL' : 'Local Path'}</button>
            ))}
          </div>

          <div className="relative">
            <input
              className="w-full bg-canvas border border-line focus:border-accent rounded-md py-2 pl-3 pr-3 text-[13px] font-mono text-txt placeholder-faint outline-none transition-colors"
              placeholder={inputType === 'github' ? 'github.com/owner/repo' : '/path/to/repo'}
              value={repoInput}
              onChange={e => setRepoInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') handleIndex(); }}
            />
          </div>

          <button
            onClick={handleIndex}
            disabled={isIndexing || !apiOnline || !repoInput.trim()}
            className="w-full bg-accent text-canvas font-semibold text-[13px] py-2 rounded-md flex items-center justify-center gap-2 transition-all hover:opacity-90 active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <span className={`material-symbols-outlined text-[18px] ${isIndexing ? 'animate-spin' : ''}`}>
              {isIndexing ? 'progress_activity' : 'add'}
            </span>
            {isIndexing ? 'Indexing…' : 'Index Repository'}
          </button>
        </div>

        {/* Indexed repos */}
        <div className="flex flex-col flex-1 min-h-0 px-3">
          <div className="flex items-center gap-1.5 px-1 py-2">
            <span className="text-[10px] font-semibold text-faint tracking-widest uppercase">Repositories</span>
            <span className="ml-auto text-[10px] font-mono text-faint bg-canvas border border-line rounded px-1.5">{repos.length}</span>
          </div>
          <div className="flex flex-col gap-0.5 overflow-y-auto custom-scrollbar pb-2">
            {repos.length === 0 && (
              <p className="text-[11px] text-faint px-2 py-6 text-center leading-relaxed">
                No repositories yet.<br />Index one to begin.
              </p>
            )}
            {repos.map(repo => {
              const meta = statusMeta[repo.status] || statusMeta.ready;
              const active = repo.repo_id === activeRepoId;
              return (
                <div
                  key={repo.repo_id}
                  onClick={() => setActiveRepoId(repo.repo_id)}
                  className={`group rounded-md px-2.5 py-2 flex items-center gap-2 cursor-pointer border transition-colors ${
                    active ? 'bg-panel2 border-line' : 'border-transparent hover:bg-panel2/60'
                  }`}
                >
                  <span className={`material-symbols-outlined text-[16px] ${meta.color} ${repo.status === 'indexing' ? 'animate-spin' : ''}`}>
                    {meta.icon}
                  </span>
                  <div className="flex flex-col min-w-0 flex-1">
                    <span className="text-[12px] font-medium truncate">{repo.display_name}</span>
                    <span className={`text-[10px] font-mono truncate ${
                      repo.status === 'error' ? 'text-danger' : repo.status === 'indexing' ? 'text-warn' : 'text-faint'
                    }`}>
                      {repo.status === 'indexing'
                        ? (repo.progress || 'indexing…')
                        : repo.status === 'error'
                          ? (repo.error || 'failed')
                          : `${repo.chunks_created.toLocaleString()} chunks`}
                    </span>
                  </div>
                  <button
                    onClick={e => handleDelete(repo.repo_id, e)}
                    className="opacity-0 group-hover:opacity-100 text-faint hover:text-danger transition-all"
                    title="Delete repository"
                  >
                    <span className="material-symbols-outlined text-[16px]">delete</span>
                  </button>
                </div>
              );
            })}
          </div>
        </div>

        {/* Footer: status + theme */}
        <div className="px-3 py-3 border-t border-line flex items-center gap-2">
          <div className="flex items-center gap-1.5 flex-1 min-w-0">
            <span className={`w-1.5 h-1.5 rounded-full ${apiOnline ? 'bg-ok' : 'bg-danger'} ${apiOnline ? 'shadow-[0_0_6px_rgb(var(--ok))]' : ''}`} />
            <span className={`text-[11px] font-mono ${apiOnline ? 'text-muted' : 'text-danger'}`}>
              api {apiOnline ? 'online' : 'offline'}
            </span>
          </div>
          <button
            onClick={toggleTheme}
            className="text-faint hover:text-txt transition-colors p-1 rounded hover:bg-panel2"
            title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          >
            <span className="material-symbols-outlined text-[18px]">{theme === 'dark' ? 'light_mode' : 'dark_mode'}</span>
          </button>
        </div>
      </aside>

      {/* ───────────────── MAIN ───────────────── */}
      <main className="flex-1 flex flex-col min-w-0 relative">
        <header className="flex items-center h-14 px-5 border-b border-line shrink-0 gap-3">
          <span className="material-symbols-outlined text-[18px] text-faint">terminal</span>
          {activeRepo ? (
            <div className="flex items-center gap-2 font-mono text-[13px] min-w-0">
              <span className="text-muted shrink-0">repo</span>
              <span className="text-faint">/</span>
              <span className="text-txt truncate">{activeRepo.display_name}</span>
            </div>
          ) : (
            <span className="font-mono text-[13px] text-faint">no repository selected</span>
          )}
          {activeRepo && (
            <span className={`ml-auto text-[11px] font-mono ${statusMeta[activeRepo.status]?.color}`}>
              ● {activeRepo.status}
            </span>
          )}
        </header>

        <div ref={scrollRef} className="flex-1 overflow-y-auto custom-scrollbar">
          {messages.length === 0 ? (
            <EmptyState
              activeRepo={activeRepo}
              canChat={canChat}
              onExample={(q) => runQuery(q)}
            />
          ) : (
            <div className="max-w-3xl mx-auto w-full px-5 py-6 space-y-6 pb-40">
              {messages.map((msg, idx) => (
                <MessageBubble key={idx} msg={msg} codeStyle={codeStyle} />
              ))}
            </div>
          )}
        </div>

        {/* Input */}
        <div className="absolute bottom-0 left-0 right-0 px-5 pb-5 pt-10 bg-gradient-to-t from-canvas via-canvas to-transparent pointer-events-none">
          <form
            onSubmit={(e) => { e.preventDefault(); runQuery(query); }}
            className="max-w-3xl mx-auto pointer-events-auto flex items-center gap-2.5 bg-panel border border-line focus-within:border-accent rounded-lg px-3.5 py-2.5 transition-colors shadow-lg"
          >
            <span className="text-accent font-mono text-sm select-none">{'>'}</span>
            <input
              className="flex-1 bg-transparent text-[14px] text-txt placeholder-faint outline-none font-mono"
              placeholder={canChat ? 'Ask anything about this codebase…' : 'Select a ready repository to start…'}
              value={query}
              onChange={e => setQuery(e.target.value)}
              disabled={!canChat || isQuerying}
            />
            <button
              type="submit"
              disabled={!canChat || isQuerying || !query.trim()}
              className="w-8 h-8 rounded-md bg-accent text-canvas flex items-center justify-center hover:opacity-90 active:scale-95 transition-all disabled:opacity-40"
            >
              <span className={`material-symbols-outlined text-[18px] ${isQuerying ? 'animate-spin' : ''}`}>
                {isQuerying ? 'progress_activity' : 'arrow_upward'}
              </span>
            </button>
          </form>
        </div>
      </main>

      {/* ───────────────── RETRIEVAL PANEL ───────────────── */}
      <aside className="w-80 shrink-0 border-l border-line bg-panel hidden lg:flex flex-col">
        <div className="flex items-center gap-2 h-14 px-5 border-b border-line shrink-0">
          <span className="material-symbols-outlined text-[18px] text-accent">manage_search</span>
          <span className="text-[13px] font-semibold">Retrieval</span>
          {panelSources.length > 0 && (
            <span className="ml-auto text-[10px] font-mono text-faint bg-canvas border border-line rounded px-1.5">
              {panelSources.length} chunks
            </span>
          )}
        </div>

        <div className="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-4">
          <RetrievalLegend />
          {activeRepo && (
            <div className="grid grid-cols-2 gap-2">
              <Stat label="files" value={activeRepo.files_indexed.toLocaleString()} />
              <Stat label="chunks" value={activeRepo.chunks_created.toLocaleString()} />
            </div>
          )}

          {awaitingSources ? (
            <ScanSkeleton />
          ) : panelSources.length > 0 ? (
            <div className="space-y-2.5">
              <div className="text-[10px] font-semibold text-faint tracking-widest uppercase pt-1">
                Why these chunks
              </div>
              {panelSources.map((src, i) => (
                <SourceCard key={i} src={src} maxRrf={Math.max(...panelSources.map(s => s.retrieval?.rrf_score || 0))} />
              ))}
            </div>
          ) : (
            <p className="text-[11px] text-faint leading-relaxed pt-2">
              Ask a question and CodeLens will show exactly which code chunks it retrieved —
              and whether each came from <span className="text-vector">semantic</span> or{' '}
              <span className="text-keyword">keyword</span> search.
            </p>
          )}
        </div>
      </aside>
    </div>
  );
}

/* ─────────────────────── sub-components ─────────────────────── */

function Logo() {
  return (
    <div className="relative w-7 h-7 shrink-0">
      <div className="absolute inset-0 rounded-md border-2 border-accent" />
      <div className="absolute inset-[5px] rounded-full border border-accent/60" />
      <div className="absolute inset-[10px] rounded-full bg-accent" />
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-canvas border border-line rounded-md px-3 py-2">
      <div className="text-[16px] font-mono font-semibold text-txt leading-none">{value}</div>
      <div className="text-[10px] text-faint uppercase tracking-wider mt-1">{label}</div>
    </div>
  );
}

function RetrievalLegend() {
  return (
    <div className="bg-canvas border border-line rounded-md p-3 space-y-2">
      <div className="text-[10px] font-semibold text-faint tracking-widest uppercase">Hybrid retrieval</div>
      <div className="flex items-center gap-2 text-[11px]">
        <span className="w-2 h-2 rounded-sm bg-vector shrink-0" />
        <span className="text-muted"><span className="text-vector font-medium">vector</span> — semantic / embedding match</span>
      </div>
      <div className="flex items-center gap-2 text-[11px]">
        <span className="w-2 h-2 rounded-sm bg-keyword shrink-0" />
        <span className="text-muted"><span className="text-keyword font-medium">keyword</span> — exact BM25 lexical match</span>
      </div>
      <div className="text-[10px] text-faint leading-relaxed pt-1 border-t border-line">
        Results are fused with Reciprocal Rank Fusion (RRF).
      </div>
    </div>
  );
}

function ScanSkeleton() {
  return (
    <div className="space-y-2.5 pt-1">
      <div className="text-[10px] font-semibold text-faint tracking-widest uppercase">Retrieving…</div>
      {[0, 1, 2].map(i => (
        <div key={i} className="relative overflow-hidden bg-canvas border border-line rounded-md h-16 p-3">
          <div className="absolute left-0 right-0 h-8 bg-gradient-to-b from-accent/20 to-transparent animate-scan" />
          <div className="h-2 w-2/3 rounded bg-line mb-2" />
          <div className="h-2 w-1/3 rounded bg-line" />
        </div>
      ))}
    </div>
  );
}

function MatchBadge({ kind }: { kind: string }) {
  const isVector = kind === 'vector';
  return (
    <span className={`inline-flex items-center gap-1 text-[9px] font-mono font-medium px-1.5 py-0.5 rounded border ${
      isVector ? 'text-vector border-vector/30 bg-vector/10' : 'text-keyword border-keyword/30 bg-keyword/10'
    }`}>
      <span className="material-symbols-outlined text-[11px]">{isVector ? 'hub' : 'tag'}</span>
      {kind}
    </span>
  );
}

function SourceCard({ src, maxRrf }: { src: SourceChunk; maxRrf: number }) {
  const r = src.retrieval;
  const matched = r?.matched_by || [];
  const pct = r && maxRrf > 0 ? Math.round((r.rrf_score / maxRrf) * 100) : 0;
  const snippet = (src.content || '').split('\n').slice(0, 3);

  return (
    <div className="bg-canvas border border-line rounded-md overflow-hidden animate-fade-up">
      <div className="px-3 pt-2.5 pb-2 space-y-2">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono text-faint shrink-0">#{(r?.rank ?? 0) + 1}</span>
          <span className="material-symbols-outlined text-[14px] text-faint shrink-0">description</span>
          <span className="text-[11px] font-mono text-txt truncate" title={src.file_path}>{baseName(src.file_path)}</span>
        </div>

        <div className="flex flex-wrap items-center gap-1">
          {matched.map(m => <MatchBadge key={m} kind={m} />)}
          {src.chunk_type && (
            <span className="text-[9px] font-mono text-muted bg-panel2 border border-line rounded px-1.5 py-0.5">{src.chunk_type}</span>
          )}
          {src.language && (
            <span className="text-[9px] font-mono text-faint">{src.language}</span>
          )}
          {(src.start_line || src.end_line) ? (
            <span className="ml-auto text-[9px] font-mono text-faint">L{src.start_line}–{src.end_line}</span>
          ) : null}
        </div>

        {/* Score breakdown */}
        <div className="flex items-center gap-2 text-[9px] font-mono text-faint">
          <div className="flex-1 h-1 rounded-full bg-panel2 overflow-hidden">
            <div className="h-full bg-accent origin-left animate-bar-grow" style={{ transform: `scaleX(${Math.max(0.04, pct / 100)})` }} />
          </div>
          <span title="Reciprocal Rank Fusion score">rrf {r?.rrf_score?.toFixed(4) ?? '—'}</span>
          {r?.bm25_score != null && <span className="text-keyword" title="BM25 lexical score">bm25 {r.bm25_score.toFixed(2)}</span>}
          {r?.dense_rank != null && <span className="text-vector" title="Rank in vector results">v#{r.dense_rank + 1}</span>}
        </div>
      </div>

      {snippet.length > 0 && (
        <div className="border-t border-line bg-panel px-3 py-2 code-gutter font-mono text-[10px] text-muted leading-relaxed overflow-x-auto">
          {snippet.map((ln, i) => (
            <div key={i} className="ln whitespace-pre">{ln || ' '}</div>
          ))}
        </div>
      )}
    </div>
  );
}

function MessageBubble({ msg, codeStyle }: { msg: ChatMessage; codeStyle: any }) {
  const isUser = msg.role === 'user';
  return (
    <div className={`flex gap-3 animate-fade-up ${isUser ? 'justify-end' : 'justify-start'}`}>
      {!isUser && (
        <div className="w-7 h-7 rounded-md bg-panel border border-line flex items-center justify-center shrink-0 mt-0.5">
          <span className="material-symbols-outlined text-[16px] text-accent">auto_awesome</span>
        </div>
      )}
      <div className={isUser ? 'max-w-[80%]' : 'flex-1 min-w-0'}>
        <div className={isUser
          ? 'bg-panel2 border border-line px-3.5 py-2.5 rounded-lg rounded-tr-sm'
          : 'bg-panel border border-line px-4 py-3 rounded-lg rounded-tl-sm'}>
          {isUser ? (
            <p className="text-[13px] leading-relaxed text-txt font-mono">{msg.content}</p>
          ) : (!msg.content && msg.streaming) ? (
            <div className="flex items-center gap-1.5 py-0.5">
              <span className="w-1.5 h-1.5 rounded-full bg-accent animate-bounce [animation-delay:-0.3s]" />
              <span className="w-1.5 h-1.5 rounded-full bg-accent animate-bounce [animation-delay:-0.15s]" />
              <span className="w-1.5 h-1.5 rounded-full bg-accent animate-bounce" />
              <span className="text-[12px] text-faint ml-1 font-mono">analyzing…</span>
            </div>
          ) : (
            <article className="md max-w-none text-[13px] leading-relaxed">
              <ReactMarkdown
                components={{
                  code({ inline, className, children, ...props }: any) {
                    const match = /language-(\w+)/.exec(className || '');
                    return !inline && match ? (
                      <SyntaxHighlighter
                        style={codeStyle}
                        language={match[1]}
                        PreTag="div"
                        customStyle={{ borderRadius: '0.5rem', fontSize: '12px', margin: 0, background: 'rgb(var(--canvas))' }}
                        {...props}
                      >
                        {String(children).replace(/\n$/, '')}
                      </SyntaxHighlighter>
                    ) : (
                      <code className="bg-panel2 border border-line px-1 py-0.5 rounded text-[12px] font-mono" {...props}>
                        {children}
                      </code>
                    );
                  },
                }}
              >
                {msg.content}
              </ReactMarkdown>
              {msg.streaming && <span className="inline-block w-1.5 h-3.5 bg-accent ml-0.5 animate-blink align-middle" />}
            </article>
          )}
        </div>

        {/* Compact citation strip */}
        {!isUser && msg.sources && msg.sources.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mt-2 pl-1">
            {msg.sources.slice(0, 6).map((src, i) => (
              <span key={i} className="inline-flex items-center gap-1 text-[10px] font-mono text-muted bg-panel border border-line rounded px-1.5 py-0.5">
                <span className="material-symbols-outlined text-[11px] text-faint">description</span>
                {baseName(src.file_path)}
                {(src.retrieval?.matched_by || []).map(m => (
                  <span key={m} className={`w-1.5 h-1.5 rounded-full ${m === 'vector' ? 'bg-vector' : 'bg-keyword'}`} />
                ))}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function EmptyState({ activeRepo, canChat, onExample }: {
  activeRepo: RepoInfo | null;
  canChat: boolean;
  onExample: (q: string) => void;
}) {
  return (
    <div className="relative h-full flex flex-col items-center justify-center px-6 text-center">
      <div className="absolute inset-0 grid-bg pointer-events-none" />
      <div className="relative z-10 max-w-md space-y-5">
        <div className="flex justify-center"><Logo /></div>
        <div className="space-y-2">
          <h2 className="text-2xl font-semibold tracking-tight">
            Ask your codebase anything.
          </h2>
          <p className="text-[13px] text-muted leading-relaxed">
            {activeRepo
              ? <>Querying <span className="font-mono text-txt">{activeRepo.display_name}</span> with hybrid semantic + keyword retrieval. Every answer is grounded in real code and shows its sources.</>
              : 'Index a repository from the sidebar, then select it to start a grounded conversation about the code.'}
          </p>
        </div>

        {canChat && (
          <div className="grid grid-cols-1 gap-2 pt-2">
            {EXAMPLE_QUERIES.map(q => (
              <button
                key={q}
                onClick={() => onExample(q)}
                className="group text-left text-[13px] font-mono text-muted hover:text-txt bg-panel border border-line hover:border-accent rounded-md px-3 py-2 transition-colors flex items-center gap-2"
              >
                <span className="text-accent select-none">{'>'}</span>
                <span className="flex-1">{q}</span>
                <span className="material-symbols-outlined text-[15px] text-faint opacity-0 group-hover:opacity-100 transition-opacity">arrow_forward</span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
