import React, { useState, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { checkHealth, indexRepository, queryCodebaseStream, listRepos, deleteRepo, RepoInfo } from './api';

const statusColor: Record<string, string> = {
  ready: 'text-emerald-500',
  indexing: 'text-amber-500',
  error: 'text-red-500',
};

export default function App() {
  const [apiOnline, setApiOnline] = useState(false);
  const [inputType, setInputType] = useState<'github' | 'local'>('github');
  const [repoInput, setRepoInput] = useState("");
  const [isIndexing, setIsIndexing] = useState(false);
  const [repos, setRepos] = useState<RepoInfo[]>([]);
  const [activeRepoId, setActiveRepoId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [isQuerying, setIsQuerying] = useState(false);
  // Conversations are kept per-repo so switching repos preserves each chat.
  const [chats, setChats] = useState<Record<string, any[]>>({});

  const activeRepo = repos.find(r => r.repo_id === activeRepoId) || null;
  const messages = activeRepoId ? (chats[activeRepoId] || []) : [];
  const canChat = !!activeRepo && activeRepo.status === 'ready' && apiOnline;
  const anyIndexing = repos.some(r => r.status === 'indexing');

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
    // Poll faster while a repo is actively indexing so status/progress feels live.
    const interval = setInterval(tick, anyIndexing ? 1500 : 5000);
    return () => clearInterval(interval);
  }, [refreshRepos, anyIndexing]);

  const handleIndex = async () => {
    if (!repoInput.trim()) return;
    setIsIndexing(true);
    try {
      const res = await indexRepository(repoInput.trim());
      setRepoInput("");
      await refreshRepos();
      setActiveRepoId(res.repo_id);
    } catch (e) {
      alert("Failed to index: " + (e as Error).message);
    } finally {
      setIsIndexing(false);
    }
  };

  const handleDelete = async (repoId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm("Delete this indexed repository?")) return;
    try {
      await deleteRepo(repoId);
      setChats(prev => {
        const next = { ...prev };
        delete next[repoId];
        return next;
      });
      await refreshRepos();
    } catch (err) {
      alert("Failed to delete: " + (err as Error).message);
    }
  };

  const handleQuery = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!query.trim() || !canChat || isQuerying || !activeRepoId) return;

    const repoId = activeRepoId;
    const newQuery = query;
    setQuery("");
    const prevMessages = chats[repoId] || [];
    const contextHistory = prevMessages.slice(-4);
    const assistantIndex = prevMessages.length + 1; // sits after the new user message
    setChats(prev => ({
      ...prev,
      [repoId]: [
        ...prevMessages,
        { role: "user", content: newQuery },
        { role: "assistant", content: "", sources: [], streaming: true },
      ],
    }));
    setIsQuerying(true);

    const updateAssistant = (updater: (m: any) => any) => setChats(prev => {
      const list = [...(prev[repoId] || [])];
      if (list[assistantIndex]) list[assistantIndex] = updater(list[assistantIndex]);
      return { ...prev, [repoId]: list };
    });

    try {
      await queryCodebaseStream(repoId, newQuery, contextHistory, {
        onSources: (sources) => updateAssistant(m => ({ ...m, sources })),
        onToken: (text) => updateAssistant(m => ({ ...m, content: m.content + text })),
      });
      updateAssistant(m => ({ ...m, streaming: false }));
    } catch (err) {
      updateAssistant(m => ({
        ...m,
        streaming: false,
        content: m.content || ("Error: " + (err as Error).message),
      }));
    } finally {
      setIsQuerying(false);
    }
  };

  return (
    <div className="flex h-screen w-full bg-surface text-on-surface overflow-hidden">
      {/* SIDEBAR */}
      <aside className="flex flex-col h-full p-4 gap-y-2 bg-zinc-900/50 dark:bg-zinc-950/50 backdrop-blur-xl w-64 border-r border-zinc-800/20 shadow-[20px_0_50px_rgba(0,0,0,0.3)] font-inter tracking-tight antialiased">
        <div className="flex items-center gap-3 px-2 mb-6">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-zinc-100 to-zinc-500 flex items-center justify-center text-on-primary">
            <span className="material-symbols-outlined text-sm">terminal</span>
          </div>
          <div className="flex flex-col">
            <span className="text-lg font-semibold tracking-tighter text-zinc-100">CodeLens AI</span>
            <span className="text-[10px] text-zinc-500 uppercase tracking-widest">v1.1.0-multirepo</span>
          </div>
        </div>

        <div className="bg-surface-container-low p-1 rounded-lg flex gap-1 mb-4 ghost-border">
          <button
            onClick={() => setInputType('github')}
            className={`flex-1 text-[11px] font-medium py-1.5 rounded transition-colors ${inputType === 'github' ? 'bg-zinc-800/50 text-zinc-100 shadow-inner' : 'text-zinc-500 hover:text-zinc-300'}`}
          >GitHub URL</button>
          <button
            onClick={() => setInputType('local')}
            className={`flex-1 text-[11px] font-medium py-1.5 rounded transition-colors ${inputType === 'local' ? 'bg-zinc-800/50 text-zinc-100 shadow-inner' : 'text-zinc-500 hover:text-zinc-300'}`}
          >Local Path</button>
        </div>

        <div className="space-y-4">
          <div className="space-y-1.5">
            <label className="text-[10px] text-on-surface-variant font-medium ml-1">
              {inputType === 'github' ? 'REPOSITORY URL' : 'LOCAL PATH'}
            </label>
            <input
              className="w-full bg-zinc-900 border-none ring-1 ring-zinc-800/50 focus:ring-zinc-600 rounded-lg py-2 px-3 text-sm text-zinc-200 placeholder-zinc-600 outline-none transition-all"
              placeholder={inputType === 'github' ? "https://github.com/..." : "/path/to/repo"}
              value={repoInput}
              onChange={e => setRepoInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') handleIndex(); }}
              type="text"
            />
          </div>
          <button
            onClick={handleIndex}
            disabled={isIndexing || !apiOnline}
            className={`w-full mt-2 bg-gradient-to-br from-zinc-100 to-zinc-400 text-zinc-950 font-bold py-2.5 rounded-lg flex items-center justify-center gap-2 transition-all duration-200 shadow-lg ${(isIndexing || !apiOnline) ? 'opacity-50 cursor-not-allowed' : 'hover:opacity-90 active:scale-95'}`}
          >
            {isIndexing ? <span className="material-symbols-outlined text-base animate-spin">progress_activity</span> : null}
            {isIndexing ? 'Indexing...' : 'Index Repository'}
          </button>
        </div>

        {/* INDEXED REPOS */}
        <div className="mt-6 flex flex-col gap-2 flex-1 min-h-0">
          <div className="flex items-center gap-2 px-1">
            <span className="material-symbols-outlined text-sm text-zinc-500">folder_open</span>
            <span className="text-[10px] font-bold text-zinc-500 tracking-widest uppercase">Indexed Repos</span>
            <span className="ml-auto text-[10px] font-mono text-zinc-600">{repos.length}</span>
          </div>
          <div className="flex flex-col gap-1 overflow-y-auto custom-scrollbar pr-1">
            {repos.length === 0 && (
              <p className="text-[11px] text-zinc-600 italic px-2 py-3">No repositories indexed yet.</p>
            )}
            {repos.map(repo => (
              <div
                key={repo.repo_id}
                onClick={() => setActiveRepoId(repo.repo_id)}
                className={`group rounded-lg px-3 py-2 flex items-center gap-2 cursor-pointer transition-all ${repo.repo_id === activeRepoId ? 'bg-zinc-800/50 text-zinc-100 shadow-inner' : 'text-zinc-400 hover:bg-zinc-800/30'}`}
              >
                <span className={`material-symbols-outlined text-sm ${statusColor[repo.status] || 'text-zinc-500'}`}>
                  {repo.status === 'indexing' ? 'progress_activity' : repo.status === 'error' ? 'error' : 'database'}
                </span>
                <div className="flex flex-col min-w-0 flex-1">
                  <span className="text-xs font-medium truncate">{repo.display_name}</span>
                  <span className={`text-[9px] font-mono truncate ${repo.status === 'error' ? 'text-red-500' : repo.status === 'indexing' ? 'text-amber-500' : 'text-zinc-600'}`}>
                    {repo.status === 'indexing'
                      ? (repo.progress || 'indexing...')
                      : repo.status === 'error'
                        ? (repo.error || 'failed')
                        : `${repo.chunks_created} chunks`}
                  </span>
                </div>
                <button
                  onClick={e => handleDelete(repo.repo_id, e)}
                  className="opacity-0 group-hover:opacity-100 text-zinc-500 hover:text-red-500 transition-all"
                  title="Delete repository"
                >
                  <span className="material-symbols-outlined text-sm">delete</span>
                </button>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-auto pt-4 flex flex-col gap-2">
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-zinc-900/40 ghost-border">
            <div className={`w-2 h-2 rounded-full ${apiOnline ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]' : 'bg-red-500'}`}></div>
            <span className={`text-xs font-medium ${apiOnline ? 'text-emerald-500' : 'text-red-500'}`}>API {apiOnline ? 'Online' : 'Offline'}</span>
            <span className="ml-auto material-symbols-outlined text-sm text-zinc-600">check_circle</span>
          </div>
        </div>
      </aside>

      {/* MAIN CHAT AREA */}
      <main className="flex-1 flex flex-col relative bg-surface">
        <header className="flex justify-between items-center h-14 px-6 w-full sticky top-0 z-40 bg-zinc-950/80 backdrop-blur-md border-b border-zinc-800/20">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-bold tracking-tighter bg-gradient-to-br from-zinc-100 to-zinc-500 bg-clip-text text-transparent">CodeLens</h1>
            {activeRepo && (
              <span className="text-xs font-mono text-zinc-500 flex items-center gap-1.5">
                <span className="material-symbols-outlined text-sm">database</span>
                {activeRepo.display_name}
              </span>
            )}
          </div>
        </header>

        <section className="flex-1 overflow-y-auto custom-scrollbar p-6 space-y-8 max-w-4xl mx-auto w-full pb-32">
          {messages.length === 0 ? (
            <div className="text-center py-12 space-y-4">
              <h2 className="text-4xl font-semibold tracking-tighter text-zinc-100">Architecting Insight.</h2>
              <p className="text-on-surface-variant max-w-md mx-auto">
                {activeRepo
                  ? `Querying ${activeRepo.display_name}. Ask anything about the codebase.`
                  : 'Index a repository, then select it from the sidebar to start semantic codebase analysis.'}
              </p>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div key={idx} className={`flex gap-4 animate-in fade-in duration-500 ${msg.role === 'user' ? 'justify-end slide-in-from-right-4' : 'justify-start slide-in-from-left-4'}`}>
                {msg.role === 'assistant' && (
                  <div className="w-8 h-8 rounded-lg bg-zinc-900 ghost-border flex items-center justify-center shrink-0">
                    <span className="material-symbols-outlined text-lg text-zinc-400">auto_awesome</span>
                  </div>
                )}

                <div className={`space-y-4 ${msg.role === 'user' ? 'max-w-[80%]' : 'flex-1'}`}>
                  <div className={`${msg.role === 'user' ? 'bg-zinc-800/50 px-4 py-3 rounded-2xl rounded-tr-sm ghost-border shadow-lg' : 'bg-zinc-950 px-5 py-4 rounded-2xl rounded-tl-sm ghost-border shadow-xl'}`}>
                    {msg.role === 'user' ? (
                      <p className="text-sm leading-relaxed text-zinc-100 font-medium">{msg.content}</p>
                    ) : (!msg.content && msg.streaming) ? (
                      <div className="flex items-center gap-2 py-1">
                        <div className="w-1.5 h-1.5 rounded-full bg-zinc-600 animate-bounce [animation-delay:-0.3s]"></div>
                        <div className="w-1.5 h-1.5 rounded-full bg-zinc-600 animate-bounce [animation-delay:-0.15s]"></div>
                        <div className="w-1.5 h-1.5 rounded-full bg-zinc-600 animate-bounce"></div>
                        <span className="text-xs text-zinc-500 ml-1 italic">Thinking...</span>
                      </div>
                    ) : (
                      <article className="prose prose-invert max-w-none text-sm text-on-surface leading-relaxed overflow-x-auto">
                        <ReactMarkdown
                          rehypePlugins={[rehypeRaw]}
                          components={{
                            code({node, inline, className, children, ...props}: any) {
                              const match = /language-(\w+)/.exec(className || '')
                              return !inline && match ? (
                                <SyntaxHighlighter
                                  style={vscDarkPlus as any}
                                  language={match[1]}
                                  PreTag="div"
                                  className="rounded-lg !bg-surface-container-low !p-4 !m-0 !text-xs !ghost-border"
                                  {...props}
                                >
                                  {String(children).replace(/\n$/, '')}
                                </SyntaxHighlighter>
                              ) : (
                                <code className="bg-zinc-800/60 px-1 py-0.5 rounded" {...props}>
                                  {children}
                                </code>
                              )
                            },
                            think({children}: any) {
                              return (
                                <details className="mb-4 bg-zinc-900/50 border border-zinc-800/50 rounded-lg overflow-hidden group">
                                  <summary className="flex items-center gap-2 px-4 py-2 bg-zinc-800/30 cursor-pointer text-xs font-semibold text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50 transition-all select-none list-none [&::-webkit-details-marker]:hidden">
                                    <span className="material-symbols-outlined text-[14px] group-open:-rotate-180 transition-transform duration-300">expand_more</span>
                                    <span className="material-symbols-outlined text-[14px]">psychology</span>
                                    AI Thought Process
                                  </summary>
                                  <div className="p-4 text-xs text-zinc-500 border-t border-zinc-800/50 italic bg-zinc-950/30 leading-relaxed font-mono">
                                    {children}
                                  </div>
                                </details>
                              )
                            }
                          } as any}
                        >
                          {msg.content}
                        </ReactMarkdown>
                      </article>
                    )}
                  </div>

                  {/* Sources Accordion */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="space-y-2 mt-4">
                      <div className="flex items-center gap-2 px-1">
                        <span className="material-symbols-outlined text-xs text-zinc-500">link</span>
                        <span className="text-[10px] font-bold text-zinc-500 tracking-widest uppercase">Verified Sources</span>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {msg.sources.map((src: any, i: number) => (
                          <div key={i} className="bg-surface-container-low p-3 rounded-xl ghost-border hover:bg-zinc-800/30 transition-all group">
                            <div className="flex items-center justify-between mb-2">
                              <div className="flex items-center gap-2 min-w-0">
                                <span className="material-symbols-outlined text-sm text-zinc-500">description</span>
                                <span className="text-[11px] font-mono text-zinc-300 truncate">{src.file_path}</span>
                              </div>
                              <span className="text-[10px] px-1.5 py-0.5 bg-emerald-500/10 text-emerald-500 rounded font-bold shrink-0 ml-2">{Math.round((src.similarity || 0) * 100)}% Match</span>
                            </div>
                            {(src.start_line || src.end_line) ? (
                              <p className="text-[9px] text-zinc-600 font-mono mb-1">lines {src.start_line}-{src.end_line}</p>
                            ) : null}
                            <div className="h-[1px] bg-zinc-800/50 w-full mb-2"></div>
                            <p className="text-[10px] text-zinc-500 line-clamp-2 italic">{src.content?.substring(0, 100)}...</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
        </section>

        <div className="absolute bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-surface via-surface to-transparent">
          <div className="max-w-4xl mx-auto">
            <div className="relative group">
              <form onSubmit={handleQuery} className="flex items-center gap-3 bg-surface-container-highest/80 backdrop-blur-xl px-4 py-3 rounded-2xl shadow-[0_20px_50px_rgba(0,0,0,0.5)] ghost-border focus-within:ring-1 ring-zinc-700 transition-all">
                <span className="material-symbols-outlined text-zinc-500 group-focus-within:text-zinc-300">search</span>
                <input
                  className="flex-1 bg-transparent border-none focus:ring-0 text-sm text-zinc-100 placeholder-zinc-600 outline-none"
                  placeholder={canChat ? "Ask anything about your codebase..." : "Select an indexed repository to start..."}
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                  disabled={!canChat || isQuerying}
                  type="text"
                />
                <button type="submit" disabled={!canChat || isQuerying || !query.trim()} className="w-8 h-8 rounded-lg bg-zinc-100 flex items-center justify-center hover:bg-zinc-200 active:scale-95 transition-all group/btn disabled:opacity-50">
                  <span className="material-symbols-outlined text-zinc-900 text-sm group-hover/btn:translate-x-0.5 transition-transform">send</span>
                </button>
              </form>
              {!activeRepo && (
                <div className="hidden absolute -top-8 left-0 right-0 text-center md:block">
                  <span className="text-[10px] font-bold text-zinc-600 tracking-tighter uppercase bg-zinc-950 px-3 py-1 rounded-full border border-zinc-800/50">
                    Index and select a repository to unlock chat
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      {/* UTILITY PANEL */}
      <aside className="w-72 border-l border-zinc-800/20 bg-surface-container-low hidden lg:flex flex-col p-4">
        <div className="space-y-6">
          <div className="space-y-4">
            <h3 className="text-[10px] font-bold text-zinc-500 tracking-widest uppercase">
              {activeRepo ? 'Active Repo' : 'Project Metrics'}
            </h3>
            {activeRepo ? (
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-xs text-zinc-400">Repository</span>
                  <span className="text-xs font-mono font-bold text-zinc-200 truncate ml-2">{activeRepo.display_name}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-xs text-zinc-400">Status</span>
                  <span className={`text-xs font-mono font-bold ${statusColor[activeRepo.status] || 'text-zinc-200'}`}>{activeRepo.status}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-xs text-zinc-400">Files Indexed</span>
                  <span className="text-xs font-mono font-bold text-zinc-200">{activeRepo.files_indexed}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-xs text-zinc-400">Chunks Created</span>
                  <span className="text-xs font-mono font-bold text-zinc-200">{activeRepo.chunks_created}</span>
                </div>
              </div>
            ) : (
              <p className="text-[11px] text-zinc-600 italic">No repository selected.</p>
            )}
          </div>
        </div>
      </aside>
    </div>
  );
}
