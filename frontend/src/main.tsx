import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertCircle,
  BarChart3,
  CalendarClock,
  CheckCircle2,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Clock3,
  Database,
  Filter,
  History,
  KeyRound,
  Layers,
  Loader2,
  LogOut,
  MessageSquareText,
  PlayCircle,
  RefreshCcw,
  Save,
  Search,
  Server,
  Settings,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Ticket,
  Trash2,
  UserPlus,
  Users
} from "lucide-react";
import "./styles.css";

type Health = {
  ok: boolean;
  error?: string;
  openai_configured?: boolean;
  chat_model?: string;
  embedding_model?: string;
  vectors?: number;
  dimension?: number;
  tickets?: number;
  ids?: number;
  faiss_dir?: string;
};

type Hit = {
  rank: number;
  score: number;
  id: number | string;
  thread_id?: number | string | null;
  title: string;
  created?: string | null;
  poster?: string | null;
  excerpt: string;
  body: string;
};

type AskResponse = {
  answer: string;
  hits: Hit[];
  model: string;
  embedding_model: string;
};

type ProblemGroup = {
  key: string;
  title: string;
  count: number;
  unique_ticket_count: number;
  priority: "Alta" | "Media" | "Bassa";
  priority_score: number;
  category: string;
  trend: string;
  recurring: boolean;
  first_seen?: string | null;
  last_seen?: string | null;
  sample_ticket_ids: Array<number | string>;
  sample_titles: string[];
  keywords: string[];
  latest_tickets: Array<{
    id: number | string;
    created?: string | null;
    title: string;
    poster?: string | null;
    excerpt: string;
  }>;
};

type RecentProblemsResponse = {
  since?: string | null;
  total_recent_tickets: number;
  groups: ProblemGroup[];
  priority_counts: Record<string, number>;
  recurring_count: number;
  ai_summary?: string | null;
  ai_error?: string | null;
  generated_at?: string | null;
  vector_count: number;
};

type DatabaseConfig = {
  ssh_host: string;
  ssh_port: number;
  ssh_user: string;
  ssh_password: string;
  db_host: string;
  db_port: number;
  db_user: string;
  db_password: string;
  db_name: string;
  query: string;
  batch_size: number;
};

type ConfigResponse = {
  config: DatabaseConfig;
  latest_local_ticket_date?: string | null;
  embedding_model_for_build: string;
};

type JobResponse = {
  id: string;
  type: string;
  status: "queued" | "running" | "completed" | "failed";
  step: string;
  message: string;
  total: number;
  current: number;
  processed: number;
  embedded: number;
  written: number;
  skipped: number;
  chunks: number;
  attempt: number;
  error?: string | null;
  started_at: string;
  finished_at?: string | null;
};

type UserRole = "admin" | "user";

type UserResponse = {
  id: number;
  username: string;
  role: UserRole;
  active: boolean | number;
  created_at?: string | null;
};

type Session = {
  token: string;
  user: UserResponse;
};

type ChatHistoryItem = {
  id: string;
  created_at: string;
  question: string;
  top_k: number;
  response: AskResponse;
};

const HITS_PAGE_SIZE = 4;
const SESSION_KEY = "ticket-ai-session";
const CHAT_HISTORY_KEY = "ticket-ai-chat-history";

const exampleQuestions = [
  "Un cliente segnala anomalie nel rinnovo dei CCNL: ci sono casi simili?",
  "Come e' stata risolta una richiesta di disattivazione utenza?",
  "Errore in fase di chiusura mensile: quali controlli erano stati fatti?"
];

const emptyConfig: DatabaseConfig = {
  ssh_host: "",
  ssh_port: 22,
  ssh_user: "",
  ssh_password: "",
  db_host: "127.0.0.1",
  db_port: 3306,
  db_user: "",
  db_password: "",
  db_name: "",
  query: "",
  batch_size: 150
};

function authHeaders(session: Session | null): HeadersInit {
  return session ? { Authorization: `Bearer ${session.token}` } : {};
}

function jsonAuthHeaders(session: Session | null): HeadersInit {
  return { "Content-Type": "application/json", ...authHeaders(session) };
}

function readJson<T>(key: string, fallback: T): T {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function writeJson(key: string, value: unknown) {
  window.localStorage.setItem(key, JSON.stringify(value));
}

function renderInlineMarkdown(text: string) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={index}>{part.slice(2, -2)}</strong>;
    }
    return <React.Fragment key={index}>{part}</React.Fragment>;
  });
}

function SummaryMarkdown({ text }: { text: string }) {
  const lines = text
    .replace(/\r\n/g, "\n")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  return (
    <div className="summary-markdown">
      {lines.map((line, index) => {
        if (line.startsWith("### ")) {
          return <h4 key={index}>{renderInlineMarkdown(line.slice(4))}</h4>;
        }
        if (line.startsWith("## ")) {
          return <h3 key={index}>{renderInlineMarkdown(line.slice(3))}</h3>;
        }

        const numbered = line.match(/^(\d+)\.\s+(.*)$/);
        if (numbered) {
          return (
            <div className="summary-numbered" key={index}>
              <span>{numbered[1]}</span>
              <p>{renderInlineMarkdown(numbered[2])}</p>
            </div>
          );
        }

        const bullet = line.match(/^[-*>]\s*(.*)$/);
        if (bullet) {
          const priority = bullet[1].match(/^\[(Alta|Media|Bassa)\]\s*(.*)$/i);
          return (
            <div className={`summary-bullet ${priority ? `priority-${priority[1].toLowerCase()}` : ""}`} key={index}>
              {priority ? <span className="priority-pill">{priority[1]}</span> : <span className="bullet-dot" />}
              <p>{renderInlineMarkdown(priority ? priority[2] : bullet[1])}</p>
            </div>
          );
        }

        return <p key={index}>{renderInlineMarkdown(line)}</p>;
      })}
    </div>
  );
}

function App() {
  const [session, setSession] = useState<Session | null>(() => readJson<Session | null>(SESSION_KEY, null));
  const [authChecking, setAuthChecking] = useState(Boolean(readJson<Session | null>(SESSION_KEY, null)));
  const [view, setView] = useState<"ask" | "analysis" | "config" | "users">("ask");
  const [analysisTab, setAnalysisTab] = useState<"priorities" | "recurring" | "summary">("priorities");
  const [health, setHealth] = useState<Health | null>(null);
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(8);
  const [askLoading, setAskLoading] = useState(false);
  const [askError, setAskError] = useState("");
  const [askCacheHit, setAskCacheHit] = useState(false);
  const [answer, setAnswer] = useState<AskResponse | null>(null);
  const [chatHistory, setChatHistory] = useState<ChatHistoryItem[]>(() => readJson<ChatHistoryItem[]>(CHAT_HISTORY_KEY, []));
  const [hitPage, setHitPage] = useState(1);
  const [days, setDays] = useState(30);
  const [analysis, setAnalysis] = useState<RecentProblemsResponse | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [analysisError, setAnalysisError] = useState("");
  const [analysisJob, setAnalysisJob] = useState<JobResponse | null>(null);
  const [priorityFilter, setPriorityFilter] = useState<"all" | "Alta" | "Media" | "Bassa" | "recurring">("all");
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());
  const [users, setUsers] = useState<UserResponse[]>([]);
  const [usersLoading, setUsersLoading] = useState(false);
  const [usersError, setUsersError] = useState("");
  const [usersMessage, setUsersMessage] = useState("");
  const [newUser, setNewUser] = useState({ username: "", password: "", role: "user" as UserRole });
  const [config, setConfig] = useState<DatabaseConfig>(emptyConfig);
  const [configLoading, setConfigLoading] = useState(false);
  const [configSaving, setConfigSaving] = useState(false);
  const [configMessage, setConfigMessage] = useState("");
  const [configError, setConfigError] = useState("");
  const [latestLocalDate, setLatestLocalDate] = useState<string | null>(null);
  const [rebuildFromDate, setRebuildFromDate] = useState("");
  const [job, setJob] = useState<JobResponse | null>(null);

  useEffect(() => {
    if (!session) {
      setAuthChecking(false);
      return;
    }
    fetch("/api/auth/me", { headers: authHeaders(session) })
      .then(async (response) => {
        if (!response.ok) throw new Error("Sessione scaduta");
        const user = await response.json();
        const nextSession = { ...session, user };
        setSession(nextSession);
        writeJson(SESSION_KEY, nextSession);
      })
      .catch(() => {
        setSession(null);
        window.localStorage.removeItem(SESSION_KEY);
      })
      .finally(() => setAuthChecking(false));
  }, []);

  useEffect(() => {
    if (!session) return;
    refreshHealth();
  }, [session?.token]);

  useEffect(() => {
    if (!session) return;
    if (view === "analysis" && !analysis && !analysisLoading) {
      loadAnalysis();
      if (session.user.role === "admin") loadLatestAnalysisJob();
    }
    if (view === "config" && session.user.role === "admin" && !configLoading && !config.ssh_host && !config.query) {
      loadConfig();
    } else if (view === "config") {
      loadLatestJob();
    }
    if (view === "users" && session.user.role === "admin" && !users.length && !usersLoading) {
      loadUsers();
    }
  }, [view, session?.token]);

  useEffect(() => {
    if (!job || !["queued", "running"].includes(job.status)) return;
    const timer = window.setInterval(async () => {
      try {
        const response = await fetch(`/api/index/jobs/${job.id}`, { headers: authHeaders(session) });
        const payload = await response.json();
        if (response.ok) {
          setJob(payload);
          if (payload.status === "completed") {
            refreshHealth();
            refreshLatestDate();
          }
        } else if (response.status === 404) {
          setJob({
            ...job,
            status: "failed",
            step: "Interrotto",
            message: "Il backend non trova piu' questo job.",
            error: payload.detail || "Job non trovato. Probabile riavvio backend durante l'operazione.",
            finished_at: new Date().toISOString()
          });
        }
      } catch {
        // Keep the current job snapshot; the next poll can recover.
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job?.id, job?.status]);

  useEffect(() => {
    if (!analysisJob || !["queued", "running"].includes(analysisJob.status)) return;
    const timer = window.setInterval(async () => {
      try {
        const response = await fetch(`/api/analysis/jobs/${analysisJob.id}`, { headers: authHeaders(session) });
        const payload = await response.json();
        if (response.ok) {
          setAnalysisJob(payload);
          if (payload.status === "completed") {
            loadAnalysis(days, true);
          }
        } else if (response.status === 404) {
          setAnalysisJob({
            ...analysisJob,
            status: "failed",
            step: "Interrotto",
            message: "Il backend non trova piu' questo job.",
            error: payload.detail || "Job non trovato. Probabile riavvio backend durante l'operazione.",
            finished_at: new Date().toISOString()
          });
        }
      } catch {
        // Non-blocking: the next poll can recover.
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [analysisJob?.id, analysisJob?.status, days]);

  const status = useMemo(() => {
    if (!health) return { label: "Connessione...", ok: false };
    if (!health.ok) return { label: "Backend non pronto", ok: false };
    if (!health.openai_configured) return { label: "OpenAI key mancante", ok: false };
    return { label: "Pronto", ok: true };
  }, [health]);

  const pagedHits = useMemo(() => {
    if (!answer) return [];
    const start = (hitPage - 1) * HITS_PAGE_SIZE;
    return answer.hits.slice(start, start + HITS_PAGE_SIZE);
  }, [answer, hitPage]);

  const hitPages = answer ? Math.max(1, Math.ceil(answer.hits.length / HITS_PAGE_SIZE)) : 1;
  const filteredProblemGroups = useMemo(() => {
    if (!analysis) return [];
    if (priorityFilter === "all") return analysis.groups;
    if (priorityFilter === "recurring") return analysis.groups.filter((group) => group.recurring);
    return analysis.groups.filter((group) => group.priority === priorityFilter);
  }, [analysis, priorityFilter]);
  const recurringProblemGroups = useMemo(() => {
    if (!analysis) return [];
    return analysis.groups.filter((group) => group.recurring);
  }, [analysis]);

  function toggleProblemGroup(key: string) {
    setExpandedGroups((current) => {
      const next = new Set(current);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  }

  async function submitQuestion(event?: React.FormEvent) {
    event?.preventDefault();
    if (!question.trim()) return;
    const normalizedQuestion = question.trim();
    const cached = chatHistory.find((item) => item.question.trim().toLowerCase() === normalizedQuestion.toLowerCase() && item.top_k === topK);
    if (cached) {
      setAskCacheHit(true);
      setAskError("");
      setAnswer(cached.response);
      setHitPage(1);
      return;
    }
    setAskCacheHit(false);
    setAskLoading(true);
    setAskError("");
    setAnswer(null);
    setHitPage(1);
    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: jsonAuthHeaders(session),
        body: JSON.stringify({ question: normalizedQuestion, top_k: topK })
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Errore durante la ricerca");
      setAnswer(payload);
      const nextHistory = [
        {
          id: `${Date.now()}`,
          created_at: new Date().toISOString(),
          question: normalizedQuestion,
          top_k: topK,
          response: payload
        },
        ...chatHistory.filter((item) => item.question.trim().toLowerCase() !== normalizedQuestion.toLowerCase()).slice(0, 24)
      ];
      setChatHistory(nextHistory);
      writeJson(CHAT_HISTORY_KEY, nextHistory);
    } catch (error) {
      setAskError(error instanceof Error ? error.message : String(error));
    } finally {
      setAskLoading(false);
    }
  }

  async function refreshHealth() {
    try {
      const response = await fetch("/api/health");
      setHealth(await response.json());
    } catch (error) {
      setHealth({ ok: false, error: String(error) });
    }
  }

  async function loadAnalysis(nextDays = days, force = false) {
    setAnalysisLoading(true);
    setAnalysisError("");
    try {
      const response = await fetch(`/api/analysis/recent-problems?days=${nextDays}&limit=24`, {
        headers: authHeaders(session)
      });
      const text = await response.text();
      let payload: RecentProblemsResponse | { detail?: string };
      try {
        payload = text ? JSON.parse(text) : {};
      } catch {
        throw new Error(
          response.ok
            ? "Risposta non valida dal server"
            : `Il server ha risposto ${response.status}: ${response.statusText || "errore proxy"}`
        );
      }
      if (!response.ok) throw new Error(payload.detail || "Errore durante l'analisi");
      setAnalysis(payload as RecentProblemsResponse);
    } catch (error) {
      setAnalysisError(error instanceof Error ? error.message : String(error));
    } finally {
      setAnalysisLoading(false);
    }
  }

  async function loadConfig() {
    setConfigLoading(true);
    setConfigError("");
    try {
      const response = await fetch("/api/config", { headers: authHeaders(session) });
      const payload: ConfigResponse = await response.json();
      if (!response.ok) throw new Error((payload as any).detail || "Errore caricamento configurazione");
      setConfig(payload.config);
      setLatestLocalDate(payload.latest_local_ticket_date ?? null);
      setRebuildFromDate(payload.latest_local_ticket_date?.slice(0, 10) ?? "");
      await loadLatestJob();
    } catch (error) {
      setConfigError(error instanceof Error ? error.message : String(error));
    } finally {
      setConfigLoading(false);
    }
  }

  async function loadLatestJob() {
    try {
      const response = await fetch("/api/index/jobs/latest", { headers: authHeaders(session) });
      const payload = await response.json();
      if (response.ok && payload && ["queued", "running", "failed"].includes(payload.status)) {
        setJob(payload);
      }
    } catch {
      // Non-blocking: the configuration page can still be used.
    }
  }

  async function loadLatestAnalysisJob() {
    try {
      const response = await fetch("/api/analysis/jobs/latest", { headers: authHeaders(session) });
      const payload = await response.json();
      if (response.ok && payload && ["queued", "running", "failed"].includes(payload.status)) {
        setAnalysisJob(payload);
      }
    } catch {
      // Non-blocking: the analysis page can still show the last saved dataset.
    }
  }

  async function startAnalysisRefresh() {
    setAnalysisError("");
    try {
      const response = await fetch("/api/analysis/recent-problems/run", {
        method: "POST",
        headers: authHeaders(session)
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Errore avvio analisi");
      setAnalysisJob(payload);
    } catch (error) {
      setAnalysisError(error instanceof Error ? error.message : String(error));
    }
  }

  async function refreshLatestDate() {
    try {
      const response = await fetch("/api/index/latest-date", { headers: authHeaders(session) });
      const payload = await response.json();
      if (response.ok) setLatestLocalDate(payload.latest_local_ticket_date ?? null);
    } catch {
      // Non-blocking UI refresh.
    }
  }

  async function saveConfig(): Promise<boolean> {
    setConfigSaving(true);
    setConfigError("");
    setConfigMessage("");
    try {
      const response = await fetch("/api/config", {
        method: "POST",
        headers: jsonAuthHeaders(session),
        body: JSON.stringify({ config })
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Errore salvataggio configurazione");
      setConfig(payload.config);
      setConfigMessage("Configurazione salvata.");
      return true;
    } catch (error) {
      setConfigError(error instanceof Error ? error.message : String(error));
      return false;
    } finally {
      setConfigSaving(false);
    }
  }

  async function startRebuild() {
    setConfigError("");
    setConfigMessage("");
    if (!rebuildFromDate) {
      setConfigError("Indica una data di partenza per ricostruire il FAISS.");
      return;
    }
    if (!(await saveConfig())) return;
    try {
      const response = await fetch("/api/index/rebuild", {
        method: "POST",
        headers: jsonAuthHeaders(session),
        body: JSON.stringify({ from_date: rebuildFromDate })
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Errore avvio rebuild");
      setJob(payload);
    } catch (error) {
      setConfigError(error instanceof Error ? error.message : String(error));
    }
  }

  async function startResumeRebuild() {
    setConfigError("");
    setConfigMessage("");
    if (!rebuildFromDate) {
      setConfigError("Indica la stessa data usata per la ricostruzione da riprendere.");
      return;
    }
    if (!(await saveConfig())) return;
    try {
      const response = await fetch("/api/index/rebuild/resume", {
        method: "POST",
        headers: jsonAuthHeaders(session),
        body: JSON.stringify({ from_date: rebuildFromDate })
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Errore ripresa rebuild");
      setJob(payload);
    } catch (error) {
      setConfigError(error instanceof Error ? error.message : String(error));
    }
  }

  async function startAppend() {
    setConfigError("");
    setConfigMessage("");
    if (!(await saveConfig())) return;
    try {
      const response = await fetch("/api/index/append", { method: "POST", headers: authHeaders(session) });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Errore avvio append");
      setJob(payload);
    } catch (error) {
      setConfigError(error instanceof Error ? error.message : String(error));
    }
  }

  function handleLogin(nextSession: Session) {
    setSession(nextSession);
    writeJson(SESSION_KEY, nextSession);
    setView("ask");
    refreshHealth();
  }

  async function logout() {
    try {
      await fetch("/api/auth/logout", { method: "POST", headers: authHeaders(session) });
    } catch {
      // Local logout still wins.
    }
    setSession(null);
    setHealth(null);
    setAnswer(null);
    setAnalysis(null);
    window.localStorage.removeItem(SESSION_KEY);
  }

  async function loadUsers() {
    setUsersLoading(true);
    setUsersError("");
    try {
      const response = await fetch("/api/users", { headers: authHeaders(session) });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Errore caricamento utenti");
      setUsers(payload);
    } catch (error) {
      setUsersError(error instanceof Error ? error.message : String(error));
    } finally {
      setUsersLoading(false);
    }
  }

  async function createUser(event: React.FormEvent) {
    event.preventDefault();
    setUsersError("");
    setUsersMessage("");
    try {
      const response = await fetch("/api/users", {
        method: "POST",
        headers: jsonAuthHeaders(session),
        body: JSON.stringify(newUser)
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Errore creazione utente");
      setUsers((current) => [...current, payload]);
      setNewUser({ username: "", password: "", role: "user" });
      setUsersMessage("Utente creato.");
    } catch (error) {
      setUsersError(error instanceof Error ? error.message : String(error));
    }
  }

  function loadHistoryItem(item: ChatHistoryItem) {
    setQuestion(item.question);
    setTopK(item.top_k);
    setAnswer(item.response);
    setAskCacheHit(true);
    setHitPage(1);
  }

  function clearChatHistory() {
    setChatHistory([]);
    window.localStorage.removeItem(CHAT_HISTORY_KEY);
  }

  function updateConfig<K extends keyof DatabaseConfig>(key: K, value: DatabaseConfig[K]) {
    setConfig((current) => ({ ...current, [key]: value }));
  }

  if (authChecking) {
    return <LoadingPanel label="Verifico la sessione..." />;
  }

  if (!session) {
    return <LoginPage onLogin={handleLogin} />;
  }

  return (
    <main className="app-shell">
      <section className="hero-panel">
        <div className="hero-copy">
          <span className="eyebrow">
            <Sparkles size={15} />
            Ticket intelligence
          </span>
          <h1>Knowledge Assistant</h1>
          <p>Trova casi simili, soluzioni gia' viste e problemi ricorrenti senza scavare a mano nei ticket.</p>
        </div>
        <div className="hero-actions">
          <div className={`status ${status.ok ? "status-ok" : "status-warn"}`}>
            {status.ok ? <CheckCircle2 size={18} /> : <AlertCircle size={18} />}
            <span>{status.label}</span>
          </div>
          <span className="soft-chip">
            <ShieldCheck size={15} />
            {session.user.username} · {session.user.role}
          </span>
          <button className="icon-button" onClick={logout} aria-label="Logout">
            <LogOut size={18} />
          </button>
        </div>
      </section>

      <section className="system-strip">
        <Metric icon={<Server size={18} />} label="Ticket" value={health?.tickets?.toLocaleString("it-IT") ?? "-"} />
        <Metric icon={<Activity size={18} />} label="Vettori" value={health?.vectors?.toLocaleString("it-IT") ?? "-"} />
        <Metric icon={<MessageSquareText size={18} />} label="Embedding" value={health?.embedding_model ?? "-"} />
        <Metric icon={<Layers size={18} />} label="Dimensioni" value={health?.dimension?.toString() ?? "-"} />
      </section>

      <nav className="tabs" aria-label="Sezioni">
        <button className={view === "ask" ? "active" : ""} onClick={() => setView("ask")}>
          <Search size={18} />
          Ricerca assistita
        </button>
        <button className={view === "analysis" ? "active" : ""} onClick={() => setView("analysis")}>
          <BarChart3 size={18} />
          Problemi noti
        </button>
        {session.user.role === "admin" ? (
          <>
            <button className={view === "config" ? "active" : ""} onClick={() => setView("config")}>
              <Settings size={18} />
              Configurazione
            </button>
            <button className={view === "users" ? "active" : ""} onClick={() => setView("users")}>
              <Users size={18} />
              Utenti
            </button>
          </>
        ) : null}
      </nav>

      {view === "ask" ? (
        <section className="workspace fade-in">
          <form className="query-panel elevated-panel" onSubmit={submitQuestion}>
            <div className="panel-heading">
              <div>
                <h2>Descrivi il problema</h2>
                <p>Scrivi come lo racconterebbe un cliente o un operatore.</p>
              </div>
              <span className="soft-chip">{topK} ticket</span>
            </div>
            <textarea
              id="question"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Esempio: Un cliente non riesce a chiudere la questione CCNL da rinnovare..."
            />
            <div className="controls-row">
              <label className="range-control">
                <span>Profondita ricerca</span>
                <input min={3} max={20} type="range" value={topK} onChange={(event) => setTopK(Number(event.target.value))} />
              </label>
              <button className="primary-button" type="submit" disabled={askLoading || !question.trim()}>
                {askLoading ? <Loader2 className="spin" size={18} /> : <Search size={18} />}
                Cerca casi simili
              </button>
            </div>
            <div className="examples">
              {exampleQuestions.map((item) => (
                <button key={item} type="button" onClick={() => setQuestion(item)}>
                  {item}
                </button>
              ))}
            </div>
          </form>

          {askError ? <ErrorBlock message={askError} /> : null}
          {askLoading ? <LoadingPanel label="Sto cercando ticket simili e preparando la risposta..." /> : null}
          {askCacheHit ? <div className="success-block">Risposta caricata dalla cronologia locale.</div> : null}

          {chatHistory.length ? (
            <section className="history-panel elevated-panel">
              <div className="list-header">
                <div className="panel-title">
                  <History size={19} />
                  <div>
                    <h2>Cronologia domande</h2>
                    <p>{chatHistory.length} risposte salvate nel browser</p>
                  </div>
                </div>
                <button className="icon-button" onClick={clearChatHistory} aria-label="Svuota cronologia">
                  <Trash2 size={18} />
                </button>
              </div>
              <div className="history-list">
                {chatHistory.slice(0, 6).map((item) => (
                  <button key={item.id} type="button" onClick={() => loadHistoryItem(item)}>
                    <span>{new Date(item.created_at).toLocaleString("it-IT")}</span>
                    <strong>{item.question}</strong>
                  </button>
                ))}
              </div>
            </section>
          ) : null}

          {answer ? (
            <div className="search-results">
              <section className="answer-panel elevated-panel">
                <div className="panel-title">
                  <MessageSquareText size={19} />
                  <h2>Risposta suggerita</h2>
                </div>
                <p className="answer-text">{answer.answer}</p>
                <div className="model-line">
                  Modello: {answer.model} - Embedding: {answer.embedding_model}
                </div>
              </section>

              <section className="hits-panel elevated-panel">
                <div className="list-header">
                  <div className="panel-title">
                    <Ticket size={19} />
                    <div>
                      <h2>Ticket piu' vicini</h2>
                      <p>{answer.hits.length} risultati</p>
                    </div>
                  </div>
                  <div className="pagination">
                    <button
                      className="icon-button"
                      onClick={() => setHitPage((page) => Math.max(1, page - 1))}
                      disabled={hitPage <= 1}
                      aria-label="Pagina precedente"
                    >
                      <ChevronLeft size={18} />
                    </button>
                    <span>
                      {hitPage} / {hitPages}
                    </span>
                    <button
                      className="icon-button"
                      onClick={() => setHitPage((page) => Math.min(hitPages, page + 1))}
                      disabled={hitPage >= hitPages}
                      aria-label="Pagina successiva"
                    >
                      <ChevronRight size={18} />
                    </button>
                  </div>
                </div>
                <div className="hit-grid">
                  {pagedHits.map((hit) => (
                    <article className="hit-card interactive-card" key={`${hit.id}-${hit.rank}`}>
                      <div className="hit-topline">
                        <strong>Ticket #{hit.id}</strong>
                        <span>{hit.score.toFixed(4)}</span>
                      </div>
                      <h3>{hit.title}</h3>
                      <p>{hit.excerpt}</p>
                      <footer>
                        <span>{hit.created || "data n/d"}</span>
                        <span>{hit.poster || "autore n/d"}</span>
                      </footer>
                    </article>
                  ))}
                </div>
              </section>
            </div>
          ) : null}
        </section>
      ) : view === "analysis" ? (
        <section className="workspace fade-in">
          <div className="analysis-toolbar elevated-panel">
            <div className="toolbar-copy">
              <h2>Problemi noti</h2>
              <p>Priorita operative, ricorrenze e gruppi ticket recenti.</p>
            </div>
            <div className="toolbar-actions">
              <label className="select-control">
                <CalendarClock size={18} />
                <span>Periodo</span>
                <select
                  value={days}
                  onChange={(event) => {
                    const value = Number(event.target.value);
                    setDays(value);
                    loadAnalysis(value);
                  }}
                >
                  <option value={7}>Ultimi 7 giorni</option>
                  <option value={30}>Ultimi 30 giorni</option>
                  <option value={90}>Ultimi 90 giorni</option>
                  <option value={180}>Ultimi 180 giorni</option>
                </select>
              </label>
              {analysis?.generated_at ? <span className="soft-chip">DB: {analysis.generated_at}</span> : <span className="soft-chip">DB condiviso</span>}
              {session.user.role === "admin" ? (
                <button
                  className="primary-action ghost-action"
                  onClick={startAnalysisRefresh}
                  disabled={analysisJob?.status === "queued" || analysisJob?.status === "running"}
                >
                  {analysisJob?.status === "queued" || analysisJob?.status === "running" ? (
                    <Loader2 className="spin" size={18} />
                  ) : (
                    <PlayCircle size={18} />
                  )}
                  Rigenera 4 periodi
                </button>
              ) : null}
              <button className="icon-button" onClick={() => loadAnalysis(days, true)} disabled={analysisLoading} aria-label="Aggiorna analisi">
                {analysisLoading ? <Loader2 className="spin" size={18} /> : <RefreshCcw size={18} />}
              </button>
            </div>
          </div>

          {analysisError ? <ErrorBlock message={analysisError} /> : null}
          {analysisJob ? <JobProgress job={analysisJob} /> : null}
          {analysisLoading && !analysis ? <AnalysisSkeleton /> : null}

          {analysis ? (
            <>
              <section className="analysis-overview">
                <Metric
                  icon={<Ticket size={18} />}
                  label="Ticket nel periodo"
                  value={analysis.total_recent_tickets.toLocaleString("it-IT")}
                />
                <Metric
                  icon={<ShieldAlert size={18} />}
                  label="Priorita alta"
                  value={(analysis.priority_counts?.Alta ?? 0).toLocaleString("it-IT")}
                />
                <Metric
                  icon={<BarChart3 size={18} />}
                  label="Ricorrenti"
                  value={analysis.recurring_count.toLocaleString("it-IT")}
                />
                <Metric icon={<Clock3 size={18} />} label="Dal" value={analysis.since?.slice(0, 10) ?? "-"} />
                <Metric icon={<Database size={18} />} label="Generata" value={analysis.generated_at?.slice(0, 16) ?? "-"} />
              </section>

              <nav className="subtabs elevated-panel" aria-label="Vista problemi noti">
                <button className={analysisTab === "priorities" ? "active" : ""} onClick={() => setAnalysisTab("priorities")}>
                  <ShieldAlert size={18} />
                  Priorita
                </button>
                <button className={analysisTab === "recurring" ? "active" : ""} onClick={() => setAnalysisTab("recurring")}>
                  <BarChart3 size={18} />
                  Ricorrenti
                </button>
                <button className={analysisTab === "summary" ? "active" : ""} onClick={() => setAnalysisTab("summary")}>
                  <Sparkles size={18} />
                  Sintesi
                </button>
              </nav>

              {analysisTab === "priorities" ? (
              <section className="known-problems-panel elevated-panel">
                <div className="list-header">
                  <div className="panel-title">
                    <ShieldAlert size={19} />
                    <div>
                      <h2>Priorita e problemi ricorrenti</h2>
                      <p>
                        {filteredProblemGroups.length.toLocaleString("it-IT")} gruppi su {analysis.groups.length.toLocaleString("it-IT")}
                      </p>
                    </div>
                  </div>
                  <label className="select-control compact-select">
                    <Filter size={18} />
                    <span>Vista</span>
                    <select value={priorityFilter} onChange={(event) => setPriorityFilter(event.target.value as typeof priorityFilter)}>
                      <option value="all">Tutte</option>
                      <option value="Alta">Alta priorita</option>
                      <option value="Media">Media priorita</option>
                      <option value="Bassa">Bassa priorita</option>
                      <option value="recurring">Solo ricorrenti</option>
                    </select>
                  </label>
                </div>

                <div className="priority-lanes">
                  {(["Alta", "Media", "Bassa"] as const).map((priority) => (
                    <button
                      className={`priority-filter priority-${priority.toLowerCase()} ${priorityFilter === priority ? "active" : ""}`}
                      key={priority}
                      type="button"
                      onClick={() => setPriorityFilter(priority)}
                    >
                      <span>{priority}</span>
                      <strong>{analysis.priority_counts?.[priority] ?? 0}</strong>
                    </button>
                  ))}
                  <button
                    className={`priority-filter recurring-filter ${priorityFilter === "recurring" ? "active" : ""}`}
                    type="button"
                    onClick={() => setPriorityFilter("recurring")}
                  >
                    <span>Ricorrenti</span>
                    <strong>{analysis.recurring_count}</strong>
                  </button>
                </div>

                <div className="known-problem-list">
                  {filteredProblemGroups.map((group, index) => (
                    <ProblemGroupCard
                      expanded={expandedGroups.has(group.key)}
                      group={group}
                      index={index}
                      key={group.key}
                      onToggle={() => toggleProblemGroup(group.key)}
                    />
                  ))}
                </div>
              </section>
              ) : null}

              {analysisTab === "recurring" ? (
              <section className="known-problems-panel elevated-panel">
                <div className="list-header">
                  <div className="panel-title">
                    <BarChart3 size={19} />
                    <div>
                      <h2>Problemi ricorrenti</h2>
                      <p>{recurringProblemGroups.length.toLocaleString("it-IT")} gruppi con piu' ticket collegati</p>
                    </div>
                  </div>
                </div>
                <div className="known-problem-list">
                  {recurringProblemGroups.map((group, index) => (
                    <ProblemGroupCard
                      expanded={expandedGroups.has(group.key)}
                      group={group}
                      index={index}
                      key={group.key}
                      onToggle={() => toggleProblemGroup(group.key)}
                    />
                  ))}
                </div>
              </section>
              ) : null}

              {analysisTab === "summary" ? (
              <section className="answer-panel summary-panel elevated-panel">
                <div className="panel-title spaced-title">
                  <div>
                    <div className="title-with-icon">
                      <Sparkles size={19} />
                      <h2>Sintesi operativa</h2>
                    </div>
                    <p>Report generato dai ticket recenti e dai gruppi ricorrenti.</p>
                  </div>
                  {analysisLoading ? <span className="soft-chip loading-chip">Aggiorno...</span> : <span className="soft-chip">AI summary</span>}
                </div>
                {analysis.ai_summary ? (
                  <SummaryMarkdown text={analysis.ai_summary} />
                ) : (
                  <p className="answer-text">
                    Trovati {analysis.total_recent_tickets.toLocaleString("it-IT")} ticket dal{" "}
                    {analysis.since || "periodo disponibile"}.
                  </p>
                )}
                {analysis.ai_error ? <p className="model-line">Sintesi IA non disponibile: {analysis.ai_error}</p> : null}
              </section>
              ) : null}
            </>
          ) : analysisLoading ? null : (
            <div className="empty-state">Nessun dato caricato.</div>
          )}
        </section>
      ) : view === "config" && session.user.role === "admin" ? (
        <section className="workspace fade-in">
          <div className="analysis-toolbar elevated-panel">
            <div className="toolbar-copy">
              <h2>Configurazione indice</h2>
              <p>Connessione SSH/DB, query sorgente e operazioni di rebuild o append del FAISS.</p>
            </div>
            <div className="toolbar-actions">
              <span className="soft-chip">Build embedding: text-embedding-3-large</span>
              <button className="icon-button" onClick={loadConfig} disabled={configLoading} aria-label="Ricarica configurazione">
                {configLoading ? <Loader2 className="spin" size={18} /> : <RefreshCcw size={18} />}
              </button>
            </div>
          </div>

          {configError ? <ErrorBlock message={configError} /> : null}
          {configMessage ? <div className="success-block">{configMessage}</div> : null}

          <section className="config-grid">
            <div className="config-panel elevated-panel">
              <div className="panel-title">
                <Server size={19} />
                <h2>SSH</h2>
              </div>
              <div className="form-grid">
                <Field label="Host SSH" value={config.ssh_host} onChange={(value) => updateConfig("ssh_host", value)} />
                <Field
                  label="Porta SSH"
                  type="number"
                  value={config.ssh_port}
                  onChange={(value) => updateConfig("ssh_port", Number(value))}
                />
                <Field label="Utente SSH" value={config.ssh_user} onChange={(value) => updateConfig("ssh_user", value)} />
                <Field
                  label="Password SSH"
                  type="password"
                  value={config.ssh_password}
                  onChange={(value) => updateConfig("ssh_password", value)}
                />
              </div>
            </div>

            <div className="config-panel elevated-panel">
              <div className="panel-title">
                <Database size={19} />
                <h2>Database</h2>
              </div>
              <div className="form-grid">
                <Field label="Host DB remoto" value={config.db_host} onChange={(value) => updateConfig("db_host", value)} />
                <Field
                  label="Porta DB"
                  type="number"
                  value={config.db_port}
                  onChange={(value) => updateConfig("db_port", Number(value))}
                />
                <Field label="Utente DB" value={config.db_user} onChange={(value) => updateConfig("db_user", value)} />
                <Field
                  label="Password DB"
                  type="password"
                  value={config.db_password}
                  onChange={(value) => updateConfig("db_password", value)}
                />
                <Field label="Nome database" value={config.db_name} onChange={(value) => updateConfig("db_name", value)} />
                <Field
                  label="Batch embedding"
                  type="number"
                  value={config.batch_size}
                  onChange={(value) => updateConfig("batch_size", Number(value))}
                />
              </div>
            </div>
          </section>

          <section className="config-panel elevated-panel">
            <div className="panel-title">
              <MessageSquareText size={19} />
              <div>
                <h2>Query ticket</h2>
                <p>La query deve restituire: id, thread_id, staff_id, user_id, poster, created, title, body.</p>
              </div>
            </div>
            <textarea
              className="query-editor"
              value={config.query}
              onChange={(event) => updateConfig("query", event.target.value)}
              spellCheck={false}
            />
            <div className="controls-row">
              <button className="secondary-button" onClick={saveConfig} disabled={configSaving || configLoading}>
                {configSaving ? <Loader2 className="spin" size={18} /> : <Save size={18} />}
                Salva configurazione
              </button>
            </div>
          </section>

          <section className="config-actions elevated-panel">
            <div>
              <div className="panel-title">
                <PlayCircle size={19} />
                <h2>Operazioni indice</h2>
              </div>
              <p className="muted-line">
                Ultimo ticket locale: <strong>{latestLocalDate || "nessun indice locale"}</strong>
              </p>
            </div>
            <div className="operation-row">
              <label className="field">
                <span>Ricostruisci da data</span>
                <input type="date" value={rebuildFromDate} onChange={(event) => setRebuildFromDate(event.target.value)} />
              </label>
              <button className="danger-button" onClick={startRebuild} disabled={job?.status === "running" || job?.status === "queued"}>
                <RefreshCcw size={18} />
                Rifai FAISS da zero
              </button>
              <button className="secondary-button" onClick={startResumeRebuild} disabled={job?.status === "running" || job?.status === "queued"}>
                <PlayCircle size={18} />
                Riprendi rebuild
              </button>
              <button className="primary-button" onClick={startAppend} disabled={job?.status === "running" || job?.status === "queued"}>
                <PlayCircle size={18} />
                Appendi fino ad oggi
              </button>
            </div>
          </section>

          {job ? <JobProgress job={job} /> : null}
        </section>
      ) : view === "users" && session.user.role === "admin" ? (
        <section className="workspace fade-in">
          <div className="analysis-toolbar elevated-panel">
            <div className="toolbar-copy">
              <h2>Utenti</h2>
              <p>Gestisci accessi e ruoli dell'assistente.</p>
            </div>
            <div className="toolbar-actions">
              <button className="icon-button" onClick={loadUsers} disabled={usersLoading} aria-label="Ricarica utenti">
                {usersLoading ? <Loader2 className="spin" size={18} /> : <RefreshCcw size={18} />}
              </button>
            </div>
          </div>

          {usersError ? <ErrorBlock message={usersError} /> : null}
          {usersMessage ? <div className="success-block">{usersMessage}</div> : null}

          <section className="user-admin-grid">
            <form className="config-panel elevated-panel" onSubmit={createUser}>
              <div className="panel-title">
                <UserPlus size={19} />
                <h2>Crea utente</h2>
              </div>
              <div className="form-grid">
                <Field label="Username" value={newUser.username} onChange={(value) => setNewUser((current) => ({ ...current, username: value }))} />
                <Field
                  label="Password"
                  type="password"
                  value={newUser.password}
                  onChange={(value) => setNewUser((current) => ({ ...current, password: value }))}
                />
                <label className="field">
                  <span>Ruolo</span>
                  <select value={newUser.role} onChange={(event) => setNewUser((current) => ({ ...current, role: event.target.value as UserRole }))}>
                    <option value="user">Normal user</option>
                    <option value="admin">Admin</option>
                  </select>
                </label>
              </div>
              <div className="controls-row">
                <button className="primary-button" type="submit">
                  <UserPlus size={18} />
                  Crea utente
                </button>
              </div>
            </form>

            <section className="config-panel elevated-panel">
              <div className="panel-title">
                <Users size={19} />
                <h2>Utenti attivi</h2>
              </div>
              <div className="user-list">
                {users.map((user) => (
                  <article className="user-row" key={user.id}>
                    <div>
                      <strong>{user.username}</strong>
                      <span>{user.created_at || "creato"}</span>
                    </div>
                    <span className={`soft-chip role-${user.role}`}>{user.role}</span>
                  </article>
                ))}
              </div>
            </section>
          </section>
        </section>
      ) : (
        <div className="empty-state">Sezione non disponibile per il tuo ruolo.</div>
      )}
    </main>
  );
}

function LoginPage({ onLogin }: { onLogin: (session: Session) => void }) {
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const response = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password })
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Login non riuscito");
      onLogin(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="login-shell">
      <section className="login-visual">
        <img className="brand-logo" src="/centro-paghe-logo.png" alt="Gruppo Centro Paghe" />
        <h1>Knowledge Assistant</h1>
        <p>Accedi per cercare casi simili, leggere problemi noti e gestire l'indice in base al tuo ruolo.</p>
      </section>
      <form className="login-card elevated-panel" onSubmit={submit}>
        <div className="panel-title">
          <KeyRound size={20} />
          <div>
            <h2>Accesso</h2>
            <p>Admin seed: username <strong>admin</strong>, password <strong>admin</strong>.</p>
          </div>
        </div>
        <Field label="Username" value={username} onChange={setUsername} />
        <Field label="Password" type="password" value={password} onChange={setPassword} />
        {error ? <ErrorBlock message={error} /> : null}
        <button className="primary-button" type="submit" disabled={loading || !username || !password}>
          {loading ? <Loader2 className="spin" size={18} /> : <KeyRound size={18} />}
          Entra
        </button>
      </form>
    </main>
  );
}

function Field({
  label,
  value,
  onChange,
  type = "text"
}: {
  label: string;
  value: string | number;
  onChange: (value: string) => void;
  type?: string;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <input type={type} value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function ProblemGroupCard({
  group,
  index,
  expanded,
  onToggle
}: {
  group: ProblemGroup;
  index: number;
  expanded: boolean;
  onToggle: () => void;
}) {
  const priorityClass = `priority-${group.priority.toLowerCase()}`;
  const visibleTickets = expanded ? group.latest_tickets : group.latest_tickets.slice(0, 2);

  return (
    <article className={`known-problem-card ${priorityClass} ${expanded ? "expanded" : ""}`}>
      <button className="known-problem-main" type="button" onClick={onToggle} aria-expanded={expanded}>
        <div className="problem-rank">{index + 1}</div>
        <div className="problem-summary">
          <div className="problem-title-row">
            <h3>{group.title}</h3>
            <span className={`priority-pill ${priorityClass}`}>{group.priority}</span>
          </div>
          <div className="problem-meta">
            <span>{group.category}</span>
            <span>{group.count.toLocaleString("it-IT")} ticket</span>
            <span>{group.trend}</span>
            <span>{group.recurring ? "Ricorrente" : "Singolo"}</span>
          </div>
          <div className="keyword-row">
            {group.keywords.slice(0, 5).map((keyword) => (
              <span key={keyword}>{keyword}</span>
            ))}
          </div>
        </div>
        <div className="problem-side">
          <strong>{Math.round(group.priority_score)}</strong>
          <span>score</span>
          <ChevronDown className={expanded ? "rotate" : ""} size={20} />
        </div>
      </button>

      <div className="problem-details">
        <div className="detail-grid">
          <div>
            <span className="detail-label">Prima segnalazione</span>
            <strong>{group.first_seen || "-"}</strong>
          </div>
          <div>
            <span className="detail-label">Ultima segnalazione</span>
            <strong>{group.last_seen || "-"}</strong>
          </div>
          <div>
            <span className="detail-label">Ticket citati</span>
            <strong>{group.sample_ticket_ids.slice(0, 8).join(", ") || "-"}</strong>
          </div>
        </div>

        <div className="ticket-snippet-list">
          {visibleTickets.map((ticket) => (
            <div className="ticket-snippet" key={`${group.key}-${ticket.id}`}>
              <div>
                <strong>#{ticket.id}</strong>
                <span>{ticket.created || "data n/d"}</span>
              </div>
              <h4>{ticket.title}</h4>
              <p>{ticket.excerpt}</p>
            </div>
          ))}
        </div>
      </div>
    </article>
  );
}

function JobProgress({ job }: { job: JobResponse }) {
  const progress = job.total > 0 ? Math.min(100, Math.round((job.current / job.total) * 100)) : 0;
  const isAnalysisJob = job.type.startsWith("analysis");
  const steps = isAnalysisJob
    ? ["Preparazione analisi", "Analisi 7 giorni", "Analisi 30 giorni", "Analisi 90 giorni", "Analisi 180 giorni", "Completato"]
    : ["Connessione SSH", "Connessione database", "Conteggio ticket", "Lettura ticket", "Embedding OpenAI", "Scrittura FAISS", "Completato"];
  const activeIndex = Math.max(0, steps.findIndex((step) => job.step.includes(step)));
  const title = isAnalysisJob
    ? job.type === "analysis_nightly"
      ? "Job notturno problemi noti"
      : "Rigenerazione problemi noti"
    : job.type === "append"
      ? "Append FAISS"
      : job.type === "resume_rebuild"
        ? "Ripresa rebuild FAISS"
        : "Ricostruzione FAISS";

  return (
    <section className={`job-panel elevated-panel job-${job.status}`}>
      <div className="list-header">
        <div className="panel-title">
          {job.status === "running" || job.status === "queued" ? <Loader2 className="spin" size={19} /> : <CheckCircle2 size={19} />}
          <div>
            <h2>{title}</h2>
            <p>{job.step}</p>
          </div>
        </div>
        <span className="soft-chip">{job.status}</span>
      </div>

      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${progress}%` }} />
      </div>
      <div className="progress-stats">
        <span>{progress}%</span>
        <span>
          {job.current.toLocaleString("it-IT")} / {job.total.toLocaleString("it-IT")} {isAnalysisJob ? "periodi" : "righe lette"}
        </span>
        <span>{job.written.toLocaleString("it-IT")} {isAnalysisJob ? "gruppi salvati" : "vettori scritti"}</span>
        <span>{job.skipped.toLocaleString("it-IT")} {isAnalysisJob ? "errori" : "saltati nel tentativo"}</span>
      </div>
      {isAnalysisJob ? (
        <p className="muted-line">
          Ticket analizzati nei periodi: <strong>{job.processed.toLocaleString("it-IT")}</strong>. Risultati condivisi salvati in SQLite.
        </p>
      ) : (
        <p className="muted-line">
          Tentativo: <strong>{job.attempt || 1}</strong>. Chunk/vettori scritti:{" "}
          <strong>{job.chunks.toLocaleString("it-IT")}</strong>. I ticket lunghi vengono suddivisi in piu' chunk senza perdere testo.
        </p>
      )}

      <div className="stepper">
        {steps.map((step, index) => (
          <div className={`step ${index <= activeIndex ? "active" : ""}`} key={step}>
            <span>{index + 1}</span>
            <p>{step}</p>
          </div>
        ))}
      </div>

      {job.message ? <p className="muted-line">{job.message}</p> : null}
      {job.error ? <ErrorBlock message={job.error} /> : null}
    </section>
  );
}

function Metric({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="metric interactive-card">
      {icon}
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </div>
  );
}

function LoadingPanel({ label }: { label: string }) {
  return (
    <section className="loading-panel elevated-panel" aria-live="polite">
      <Loader2 className="spin" size={20} />
      <span>{label}</span>
      <div className="skeleton-line" />
    </section>
  );
}

function AnalysisSkeleton() {
  return (
    <section className="skeleton-stack">
      <div className="skeleton-card" />
      <div className="skeleton-card short" />
      <div className="skeleton-card" />
    </section>
  );
}

function ErrorBlock({ message }: { message: string }) {
  return (
    <div className="error-block">
      <AlertCircle size={19} />
      <span>{message}</span>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
