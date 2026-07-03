import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertCircle,
  BarChart3,
  CalendarClock,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Clock3,
  Database,
  Layers,
  Loader2,
  MessageSquareText,
  PlayCircle,
  RefreshCcw,
  Save,
  Search,
  Server,
  Settings,
  ShieldAlert,
  Sparkles,
  Ticket
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
  first_seen?: string | null;
  last_seen?: string | null;
  sample_ticket_ids: Array<number | string>;
  sample_titles: string[];
};

type RecentProblemsResponse = {
  since?: string | null;
  total_recent_tickets: number;
  groups: ProblemGroup[];
  ai_summary?: string | null;
  ai_error?: string | null;
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

const HITS_PAGE_SIZE = 4;

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
  const [view, setView] = useState<"ask" | "analysis" | "config">("ask");
  const [health, setHealth] = useState<Health | null>(null);
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(8);
  const [askLoading, setAskLoading] = useState(false);
  const [askError, setAskError] = useState("");
  const [answer, setAnswer] = useState<AskResponse | null>(null);
  const [hitPage, setHitPage] = useState(1);
  const [days, setDays] = useState(30);
  const [analysis, setAnalysis] = useState<RecentProblemsResponse | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [analysisError, setAnalysisError] = useState("");
  const [config, setConfig] = useState<DatabaseConfig>(emptyConfig);
  const [configLoading, setConfigLoading] = useState(false);
  const [configSaving, setConfigSaving] = useState(false);
  const [configMessage, setConfigMessage] = useState("");
  const [configError, setConfigError] = useState("");
  const [latestLocalDate, setLatestLocalDate] = useState<string | null>(null);
  const [rebuildFromDate, setRebuildFromDate] = useState("");
  const [job, setJob] = useState<JobResponse | null>(null);

  useEffect(() => {
    fetch("/api/health")
      .then((response) => response.json())
      .then(setHealth)
      .catch((error) => setHealth({ ok: false, error: String(error) }));
  }, []);

  useEffect(() => {
    if (view === "analysis" && !analysis && !analysisLoading) {
      loadAnalysis();
    }
    if (view === "config" && !configLoading && !config.ssh_host && !config.query) {
      loadConfig();
    } else if (view === "config") {
      loadLatestJob();
    }
  }, [view]);

  useEffect(() => {
    if (!job || !["queued", "running"].includes(job.status)) return;
    const timer = window.setInterval(async () => {
      try {
        const response = await fetch(`/api/index/jobs/${job.id}`);
        const payload = await response.json();
        if (response.ok) {
          setJob(payload);
          if (payload.status === "completed") {
            refreshHealth();
            refreshLatestDate();
          }
        }
      } catch {
        // Keep the current job snapshot; the next poll can recover.
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job?.id, job?.status]);

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

  async function submitQuestion(event?: React.FormEvent) {
    event?.preventDefault();
    if (!question.trim()) return;
    setAskLoading(true);
    setAskError("");
    setAnswer(null);
    setHitPage(1);
    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: topK })
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Errore durante la ricerca");
      setAnswer(payload);
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

  async function loadAnalysis(nextDays = days) {
    setAnalysisLoading(true);
    setAnalysisError("");
    try {
      const response = await fetch(`/api/analysis/recent-problems?days=${nextDays}&limit=24&include_ai=true`);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Errore durante l'analisi");
      setAnalysis(payload);
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
      const response = await fetch("/api/config");
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
      const response = await fetch("/api/index/jobs/latest");
      const payload = await response.json();
      if (response.ok && payload && ["queued", "running", "failed"].includes(payload.status)) {
        setJob(payload);
      }
    } catch {
      // Non-blocking: the configuration page can still be used.
    }
  }

  async function refreshLatestDate() {
    try {
      const response = await fetch("/api/index/latest-date");
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
        headers: { "Content-Type": "application/json" },
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
        headers: { "Content-Type": "application/json" },
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
        headers: { "Content-Type": "application/json" },
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
      const response = await fetch("/api/index/append", { method: "POST" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Errore avvio append");
      setJob(payload);
    } catch (error) {
      setConfigError(error instanceof Error ? error.message : String(error));
    }
  }

  function updateConfig<K extends keyof DatabaseConfig>(key: K, value: DatabaseConfig[K]) {
    setConfig((current) => ({ ...current, [key]: value }));
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
        <div className={`status ${status.ok ? "status-ok" : "status-warn"}`}>
          {status.ok ? <CheckCircle2 size={18} /> : <AlertCircle size={18} />}
          <span>{status.label}</span>
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
        <button className={view === "config" ? "active" : ""} onClick={() => setView("config")}>
          <Settings size={18} />
          Configurazione
        </button>
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
              <button className="icon-button" onClick={() => loadAnalysis()} disabled={analysisLoading} aria-label="Aggiorna analisi">
                {analysisLoading ? <Loader2 className="spin" size={18} /> : <RefreshCcw size={18} />}
              </button>
            </div>
          </div>

          {analysisError ? <ErrorBlock message={analysisError} /> : null}
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
                  label="Picco principale"
                  value={analysis.groups[0] ? `${analysis.groups[0].count} casi` : "-"}
                />
                <Metric icon={<Clock3 size={18} />} label="Dal" value={analysis.since?.slice(0, 10) ?? "-"} />
                <Metric icon={<BarChart3 size={18} />} label="Periodo" value={`${days} giorni`} />
              </section>

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
            </>
          ) : analysisLoading ? null : (
            <div className="empty-state">Nessun dato caricato.</div>
          )}
        </section>
      ) : (
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
      )}
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

function JobProgress({ job }: { job: JobResponse }) {
  const progress = job.total > 0 ? Math.min(100, Math.round((job.current / job.total) * 100)) : 0;
  const steps = ["Connessione SSH", "Connessione database", "Conteggio ticket", "Lettura ticket", "Embedding OpenAI", "Scrittura FAISS", "Completato"];
  const activeIndex = Math.max(0, steps.findIndex((step) => job.step.includes(step)));
  const title = job.type === "append" ? "Append FAISS" : job.type === "resume_rebuild" ? "Ripresa rebuild FAISS" : "Ricostruzione FAISS";

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
          {job.current.toLocaleString("it-IT")} / {job.total.toLocaleString("it-IT")} righe lette
        </span>
        <span>{job.written.toLocaleString("it-IT")} vettori scritti</span>
        <span>{job.skipped.toLocaleString("it-IT")} saltati nel tentativo</span>
      </div>
      <p className="muted-line">
        Tentativo: <strong>{job.attempt || 1}</strong>. Chunk/vettori scritti:{" "}
        <strong>{job.chunks.toLocaleString("it-IT")}</strong>. I ticket lunghi vengono suddivisi in piu' chunk senza perdere testo.
      </p>

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
