# Ticket Knowledge Assistant

App React + FastAPI per interrogare un vector store FAISS di ticket storici e trovare casi simili gia' risolti.

## Prerequisiti

- Python 3.11+
- Node.js 20+
- Variabile ambiente `OPENAI_API_KEY`
- File in `FAISS/`:
  - `ticket_index.faiss`
  - `ticket_ids.txt`
  - `ticket_data.csv`

Il backend puo' scegliere automaticamente il modello embedding leggendo la dimensione dell'indice FAISS.

## Avvio backend

```powershell
copy .env.example .env
# modifica .env e inserisci OPENAI_API_KEY
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

## Avvio frontend

```powershell
cd frontend
corepack pnpm install
corepack pnpm run dev
```

Apri l'URL mostrato da Vite, di solito `http://localhost:5173`.

Al primo avvio viene creato un utente admin locale in `backend/app_config.sqlite`:

- username: `admin`
- password: `admin`

Da UI l'admin puo' creare altri utenti con ruolo `admin` o `user`.

Se Node non riesce a verificare il certificato del registry npm, usa lo store certificati di Windows nella sessione corrente:

```powershell
$env:NODE_OPTIONS='--use-system-ca'
corepack pnpm install
```

## Configurazione

Le opzioni principali sono in `.env`:

```env
OPENAI_API_KEY=...
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=auto
FAISS_DIR=FAISS
```

Con `OPENAI_EMBEDDING_MODEL=auto`:

- indice a `1536` dimensioni -> `text-embedding-3-small`
- indice a `3072` dimensioni -> `text-embedding-3-large`

Puoi comunque impostare un modello esplicito nel `.env` se hai creato l'indice con un modello diverso.

## API principali

- `GET /api/health` controlla indice, CSV e configurazione.
- `POST /api/auth/login` crea una sessione bearer.
- `GET/POST /api/users` gestisce utenti, solo per admin.
- `POST /api/ask` cerca ticket simili e genera una risposta con citazioni ai ticket.
- `GET /api/analysis/recent-problems` raggruppa i ticket recenti per problema noto e produce una sintesi opzionale con OpenAI.
- `GET/POST /api/config` legge e salva configurazione SSH/DB/query in SQLite locale.
- `POST /api/index/rebuild` ricostruisce FAISS da zero partendo da una data.
- `POST /api/index/append` aggiunge al FAISS i ticket successivi all'ultima data locale.
- `GET /api/index/jobs/{job_id}` espone stato, step e avanzamento del job.

## Configurazione indice

La pagina `Configurazione` del frontend permette di salvare SSH, database e query manuale in `backend/app_config.sqlite`.
Quel file e' ignorato da git perche' puo' contenere credenziali.

Le operazioni di rebuild/append usano sempre `text-embedding-3-large`.
La query deve restituire queste colonne, in questo ordine:

```sql
id, thread_id, staff_id, user_id, poster, created, title, body
```

La rebuild usa una cartella temporanea `FAISS_build/` e pubblica i file finali in `FAISS/` solo a completamento.

## Cache browser

Il frontend salva nel localStorage:

- cronologia delle domande con risposte gia' generate;
- cache dei problemi noti per periodo e numero vettori FAISS.

Il pulsante di refresh nei problemi noti forza una nuova chiamata e aggiorna la cache.
