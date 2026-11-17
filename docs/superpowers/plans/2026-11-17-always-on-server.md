# Moving Spix to an always-on server: migration plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or superpowers:subagent-driven-development
> if the owner asks for subagents) to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Run the website and every scheduled job on a small rented server that never sleeps. The Mac becomes a
development machine that may sleep. No published pick changes.

**Architecture:** Lift and shift. Same code, same SQLite files, same schedules. The Mac's launchd jobs become systemd
timers on Ubuntu. The server stays private at first, reachable only through Tailscale. Code reaches it with
`git pull` from GitHub, and data reaches the Mac as a daily copy.

**Tech stack:** Ubuntu 24.04 LTS, Python 3.13 (installed with `uv`), systemd timers, Tailscale, rsync, SQLite's
backup API.

**Spec:** No separate spec. The owner's request (17 Nov): "Write a migration plan. Do not make any code changes yet.
Include how difficult moving will be, estimated cost per month for now (lets not use the best resources before we
have real users, we can upgrade later)." Decision D115 in `docs/rework-tracker.md`.

**Status:** Waiting for the owner's choices (section 5). No code has changed.

---

## 1. In short

- **What moves:** the website and the 10 scheduled jobs. Chroma (19 GB) and the research files stay on the Mac.
- **Difficulty: medium-low.** The app is plain Python with SQLite files. Only four things tie it to the Mac:
  - the job scheduler (launchd);
  - `caffeinate`, used only on a Mac;
  - a sleep check that already answers "awake" when it cannot read the lid;
  - 36 old scripts with the Mac's home folder written into them.

  The daily refresh runs those old scripts, so they are the one real code fix. Nothing changes in the database.
- **The hard parts:**
  - never letting both machines run the jobs at once;
  - uploading about 17 GB of live data;
  - changing how the three Claude folders work, since production no longer lives on the Mac.
- **Effort:**
  - Claude: about 3–4 days of work, spread over a week, checks included.
  - You: about 1–2 hours in all: the account and payment, an SSH key, Tailscale, copying `.env`, the go-aheads, and
    being there for the switch.
- **Downtime:** the site is off for about 30–60 minutes, once.
- **Cost now:** about €10–17 a month, all in (section 3). Everything else stays the same price.
- **When:**
  - Prepare now (tasks 1–4 change nothing live).
  - Set the server up during the week of 23 Nov.
  - Switch on a morning in a week with no engine stage going live (decision 2).
  - Not during the 20–22 Nov weekend.

## 2. What runs today (measured 17 Nov)

| Job (launchd `com.bettingrag.*`) | Command | When |
|---|---|---|
| website | `uvicorn Scripts.web_app.api:app --host 0.0.0.0 --port 8000` | always (restarts itself) |
| daily-data-refresh | `caffeinate -i -s … -m Scripts.data_platform refresh-season --include-europe --if-due` | every hour at :15 |
| match-read-worker | `-m Scripts.data_platform match-read-cycle --once --mode website` | every 10 min |
| prediction-measurement | `-m Scripts.data_platform measurement-cycle --once` | every minute |
| fixture-schedule | `-m Scripts.data_platform sync-schedule --days-ahead 35 --if-stale-hours 6` | every hour |
| odds-archive | `-m Scripts.ops.odds_archive --execute` | every 3 hours |
| odds-archive-near-kickoff | `-m Scripts.ops.odds_archive --execute --near-kickoff-hours 3` | every 10 min |
| true-close | `-m Scripts.ops.true_close` | every hour at :30 |
| bookmaker-coverage | `-m Scripts.ops.bookmaker_coverage` | 09:45, 13:45, 17:45, 21:45 |
| scorecard | `-m Scripts.ops.scorecard --notify` | Mondays 08:00 |
| phase6-shadow | `Research/phase6-shadow-logo-2026-10-07/controller.py capture` (not in git) | every hour |

**Other facts**
- The times are Bucharest time (`Europe/Bucharest`), and the server must use the same zone.
- No job's settings hold secrets. The website's two settings are `PYTHONPATH` and `PYTHONDONTWRITEBYTECODE`, and
  every key lives in `.env`.
- The Discord bot is not running.
- **Memory:** the website uses about 60 MB, and the refresh's steps use a few hundred MB each.
- **The Mac:** Apple M3 Pro, 11 cores, 18 GB, Python 3.13.3.

**Live data to move: about 17 GB**
- `Index/platform.db`: 11 GB, growing.
- `Output/`: 0.9 GB. The old scripts read and write it.
- `Index/oddspapi`: 2.5 GB.
- Under 1 GB each:
  - `live.db`, `odds_archive`, `raw_archive`, `documents.sqlite3`, `football_data.sqlite3` (the pick evidence reads
    it);
  - the owner dashboard and stats-questions files, `home_wins.json`, the normalized and referee files, `cache`,
    `ops-alerts`, `bookmaker_coverage`, `golden`.

**Data that stays on the Mac: about 37 GB**
- `chroma` (19 GB): nothing running reads it (D81).
- Research data: `prediction_experiments` 15 GB, `history_staging` 2.1 GB, `research_environments`, the
  `game_read_*` files.
- The two old copies of `platform.db`, and `Research/` (5.6 GB).

## 3. The server and the cost

**The size it needs, for now:** 4 vCPU, 8 GB of memory, 75 GB or more of disk.
- **Disk:** about 17 GB of live data, 22 GB for two nightly copies of `platform.db`, plus room to grow and for logs.
- **Memory:** 8 GB leaves room for SQLite's cache and the refresh's pandas steps, and costs only a few euros more
  than 4 GB.
- **Speed:** the server is slower than the M3 Pro. The refresh waits mostly on the APIs, so this should be fine, but
  task 9 times it.

**Options (prices from public pages, checked 17 Nov 2026; confirm when ordering)**

| Provider and plan | Size | Price a month | Notes |
|---|---|---|---|
| **OVHcloud VPS-2 (recommended)** | 4 vCore, 8 GB, 75 GB NVMe | $8.50 on OVH's US page; expect about €8–11 with VAT in Europe | Daily backup included; can be ordered now |
| Netcup VPS 1000 | 4 vCore, 8 GB, 128 GB SSD | €14.50 on a 12-month term, €16.68 month to month (page says incl. 19% VAT) | More disk; Nuremberg, Vienna or Amsterdam |
| Hetzner CX33 | 4 vCPU, 8 GB, 80 GB | €8.49 + VAT; backups +20% | Hetzner has limited new cloud servers since 26 Jun 2026, and its CX plans were reported unavailable on 4 Nov. Take it only if the console offers it |
| DigitalOcean Basic 8 GB | 4 vCPU, 8 GB, 160 GB | $48 | Too dear for now |

**Monthly cost now**

| Item | Cost |
|---|---|
| Server (OVH VPS-2, or Netcup if more disk is wanted) | about €8–17 |
| The provider's daily disk backup | included at OVH; about €2 at Hetzner |
| Our nightly copy of `platform.db`, kept on the server and pulled to the Mac | €0 |
| Private access (Tailscale's free personal plan) | €0 |
| Domain and HTTPS | €0 for now; not needed while the server is private |
| **Total** | **about €10–17** |

API-Football, OddsPapi, OpenAI and Stripe cost the same as today, and their calls don't double.

**Later, with real users (not in this plan)**
- A bigger server (12–16 GB): about €12–30 a month, by resizing the same server.
- Off-site backup storage: about €3–4 a month (for example a 1 TB Hetzner Storage Box).
- A domain: about €10–15 a year. HTTPS is free with Let's Encrypt.

Still well under €50 a month.

## 4. How the work changes after the move

- **Pick changes:** no change. The stage rule (one pick-changing step a week) sets that pace, not the hosting.
- **Releases: one extra step.**
  - Today: merge, then restart the website.
  - After: merge, push, then run `deploy/deploy.sh` (pull, install if the requirements changed, restart), a few
    minutes more.
  - Deploys still wait for the owner's go-ahead, as restarts do today.
- **Research and backend work get faster.**
  - Heavy runs no longer share the machine with production.
  - No waiting for the refresh to finish before running the suite.
  - The Mac can sleep.
- **Real data in development:**
  - The Mac's `Index/platform.db` becomes a copy, at most a day old. Pull it again whenever fresher data is needed.
  - The test sites on 8100 and 8200 read that copy.
  - Checks that need the live database run on the server: read-only, inside the gate, as today.
- **Switches are safer.**
  - The live switches sit in the server's `.env`. The owner edits them there, and Claude says which line.
  - A switch changed in a Mac folder no longer changes the live site.
- **The three Claude folders stay as they are.** The orchestrator's folder is no longer production. It merges,
  pushes and deploys.

## 5. Decisions for the owner

1. **Provider.** OVHcloud VPS-2 (recommended: cheapest, daily backup included, can be ordered now), Netcup VPS 1000
   (more disk, about €5 more a month), or Hetzner CX33 if its console offers it.
2. **The switch date against the next engine stage.** The earliest next stage is Tue 24 Nov.
   - (a) Switch on Mon 23 Nov after the 08:00 scorecard and the daily refresh, and hold the next stage a week.
   - (b) Run the next stage first, and switch after its scorecard (Mon 30 Nov).
   - Either way, a switch never shares a week with a stage going live, so the scorecards stay clean.
3. **The phase6-shadow job** (hourly, from a `Research/` folder that is not in git): move it, or stop it?
4. **The owner dashboard (8300) and stats questions (8400):**
   - on the server: live data, opened through Tailscale;
   - or on the Mac: the day-old copy.

## Global constraints

- **No published pick changes.**
  - The golden check (`golden_snapshots check`, the switches written out inline) must give "no changes across 54
    fixtures" on the server before the switch.
  - The move is not an engine stage, but it never shares a week with one going live (decision 2).
- **One runner at a time.** The scheduled jobs run on exactly one machine at any moment. Two runners would double the
  API calls and split the database in two.
- **Secrets:**
  - Never read `.env`. The owner copies it to the server.
  - Never print keys or Telegram URLs, and redact `apiKey=` in log reads.
  - Never read `~/.claude/telegram-notify.env`. It is Claude Code's own file, and the server does not need it.
- **`platform.db` rules hold on the server as on the Mac:**
  - reads only inside `data_access()`;
  - never while the refresh runs;
  - short, read-only statements;
  - never go around the gate marker.
- **Go-aheads.** Every outward-facing step waits for the owner:
  - ordering and paying;
  - every command that changes the server;
  - pushes and deploys;
  - the switch;
  - restarts.
- **The server stays private** (Tailscale only, firewall closed) until the auth and security review that comes before
  real users.
- **Who does what:**
  - Code (tasks 1–4) goes on `backend` and is merged by the orchestrator through the usual checks.
  - The owner does task 5.
  - The orchestrator does the server work (tasks 6–9), since it owns production.
- **Commit names** (Large builds):
  - `Start server-move task N: …`;
  - `Done server-move task N: … (<tests>, D115)`;
  - `Fix server-move task N: …`.
- **Time zone:** the server runs in `Europe/Bucharest`.

## Review focus

1. **Both machines running jobs** (a job left enabled on the Mac, or started by hand). The second runner must exit
   at once, without an API call. Task 2's tests cover this.
2. **Times off by the zone** (the server left on UTC, or a timer written in UTC). Every job must fire at the same local
   time as today, and the refresh's wait for match time (D109) must read local time. Task 3's tests cover this.
3. **The Linux build giving different numbers** (other wheels, other floating point). Picks must not move. The golden
   check on the server (task 6) must be unchanged; if it isn't, stop and find out why before the switch.
4. **The disk filling up** (database growth, nightly copies, logs). The backup must refuse and alert while it can
   still be acted on. Task 4's tests cover this.
5. **Replacing the Mac's copy while a test site has it open.** Readers must keep working, and the swap must never
   leave a half-written file. Task 4's tests cover this.

## Progress

| Task | What | Status | Start | Done | Tests |
|---|---|---|---|---|---|
| 1 | No Mac home folder in the old scripts | Not started | | | |
| 2 | One runner at a time | Not started | | | |
| 3 | Server files: timers, install and deploy scripts | Not started | | | |
| 4 | Nightly copy of `platform.db` and the Mac's copy | Not started | | | |
| 5 | Owner: the server, SSH key, Tailscale | Not started | | | |
| 6 | Install on the server; tests and golden check there | Not started | | | |
| 7 | Seed the data; the website on the server, no jobs | Not started | | | |
| 8 | The switch | Not started | | | |
| 9 | The first week, then the docs | Not started | | | |

---

## Phase A: code on `backend` (changes nothing live)

### Task 1: No Mac home folder in the old scripts

**Files:**
- Modify: the 36 scripts that contain `/Users/sanduandrei/` (per-league stats, features and profile scripts in
  `Scripts/{Premier_League,La_Liga,Seria_A,Bundesliga,Ligue_1,Champions_League,Europa_League,Conference_League}/`,
  `Scripts/top_5_leagues_players.py` and two in `Scripts/rag_ingest/`).
- Test: `Scripts/tests/test_no_home_paths.py`.

**Interfaces:** each script builds its paths from `ROOT = Path(__file__).resolve().parents[N]` (the repo root), as
`Scripts/pull_season.py` already does.

- [ ] **Step 1: Write the failing tests.**
  - `test_no_script_names_the_mac_home_folder`: no `.py` file under `Scripts/` outside `tests/` contains `/Users/`.
  - `test_old_paths_resolve_to_the_same_folders`: on the Mac, each changed constant, evaluated with its module's
    `__file__`, equals the old absolute path. Evaluate the expression from the source with `ast`; never import the
    scripts, since some run at import.
- [ ] **Step 2: Run them.** Expected: FAIL, with 36 files named.
- [ ] **Step 3: Replace each absolute path** with the root-relative one.
- [ ] **Step 4: Run them.** Expected: PASS. Then run the full suite and the golden check (unchanged).
- [ ] **Step 5: Commit.** After the merge, the next daily refresh's `legacy_model_refresh` stage on the Mac is the
  real check. It must succeed before task 6.

### Task 2: One runner at a time

**Files:**
- Create: `Scripts/data_platform/services/runner_guard.py`.
- Modify: the 8 entry points behind the scheduled jobs (both odds-archive jobs share one; phase6-shadow follows
  decision 3):
  - `refresh-season`, `match-read-cycle`, `measurement-cycle` and `sync-schedule` in `Scripts/data_platform`;
  - `odds_archive`, `true_close`, `bookmaker_coverage` and `scorecard` in `Scripts/ops`.
- Test: `Scripts/tests/data_platform/test_runner_guard.py`.

**Interfaces:**
- `Index/runner-host` holds the one host name allowed to run scheduled jobs.
- `runner_guard.allowed(path=…, host=socket.gethostname()) -> bool`: True when the file is missing (today's
  behaviour) or names this host.
- Each job's entry point exits with code 0 and one log line ("not the runner: <host>") before any API call or write.

- [ ] **Step 1: Write the failing tests.**
  - `test_a_missing_marker_lets_the_job_run`.
  - `test_another_host_stops_the_job_before_any_call`: the job's API client and database are never touched.
  - `test_every_scheduled_entry_point_checks_the_marker`: the 8 commands above, each run with a marker naming
    another host, exit 0 without work.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement** the guard and call it first in each entry point.
- [ ] **Step 4: Run them.** Expected: PASS. Then run the full suite and the golden check (unchanged).
- [ ] **Step 5: Commit.**

### Task 3: Server files: timers, install and deploy scripts

**Files:**
- Create in `deploy/`:
  - `systemd/spix-website.service`;
  - `systemd/spix-<job>.service` and `spix-<job>.timer` for each job in section 2 (phase6-shadow per decision 3);
  - `logrotate/spix`;
  - `install.sh`;
  - `deploy.sh`;
  - `README.md` (the commands, in plain words).
- Test: `Scripts/tests/test_deploy_files.py`.

**Interfaces:**
- **The services:**
  - run as user `spix` in `/srv/spix`, with `/srv/spix/.venv/bin/python`;
  - write to the same log files as today (`Index/logs/<job>.stdout.log`, `.stderr.log`), rotated weekly and kept 8
    weeks;
  - drop `caffeinate`;
  - add no `EnvironmentFile` (the app loads `.env` itself, as today).
- **The timers keep today's times, in local time:**
  - hourly at :15 (refresh) and :30 (true-close);
  - Mondays 08:00 (scorecard);
  - 09:45, 13:45, 17:45 and 21:45 (bookmaker coverage);
  - every 60 s, 10 min, 1 h and 3 h for the interval jobs;
  - calendar timers use `Persistent=true`.
- **The website** runs `uvicorn Scripts.web_app.api:app --host 0.0.0.0 --port 8000` with `Restart=always`. The
  firewall opens port 8000 only on `tailscale0`.
- **`install.sh`** is safe to run twice. It:
  - creates user `spix` and sets the time zone to `Europe/Bucharest`;
  - installs Python 3.13 with `uv` and builds the venv from `requirements.txt`;
  - sets up `ufw` (deny inbound; SSH and 8000 on `tailscale0` only), unattended security upgrades, the logrotate
    file and the units (installed, not enabled).
- **`deploy.sh`:**
  - `git pull --ff-only` on `main`;
  - `pip install -r requirements.txt` only when it changed;
  - restart the website;
  - print the commit now running.

- [ ] **Step 1: Write the failing tests.**
  - `test_every_launchd_job_has_a_systemd_twin`: a table of today's jobs and times (section 2) against the parsed
    `.timer` and `.service` files.
  - `test_calendar_timers_are_local_and_persistent`.
  - `test_install_sets_the_bucharest_zone_and_closes_the_firewall`.
  - `test_no_unit_reads_or_names_a_secret`: no unit sets a key or an `EnvironmentFile`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Write the files.**
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Commit.**

### Task 4: Nightly copy of `platform.db` and the Mac's copy

**Files:**
- Create:
  - `Scripts/ops/platform_backup.py`;
  - `Scripts/ops/dev_copy.py`;
  - `deploy/systemd/spix-backup.service` and `.timer`, at 04:30.
- Test: `Scripts/tests/test_platform_backup.py`.

**Interfaces:**
- `platform_backup` copies `platform.db` with SQLite's backup API inside `data_access()`.
  - It refuses while the refresh runs.
  - It writes `Index/backups/platform-YYYY-MM-DD.db` through a temporary name, runs `PRAGMA quick_check` on the copy,
    and keeps the newest 2.
  - It refuses, and sends an ops alert, when the copy would leave less than 15 GB free.
- `dev_copy` (run on the Mac) fetches the newest copy with `rsync` into a temporary file beside `Index/platform.db`,
  checks it, then renames it into place. Open readers keep the old file, and a half-written file is never in place.

- [ ] **Step 1: Write the failing tests.**
  - `test_backup_runs_inside_the_gate_and_refuses_while_the_refresh_runs`.
  - `test_backup_keeps_the_newest_two_and_checks_each_copy`.
  - `test_backup_refuses_and_alerts_when_the_disk_would_fall_below_15_gb`.
  - `test_dev_copy_swaps_in_one_rename_and_leaves_readers_working`.
  - `test_a_failed_check_leaves_the_old_copy_in_place`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them.** Expected: PASS. Then run the full suite and the golden check (unchanged).
- [ ] **Step 5: Commit.**

## Phase B: the server beside the Mac (no jobs run there)

### Task 5: The owner sets up the server, SSH key and Tailscale

- [ ] Order the server chosen in decision 1: Ubuntu 24.04 LTS, an EU location. Add your SSH public key when
  ordering.
- [ ] Make a Tailscale account (free personal plan). Install it on the Mac and on the server, and note the server's
  Tailscale name.
- [ ] In the API-Football and OddsPapi dashboards, check whether access is limited to listed IP addresses. If it is,
  add the server's address.
- [ ] On GitHub, add the server's read-only deploy key. The orchestrator makes the key on the server and shows its
  public half, which is not a secret.

### Task 6: Install on the server; tests and golden check there

- [ ] Run `deploy/install.sh` over SSH and clone `main` into `/srv/spix`.
- [ ] You copy `.env` yourself, and Claude never reads it:
  `scp ~/Desktop/Betting_RAG/.env spix@<server>:/srv/spix/.env`, then `chmod 600 /srv/spix/.env`.
- [ ] Check that `engine_switches.snapshot()` on the server matches the Mac's (switch values only).
- [ ] Run the full suite on the server. Expected: the same passes as on the Mac, without the 3 worktree-only errors.
- [ ] Copy `Index/golden` and run the golden check with the switches written out inline. Expected: "no changes across
  54 fixtures". If not, stop and find out why (review focus 3).

### Task 7: Seed the data; the website on the server, with no jobs

- [ ] On the Mac, outside refresh hours, make the copy with `platform_backup` (inside the gate), then `rsync` it and
  the rest of the live data (section 2) and `Output/` to the server. Expect about 17 GB: at 50 Mbit/s upload, about
  45 minutes.
- [ ] On the server, run `PRAGMA quick_check` on `platform.db`.
- [ ] Start only `spix-website` (no timers). Over Tailscale, check the home page, today's board, a match page, the
  Research Area, Best Bets and `/api/health/platform` against the live site. The data is the copy's day.
- [ ] Stop the website again. Nothing on the server has called an API.

## Phase C: the switch

### Task 8: The switch

**When:** the date chosen in decision 2, in the morning, with the owner there. Three conditions:
- the daily refresh has finished (`pgrep -f refresh-season` finds nothing, and there is no gate marker);
- no covered match kicks off within 3 hours;
- no engine stage goes live that week.

- [ ] **Stop the Mac.**
  - Stop each of the 11 launchd jobs with
    `launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.bettingrag.<job>.plist`.
  - Move the plists to `Index/launchd-backup/<date>/`, so they don't load again at login.
  - Check that no job process is left.
- [ ] **Mark the Mac:** write the server's host name into the Mac's `Index/runner-host`.
- [ ] **Final copy.** `rsync` `platform.db` (only the changed blocks travel) and the other live files, then run
  `quick_check` on the server.
- [ ] **Start the server.**
  - Write its own host name into its `Index/runner-host`.
  - Enable and start the website and every timer.
- [ ] **Watch the first hour:**
  - a measurement cycle every minute;
  - a Match Reads cycle;
  - an odds archive run;
  - the website's pages;
  - clean logs (redacted reads);
  - `/api/health/platform`.
- [ ] **Downtime** runs from the first step to this one: about 30–60 minutes.
- [ ] **Going back,** if needed in the first days:
  - stop the server's timers and website;
  - `rsync` its `platform.db` and live files back to the Mac, so the data written there is kept;
  - remove the Mac's `Index/runner-host`;
  - put the plists back and load them.

### Task 9: The first week, then the docs

- [ ] **Watch the next daily refresh end to end,** with its time against the Mac's (about 40 minutes).
  - The `legacy_model_refresh` stage must succeed (task 1).
  - If the refresh runs much longer, resize the server one step (about €3–6 more a month).
- [ ] **Check that a failed refresh would still alert to Telegram.** Send a test message only with the owner's
  go-ahead.
- [ ] **Check the nightly copy** and one `dev_copy` pull to the Mac.
- [ ] **After a quiet week, the orchestrator updates the docs:**
  - CLAUDE.md: Running things, Parallel sessions, the data rules (`platform.db` lives on the server, and the Mac holds
    a copy) and the status block;
  - the tracker under D115.
- [ ] **Chroma** stays on the Mac until A7.

## Phase D: later, with real users (not in this plan)

- Public HTTPS with a domain (Caddy and Let's Encrypt), after the auth and security review.
- Stripe webhooks.
- A bigger server if the load needs it.
- Off-site backups with a second provider.

## Sources (prices, checked 17 Nov 2026)

- OVHcloud VPS: https://www.ovhcloud.com/en/vps/
- Netcup VPS: https://www.netcup.com/en/server/vps
- Hetzner's 2026 prices and limits:
  - https://northflank.com/blog/hetzner-cloud-server-price-increases
  - https://www.vincentschmalbach.com/hetzner-cheap-cloud-unavailable-price-increases/
  - https://hostmath.dev/guides/hetzner-cx-cax-unavailable-what-to-buy
- Hetzner Storage Box: https://www.hetzner.com/storage/storage-box/bx11/
- DigitalOcean: https://onedollarvps.com/pricing/digitalocean-pricing
