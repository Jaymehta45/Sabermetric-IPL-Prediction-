(function () {
  const form = document.getElementById("predict-form");
  const errEl = document.getElementById("form-error");
  const resultsSection = document.getElementById("results-section");
  const mount = document.getElementById("results-mount");
  const submitBtn = document.getElementById("submit-btn");
  const btnText = submitBtn.querySelector(".btn-text");
  const btnLoading = submitBtn.querySelector(".btn-loading");
  const fixtureSelect = document.getElementById("fixture-select");

  if (fixtureSelect) {
    fixtureSelect.addEventListener("change", async () => {
      const opt = fixtureSelect.selectedOptions[0];
      if (!opt || !opt.value) return;
      const home = opt.dataset.home || "";
      const away = opt.dataset.away || "";
      const date = opt.dataset.date || "";
      document.getElementById("team1_name").value = home;
      document.getElementById("team2_name").value = away;
      if (date) document.getElementById("match_date").value = date;
      document.getElementById("match_key").value = `IPL2026-M${String(opt.value).padStart(2, "0")}`;
      try {
        const r = await fetch(`/api/fixtures/${opt.value}`);
        if (r.ok) {
          const j = await r.json();
          if (j.venue_canonical) document.getElementById("venue").value = j.venue_canonical;
          else if (j.venue_city) document.getElementById("venue").value = j.venue_city;
        }
      } catch (_) {}
    });
  }

  function esc(s) {
    const d = document.createElement("div");
    d.textContent = s == null ? "" : String(s);
    return d.innerHTML;
  }

  function fmtNum(x, decimals) {
    if (x == null || x === "") return "—";
    const n = Number(x);
    if (Number.isNaN(n)) return "—";
    return n.toFixed(decimals);
  }

  function renderResults(data) {
    const p = data.predictions || {};
    const sim = data.simulation || {};
    const teams = data.teams_display || {};
    const ens = sim.ensemble_p_team1;
    const t1 = teams.team1_first_innings || "Team 1 (1st inn.)";
    const t2 = teams.team2_chase || "Team 2 (chase)";

    let probPct = ens != null ? Math.round(ens * 1000) / 10 : null;
    if (probPct == null && sim.win_probability_team1 != null) {
      probPct = Math.round(sim.win_probability_team1 * 1000) / 10;
    }

    const top5 = p.top5_batsmen || [];
    const top3 = p.top3_bowlers || [];
    const bb = p.best_batsman || {};
    const bw = p.best_bowler || {};
    const pom = p.player_of_the_match || {};

    const rows5 = top5
      .map(
        (r, i) =>
          `<tr><td>${i + 1}</td><td class="mono">${esc(r.player_id)}</td><td>${esc(r.side)}</td><td>${fmtNum(
            r.predicted_runs,
            2
          )}</td><td>${r.strike_rate != null ? fmtNum(r.strike_rate, 1) : "—"}</td></tr>`
      )
      .join("");

    const rows3 = top3
      .map(
        (r, i) =>
          `<tr><td>${i + 1}</td><td class="mono">${esc(r.player_id)}</td><td>${esc(r.side)}</td><td>${fmtNum(
            r.predicted_wickets,
            2
          )}</td><td>${r.economy != null ? fmtNum(r.economy, 2) : "—"}</td></tr>`
      )
      .join("");

    const exportJson = data.prediction_log_export
      ? JSON.stringify(data.prediction_log_export, null, 2)
      : "";

    mount.innerHTML = `
      <p class="page-desc result-context" style="margin-bottom:1rem;">
        Innings order in the model: first dig = <strong>${esc(t1)}</strong> · chase = <strong>${esc(t2)}</strong>
        ${teams.batting_order_swapped ? " · <em>(toss order differed from your Team 1 / Team 2 typing order)</em>" : ""}
      </p>

      <div class="result-lists panel">
        <h3 class="panel-title result-lists-heading">Top 5 batters &amp; top 3 bowlers</h3>
        <p class="field-help result-lists-lead">
          <strong>Predicted runs / wickets</strong> = model score for this line-up. <strong>Strike rate</strong> and <strong>economy</strong> = your players’ recent rolling pre-match stats (same inputs the model uses), not live match numbers.
        </p>
        <div class="result-lists-grid">
          <div class="result-list-col">
            <h4 class="result-list-title">Top 5 batters</h4>
            <div class="table-wrap">
              <table class="mini-table mini-table--rich">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Player</th>
                    <th>Side</th>
                    <th>Predicted runs</th>
                    <th>SR (recent)</th>
                  </tr>
                </thead>
                <tbody>${rows5 || '<tr><td colspan="5">—</td></tr>'}</tbody>
              </table>
            </div>
          </div>
          <div class="result-list-col">
            <h4 class="result-list-title">Top 3 bowlers</h4>
            <div class="table-wrap">
              <table class="mini-table mini-table--rich">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Player</th>
                    <th>Side</th>
                    <th>Predicted wkts</th>
                    <th>Econ. (recent)</th>
                  </tr>
                </thead>
                <tbody>${rows3 || '<tr><td colspan="5">—</td></tr>'}</tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <div class="result-grid">
        <div class="result-card highlight">
          <div class="result-label">Model pick to win</div>
          <div class="result-value">${esc(p.winning_team)}</div>
        </div>
        <div class="result-card">
          <div class="result-label">Model’s chance ${esc(t1)} wins</div>
          <div class="result-value">${probPct != null ? probPct + "%" : "—"}</div>
          <div class="result-sub">Blended model + Monte Carlo when sims &gt; 0</div>
          ${
            probPct != null
              ? `<div class="prob-bar"><div class="prob-bar-fill" style="width:${Math.min(
                  100,
                  probPct
                )}%"></div></div>`
              : ""
          }
        </div>
        <div class="result-card">
          <div class="result-label">Best batsman (pred. runs)</div>
          <div class="result-value">${esc(bb.player_id)}</div>
          <div class="result-sub">${esc(bb.side || "")} · ${bb.predicted_runs != null ? Number(bb.predicted_runs).toFixed(2) : "—"} runs</div>
        </div>
        <div class="result-card">
          <div class="result-label">Best bowler (pred. wkts)</div>
          <div class="result-value">${esc(bw.player_id)}</div>
          <div class="result-sub">${esc(bw.side || "")} · ${bw.predicted_wickets != null ? Number(bw.predicted_wickets).toFixed(2) : "—"} wkts</div>
        </div>
        <div class="result-card highlight">
          <div class="result-label">Model player of the match (score)</div>
          <div class="result-value">${esc(pom.player_id)}</div>
          <div class="result-sub">${esc(pom.side || "")}</div>
        </div>
      </div>
      ${
        exportJson
          ? `<div class="json-export">
          <h3 class="panel-title" style="font-size:1rem">Log export (for <code>log_prediction.py pre</code>)</h3>
          <pre class="json-pre">${esc(exportJson)}</pre>
          <button type="button" class="btn btn-secondary" id="copy-json">Copy JSON</button>
        </div>`
          : "<p class=\"page-desc\">Set <strong>match date</strong> above to generate log JSON.</p>"
      }
    `;

    const copyBtn = document.getElementById("copy-json");
    if (copyBtn && exportJson) {
      copyBtn.addEventListener("click", () => {
        navigator.clipboard.writeText(exportJson).then(() => {
          copyBtn.textContent = "Copied!";
          setTimeout(() => (copyBtn.textContent = "Copy JSON"), 2000);
        });
      });
    }

    resultsSection.hidden = false;
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    errEl.hidden = true;
    btnText.hidden = true;
    btnLoading.hidden = false;
    submitBtn.disabled = true;

    const fd = new FormData(form);
    const body = {
      team1_name: fd.get("team1_name")?.toString().trim(),
      team2_name: fd.get("team2_name")?.toString().trim(),
      xi1: fd.get("xi1")?.toString() || "",
      xi2: fd.get("xi2")?.toString() || "",
      batting_first: fd.get("batting_first")?.toString() || null,
      pitch_text: fd.get("pitch_text")?.toString() || null,
      match_date: fd.get("match_date")?.toString() || null,
      venue: fd.get("venue")?.toString() || null,
      match_key: fd.get("match_key")?.toString() || null,
      use_registry: fd.get("use_registry") === "on",
      no_squad_check: fd.get("no_squad_check") === "on",
      use_shrinkage: fd.get("use_shrinkage") === "on",
      sims: parseInt(fd.get("sims")?.toString() || "400", 10),
      team1_impact: fd.get("team1_impact")?.toString().trim() || null,
      team2_impact: fd.get("team2_impact")?.toString().trim() || null,
    };

    if (body.batting_first === "") body.batting_first = null;

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const text = await res.text();
      let data = {};
      try {
        data = JSON.parse(text);
      } catch (_) {
        if (!res.ok) throw new Error(text || res.statusText);
      }
      if (!res.ok) {
        let d = data.detail;
        if (Array.isArray(d)) d = d.map((x) => x.msg || JSON.stringify(x)).join("; ");
        throw new Error(d || text || res.statusText || "Request failed");
      }
      renderResults(data);
    } catch (err) {
      errEl.textContent = err.message || String(err);
      errEl.hidden = false;
    } finally {
      btnText.hidden = false;
      btnLoading.hidden = true;
      submitBtn.disabled = false;
    }
  });
})();
