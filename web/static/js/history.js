(function () {
  const tablePanel = document.getElementById("view-table");
  const chartsPanel = document.getElementById("view-charts");
  const tableBtn = document.getElementById("view-table-btn");
  const chartsBtn = document.getElementById("view-charts-btn");
  const matchFilter = document.getElementById("match-filter");
  const searchInput = document.getElementById("log-filter");
  const countEl = document.getElementById("compare-count");

  function setView(which) {
    const showTable = which === "table";
    if (tablePanel) {
      tablePanel.classList.toggle("hidden", !showTable);
      tablePanel.toggleAttribute("hidden", !showTable);
    }
    if (chartsPanel) {
      chartsPanel.classList.toggle("hidden", showTable);
      chartsPanel.toggleAttribute("hidden", showTable);
    }
    if (tableBtn) tableBtn.classList.toggle("is-active", showTable);
    if (chartsBtn) chartsBtn.classList.toggle("is-active", !showTable);
    if (!showTable) initChartsOnce();
  }

  if (tableBtn) tableBtn.addEventListener("click", () => setView("table"));
  if (chartsBtn) chartsBtn.addEventListener("click", () => setView("charts"));

  const cards = Array.from(document.querySelectorAll(".compare-card"));

  function cardVisible(card, q, matchVal) {
    const mk = card.getAttribute("data-match-key") || "";
    if (matchVal && mk !== matchVal) return false;
    if (!q) return true;
    return card.textContent.toLowerCase().includes(q);
  }

  function applyFilters() {
    const q = searchInput ? searchInput.value.trim().toLowerCase() : "";
    const matchVal = matchFilter ? matchFilter.value : "";
    cards.forEach((card) => {
      card.style.display = cardVisible(card, q, matchVal) ? "" : "none";
    });
    if (countEl) {
      const n = cards.filter((c) => c.style.display !== "none").length;
      const total = cards.length;
      countEl.textContent =
        total === 0
          ? ""
          : n === total
            ? `Showing all ${n} match${n === 1 ? "" : "es"}`
            : `Showing ${n} of ${total} matches`;
    }
  }

  if (searchInput) searchInput.addEventListener("input", applyFilters);
  if (matchFilter) matchFilter.addEventListener("change", applyFilters);
  applyFilters();

  let chartsDone = false;
  function initChartsOnce() {
    if (chartsDone) return;
    const raw = document.getElementById("chart-summary-data");
    if (!raw || typeof Chart === "undefined") return;

    let summary;
    try {
      summary = JSON.parse(raw.textContent);
    } catch {
      return;
    }

    chartsDone = true;

    const css = getComputedStyle(document.documentElement);
    const accent = css.getPropertyValue("--accent").trim() || "#4ade80";
    const danger = css.getPropertyValue("--danger").trim() || "#f87171";
    const muted = css.getPropertyValue("--text-muted").trim() || "#a8a8b3";

    Chart.defaults.color = "#a8a8b3";
    Chart.defaults.borderColor = "rgba(255, 255, 255, 0.06)";
    Chart.defaults.font.family = 'Inter, system-ui, sans-serif';
    Chart.defaults.font.size = 13;

    const commonLegend = {
      position: "bottom",
      labels: { boxWidth: 14, padding: 18, font: { size: 13 } },
    };

    const w = summary.winner || {};
    const wData = [w.correct || 0, w.wrong || 0, w.pending || 0];
    const wSum = wData.reduce((a, b) => a + b, 0);
    const wEl = document.getElementById("chart-winner");
    if (wEl && wSum > 0) {
      new Chart(wEl, {
        type: "doughnut",
        data: {
          labels: ["Match", "Miss", "Waiting / n/a"],
          datasets: [
            {
              data: wData,
              backgroundColor: [accent, danger, "rgba(168, 168, 179, 0.35)"],
              borderWidth: 0,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: true,
          plugins: {
            legend: commonLegend,
            tooltip: {
              callbacks: {
                label: (ctx) => {
                  const n = ctx.raw || 0;
                  const pct = wSum ? ((n / wSum) * 100).toFixed(0) : "0";
                  return ` ${ctx.label}: ${n} (${pct}%)`;
                },
              },
            },
          },
          cutout: "58%",
        },
      });
    }

    const p = summary.potm || {};
    const pData = [p.correct || 0, p.wrong || 0, p.pending || 0];
    const pSum = pData.reduce((a, b) => a + b, 0);
    const pEl = document.getElementById("chart-potm");
    if (pEl && pSum > 0) {
      new Chart(pEl, {
        type: "doughnut",
        data: {
          labels: ["Match", "Miss", "Waiting / n/a"],
          datasets: [
            {
              data: pData,
              backgroundColor: [accent, danger, "rgba(168, 168, 179, 0.35)"],
              borderWidth: 0,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: true,
          plugins: {
            legend: commonLegend,
            tooltip: {
              callbacks: {
                label: (ctx) => {
                  const n = ctx.raw || 0;
                  const pct = pSum ? ((n / pSum) * 100).toFixed(0) : "0";
                  return ` ${ctx.label}: ${n} (${pct}%)`;
                },
              },
            },
          },
          cutout: "58%",
        },
      });
    }

    const tl = summary.timeline || [];
    const tlEl = document.getElementById("chart-timeline");
    if (tlEl && tl.length > 0) {
      const labels = tl.map((x) => x.label);
      const values = tl.map((x) => x.p_team1_pct);
      const colors = tl.map((x) => {
        if (x.outcome === "hit") return accent;
        if (x.outcome === "miss") return danger;
        return muted;
      });
      new Chart(tlEl, {
        type: "bar",
        data: {
          labels,
          datasets: [
            {
              label: "Chance for left team in log %",
              data: values,
              backgroundColor: colors,
              borderRadius: 6,
              maxBarThickness: 36,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              ticks: { maxRotation: 45, minRotation: 0, font: { size: 12 } },
              grid: { display: false },
            },
            y: {
              min: 0,
              max: 100,
              title: {
                display: true,
                text: "Probability (%)",
                font: { size: 13 },
              },
              ticks: { font: { size: 12 } },
              grid: { color: "rgba(255, 255, 255, 0.06)" },
            },
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                afterLabel: (ctx) => {
                  const o = tl[ctx.dataIndex]?.outcome;
                  if (o === "hit") return "Winner pick: Match";
                  if (o === "miss") return "Winner pick: Miss";
                  return "Not graded yet";
                },
              },
            },
          },
        },
      });
    }

    const cal = summary.calibration || [];
    const calEl = document.getElementById("chart-calibration");
    if (calEl && cal.length > 0) {
      const labels = cal.map((c) => c.label);
      const rates = cal.map((c) => (c.rate == null ? 0 : c.rate));
      const graded = cal.map((c) => c.graded || 0);
      new Chart(calEl, {
        type: "bar",
        data: {
          labels,
          datasets: [
            {
              label: "% correct in bucket",
              data: rates,
              backgroundColor: rates.map((r, i) =>
                graded[i] === 0 ? "rgba(168, 168, 179, 0.35)" : accent
              ),
              borderRadius: 6,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { ticks: { font: { size: 12 } }, grid: { display: false } },
            y: {
              min: 0,
              max: 100,
              title: {
                display: true,
                text: "Accuracy (%)",
                font: { size: 13 },
              },
              ticks: { font: { size: 12 } },
              grid: { color: "rgba(255, 255, 255, 0.06)" },
            },
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: (ctx) => {
                  const i = ctx.dataIndex;
                  const g = graded[i];
                  if (!g) return " No graded rows in this band";
                  return ` ${rates[i]}% right (${cal[i].correct}/${g} graded)`;
                },
              },
            },
          },
        },
      });
    }
  }
})();
