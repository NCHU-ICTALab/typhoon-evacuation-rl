(function () {
  "use strict";

  var PROFILES = {
    count: { key: "count", label: "艘數優先" },
    balanced: { key: "balanced", label: "平衡方案" },
    gt: { key: "gt", label: "GT 優先" },
    risk: { key: "risk", label: "風險優先" }
  };
  var METHODS = {
    fcfs: { label: "FCFS", detail: "先整備先派" },
    risk_aware: { label: "Risk-aware", detail: "安全餘裕優先" },
    value_density: { label: "Value-density", detail: "偏好價值／通航時間" },
    rl: { label: "Python RL", detail: "MaskablePPO 實際推論" }
  };
  var BASELINE_ORDER = ["fcfs", "risk_aware", "value_density"];
  var PROFILE_ORDER = ["count", "balanced", "gt", "risk"];

  var state = {
    closure: 4,
    tugs: 8,
    pressure: 3,
    method: "rl",
    policy: "balanced",
    filter: "all",
    payload: null,
    result: null,
    requestSerial: 0
  };

  var byId = function (id) { return document.getElementById(id); };
  var fmt = new Intl.NumberFormat("zh-TW", { maximumFractionDigits: 0 });
  var esc = function (value) {
    return String(value).replace(/[&<>"']/g, function (char) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char];
    });
  };

  function delta(value, base, suffix) {
    var difference = value - base;
    return (difference > 0 ? "+" : "") + fmt.format(difference) + (suffix || "");
  }

  function riskClass(points) {
    if (points >= 4) return "risk-high";
    if (points >= 2) return "risk-mid";
    return "risk-low";
  }

  function outcomeKey(result) {
    return [result.kpi.count, result.kpi.gt, result.kpi.risk].join("|");
  }

  function allVessels(result) {
    return result.schedule.concat(result.remaining);
  }

  function selectedResult() {
    if (!state.payload) return null;
    if (state.method === "rl") return state.payload.rl_results[state.policy];
    return state.payload.baselines[state.method];
  }

  function weightText(preference) {
    return preference.weights.map(function (weight) {
      return Math.round(weight * 100);
    }).join(" / ");
  }

  function resultCard(result, title, attributes, tag, detail, active) {
    return '<button class="pareto-card' + (active ? ' active' : '') + '" ' + attributes + '>' +
      '<header><strong>' + esc(title) + '</strong><span>' + esc(tag) + '</span></header>' +
      '<small>' + esc(detail) + '</small><div class="pareto-metrics">' +
      '<span>撤離艘數<b>' + result.kpi.count + '</b></span>' +
      '<span>撤離 GT<b>' + fmt.format(result.kpi.gt) + '</b></span>' +
      '<span>風險點<b>' + result.kpi.risk + '</b></span></div></button>';
  }

  function renderComparison() {
    var baselines = BASELINE_ORDER.map(function (key) {
      var result = state.payload.baselines[key];
      return resultCard(result, METHODS[key].label, 'data-method-card="' + key + '"',
        "Python 基線", METHODS[key].detail, state.method === key);
    }).join("");

    var rlResults = PROFILE_ORDER.map(function (key) {
      return state.payload.rl_results[key];
    });
    var outcomeCounts = {};
    rlResults.forEach(function (result) {
      outcomeCounts[outcomeKey(result)] = (outcomeCounts[outcomeKey(result)] || 0) + 1;
    });
    var paretoOutcomes = new Set(state.payload.pareto_rl.map(outcomeKey));
    var rlCards = PROFILE_ORDER.map(function (key) {
      var result = state.payload.rl_results[key];
      var repeated = outcomeCounts[outcomeKey(result)] > 1;
      var tag = repeated ? "RL 結果重合" : (paretoOutcomes.has(outcomeKey(result)) ? "RL 非支配" : "RL 候選");
      return resultCard(result, result.preference.label, 'data-rl-policy="' + key + '"', tag,
        "艘數 / GT / 風險權重：" + weightText(result.preference),
        state.method === "rl" && state.policy === key);
    }).join("");
    byId("paretoGrid").innerHTML = baselines + rlCards;

    document.querySelectorAll("[data-method-card]").forEach(function (button) {
      button.addEventListener("click", function () {
        state.method = button.dataset.methodCard;
        renderPayload();
      });
    });
    document.querySelectorAll("[data-rl-policy]").forEach(function (button) {
      button.addEventListener("click", function () {
        state.method = "rl";
        state.policy = button.dataset.rlPolicy;
        renderPayload();
      });
    });
  }

  function renderKpis(result) {
    var base = state.payload.baselines.fcfs.kpi;
    var kpi = result.kpi;
    byId("evacuatedKpi").textContent = kpi.count + " / " + kpi.total + " 艘";
    byId("evacuatedDelta").textContent = delta(kpi.count, base.count, " 艘 vs FCFS");
    byId("gtKpi").textContent = fmt.format(kpi.gt) + " GT";
    byId("gtDelta").textContent = delta(kpi.gt, base.gt, " GT vs FCFS");
    byId("riskKpi").textContent = kpi.risk + " 點";
    byId("riskDelta").textContent = delta(kpi.risk, base.risk, " 點 vs FCFS");
    byId("remainingKpi").textContent = kpi.remaining_count + " 艘";
    byId("remainingGt").textContent = fmt.format(kpi.remaining_gt) + " GT 留港";
  }

  function renderInsight(result) {
    var rlResults = PROFILE_ORDER.map(function (key) { return state.payload.rl_results[key]; });
    var uniqueOutcomes = new Set(rlResults.map(outcomeKey)).size;
    var label = state.method === "rl" ? PROFILES[state.policy].label + " Python RL" : METHODS[state.method].label;
    var base = state.payload.baselines.fcfs.kpi;
    var text = "Python 後端已執行 3 個規則基線與 4 次 MaskablePPO 推論；";
    if (uniqueOutcomes === 1) {
      text += "四種 RL 偏好得到同一排程，代表此情境下模型尚未呈現偏好條件化分化。";
    } else {
      text += "四種 RL 偏好形成 " + uniqueOutcomes + " 個不同結果，其中 " +
        state.payload.pareto_rl.length + " 個為非支配結果。";
    }
    text += " 所選 " + label + " 相較 FCFS：" +
      delta(result.kpi.count, base.count, " 艘、") +
      delta(result.kpi.gt, base.gt, " GT、") +
      delta(result.kpi.risk, base.risk, " 風險點。");
    byId("insightText").textContent = text;
    byId("alertCopy").textContent = text;
  }

  function renderActionLog(result) {
    var limit = 16;
    var html = result.decisions.slice(0, limit).map(function (decision) {
      if (decision.type === "wait") {
        return '<div class="decision-step wait"><b>A' + decision.action +
          ' · 等待</b><span>T+' + decision.time.toFixed(1) + ' → T+' +
          decision.until.toFixed(1) + '</span><span>Python env 推進下一事件</span></div>';
      }
      var vessel = decision.vessel;
      return '<div class="decision-step"><b>A' + decision.action +
        ' · 派船</b><span>T+' + decision.time.toFixed(1) + '｜' +
        esc(vessel.short_name) + '</span><span>入口 ' + vessel.entrance +
        ' · 拖船 ' + vessel.tugs + '</span><span>同入口可行候選 ' +
        decision.entrance_candidate_count + ' 艘</span></div>';
    }).join("");
    if (result.decisions.length > limit) {
      html += '<div class="decision-step"><b>…</b><span>另有 ' +
        (result.decisions.length - limit) + ' 個 Python 動作</span></div>';
    }
    byId("actionLog").innerHTML = html;
  }

  function renderTimeline(result) {
    var closure = state.payload.scenario.closure_hour;
    var step = Math.max(1, Math.ceil(closure / 4));
    var axis = [];
    for (var hour = 0; hour <= closure; hour += step) axis.push("<span>T+" + hour + "h</span>");
    if ((closure % step) !== 0) axis.push("<span>T+" + closure + "h</span>");
    byId("timeAxis").innerHTML = axis.join("");

    var vessels = allVessels(result);
    byId("timeline").innerHTML = ["1", "2"].map(function (entrance) {
      var entranceVessels = vessels.filter(function (v) { return v.entrance === entrance; });
      var earliestReady = Math.min.apply(null, entranceVessels.map(function (v) { return v.ready_hour; }));
      var gapWidth = Math.min(earliestReady, closure) / closure * 100;
      var readinessGap = gapWidth > 0 ? '<span class="readiness-gap" style="width:' + gapWidth +
        '%" title="此入口尚無已整備船">待整備至 T+' + earliestReady.toFixed(2) + '</span>' : '';
      var bars = result.schedule.filter(function (v) { return v.entrance === entrance; }).map(function (v) {
        var left = v.start_hour / closure * 100;
        var width = (v.finish_hour - v.start_hour) / closure * 100;
        var title = esc(v.name) + "｜" + fmt.format(v.gross_tonnage) + " GT｜拖船 " + v.tugs;
        return '<span class="ship-bar ' + riskClass(v.risk_points) + '" style="left:' + left +
          '%;width:' + width + '%" title="' + title + '">' + esc(v.short_name) + '</span>';
      }).join("");
      return '<div class="timeline-row"><div class="timeline-label"><strong>第 ' + entrance +
        ' 港口</strong><span>入口容量 1</span></div><div class="timeline-track">' + readinessGap + bars +
        '<i class="closure-line" title="停止出港"></i></div></div>';
    }).join("");

    var firstReady = Math.min.apply(null, vessels.filter(function (v) { return v.entrance === "1"; })
      .map(function (v) { return v.ready_hour; }));
    var firstDecisions = result.decisions.filter(function (decision) {
      return decision.type === "dispatch" && decision.vessel.entrance === "1";
    });
    var hasChoice = firstDecisions.some(function (decision) {
      return decision.entrance_candidate_count > 1;
    });
    byId("timelineNote").textContent = "第一港口首艘船在 T+" + firstReady.toFixed(2) +
      " 才整備完成；清單列的是全體船舶，不代表 T+0 已可派。" +
      (hasChoice ? " Python action mask 顯示至少一個時點有多艘同入口候選。" :
        " Python action mask 顯示每個第一港口派船時點都只有 1 艘同入口候選。 ");
  }

  function renderTable(result) {
    var scheduled = result.schedule.map(function (v, index) {
      return Object.assign({}, v, { status: "evacuated", rank: index + 1 });
    });
    var remaining = result.remaining.map(function (v) {
      return Object.assign({}, v, { status: "remaining", rank: null });
    });
    var rows = scheduled.concat(remaining).filter(function (v) {
      return state.filter === "all" || v.status === state.filter;
    });
    var closure = state.payload.scenario.closure_hour;
    byId("vesselTable").innerHTML = rows.map(function (v) {
      var time = v.status === "evacuated" ? "整備 T+" + v.ready_hour.toFixed(1) + "｜出港 T+" +
        v.start_hour.toFixed(1) + " → " + v.finish_hour.toFixed(1) + "h" :
        (v.ready_hour >= closure ? "整備 T+" + v.ready_hour.toFixed(1) + "（封港後）" :
          "整備 T+" + v.ready_hour.toFixed(1) + "｜未排入");
      var tag = v.status === "evacuated" ? '<span class="tag ok">安排出港</span>' :
        '<span class="tag hold">留港覆核</span>';
      return '<tr><td class="ship-rank">' + (v.rank ? String(v.rank).padStart(2, "0") : "—") +
        '</td><td><strong>' + esc(v.name) + '</strong><small>風險 ' + v.risk_points +
        ' 點 · 整備 T+' + v.ready_hour.toFixed(1) + '</small></td><td>第 ' + v.entrance +
        ' 港口</td><td>' + fmt.format(v.gross_tonnage) + '</td><td>' + v.tugs +
        ' 艘</td><td>' + time + '</td><td>' + tag + '</td></tr>';
    }).join("");
  }

  function updateRange(input, output, formatter) {
    var element = byId(input);
    var value = +element.value;
    element.style.setProperty("--range", ((value - +element.min) / (+element.max - +element.min) * 100) + "%");
    byId(output).textContent = formatter(value);
  }

  function syncControls() {
    updateRange("closureInput", "closureOutput", function (v) { return v + " 小時"; });
    updateRange("tugInput", "tugOutput", function (v) { return v + " 艘"; });
    updateRange("pressureInput", "pressureOutput", function (v) { return v.toFixed(1) + "×"; });
    byId("countdownValue").textContent = String(state.closure).padStart(2, "0") + ":00";
    document.querySelectorAll("[data-method]").forEach(function (button) {
      button.classList.toggle("active", button.dataset.method === state.method);
    });
  }

  function renderPayload() {
    var result = selectedResult();
    if (!result) return;
    state.result = result;
    renderKpis(result);
    renderComparison();
    renderInsight(result);
    renderActionLog(result);
    renderTimeline(result);
    renderTable(result);
    var label = state.method === "rl" ? PROFILES[state.policy].label + " · Python RL" : METHODS[state.method].label;
    byId("policyBadge").textContent = label;
    byId("alertTitle").textContent = label + " 已由 Python 完成";
    byId("engineStatus").textContent = "Python RL 已連線";
    byId("scenarioSource").textContent = "船舶：公開資料保留日 " + state.payload.scenario.date;
    byId("modelSource").textContent = "模型：" + state.payload.model.file + " · " +
      fmt.format(state.payload.model.trained_steps) + " steps";
    byId("alertTime").textContent = new Date().toLocaleTimeString("zh-TW", {
      hour: "2-digit", minute: "2-digit"
    });
    syncControls();
  }

  function setBusy(busy) {
    byId("runButton").classList.toggle("running", busy);
    byId("runButton").disabled = busy;
    if (busy) {
      byId("engineStatus").textContent = "Python RL 執行中…";
      byId("alertTitle").textContent = "正在執行 MaskablePPO";
      byId("alertCopy").textContent = "Python 正在建立情境、套用 action mask 並執行四種偏好。";
    }
  }

  async function calculate() {
    var serial = ++state.requestSerial;
    setBusy(true);
    try {
      var response = await fetch("/api/typhoon/schedule", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          closure_hour: state.closure,
          tug_capacity: state.tugs,
          demand_compression: state.pressure,
          preference: "balanced"
        })
      });
      var payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || ("HTTP " + response.status));
      if (serial !== state.requestSerial) return;
      state.payload = payload;
      renderPayload();
    } catch (error) {
      if (serial !== state.requestSerial) return;
      byId("engineStatus").textContent = "Python RL 未連線";
      byId("alertTitle").textContent = "無法執行 Python RL";
      byId("alertCopy").textContent = error.message + "；請用 python -m typhoon.api 啟動本頁。";
    } finally {
      if (serial === state.requestSerial) setBusy(false);
    }
  }

  ["closureInput", "tugInput", "pressureInput"].forEach(function (id) {
    byId(id).addEventListener("input", function () {
      state.closure = +byId("closureInput").value;
      state.tugs = +byId("tugInput").value;
      state.pressure = +byId("pressureInput").value;
      syncControls();
    });
  });
  document.querySelectorAll("[data-method]").forEach(function (button) {
    button.addEventListener("click", function () {
      state.method = button.dataset.method;
      if (state.payload) renderPayload();
    });
  });
  document.querySelectorAll("[data-filter]").forEach(function (button) {
    button.addEventListener("click", function () {
      document.querySelectorAll("[data-filter]").forEach(function (item) { item.classList.remove("active"); });
      button.classList.add("active");
      state.filter = button.dataset.filter;
      if (state.result) renderTable(state.result);
    });
  });
  byId("runButton").addEventListener("click", calculate);
  byId("resetButton").addEventListener("click", function () {
    byId("closureInput").value = 4;
    byId("tugInput").value = 8;
    byId("pressureInput").value = 3;
    state.closure = 4;
    state.tugs = 8;
    state.pressure = 3;
    state.method = "rl";
    state.policy = "balanced";
    syncControls();
    calculate();
  });

  syncControls();
  calculate();
}());
