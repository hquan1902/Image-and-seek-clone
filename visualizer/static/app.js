const $ = (id) => document.getElementById(id);
let cases = [];
const fmtRank = (rank) => rank > 50 ? ">50" : `#${rank}`;
const signed = (value) => `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;

async function json(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status}: ${url}`);
  return response.json();
}

function renderSummary(data) {
  $("dataset-meta").textContent = `${data.dataset} · ${data.queries.toLocaleString("vi-VN")} queries · ${data.gallery_images.toLocaleString("vi-VN")} ảnh gallery`;
  $("metrics").innerHTML = data.metrics.map(metric => `
    <div class="metric ${metric.delta > 0 ? "positive" : ""}">
      <span>${metric.name.replace("group_", "Group ")}</span>
      <div><small>${metric.baseline.toFixed(2)}%</small><b>${metric.improved.toFixed(2)}%</b></div>
      <em>${signed(metric.delta)} điểm %</em>
    </div>`).join("");
  $("case-counts").innerHTML = `<span class="count-good">${data.case_counts.improved} cải thiện</span><span class="count-bad">${data.case_counts.regressed} suy giảm</span><span>${data.case_counts.unchanged} không đổi</span>`;
}

async function loadCases() {
  const status = $("status-filter").value;
  const q = encodeURIComponent($("search").value.trim());
  const data = await json(`/api/cases?status=${status}&q=${q}`);
  cases = data.cases;
  $("case-select").innerHTML = cases.map((item, index) => `<option value="${item.pair_id}">${index + 1}. [${fmtRank(item.baseline_rank)}→${fmtRank(item.improved_rank)}] ${item.caption}</option>`).join("");
  if (cases.length) await loadCase(cases[0].pair_id); else clearCase();
}

function resultCard(item) {
  const classes = ["result-item", item.is_target ? "is-target" : "", item.is_group_member && !item.is_target ? "is-group" : ""].join(" ");
  const tag = item.is_target ? "GROUND TRUTH" : item.is_group_member ? "CÙNG NHÓM" : "";
  return `<figure class="${classes}"><div class="rank">${item.rank}</div><img src="${item.image_url}" loading="lazy" alt="Kết quả hạng ${item.rank}"><figcaption><code>${item.image_id}</code>${tag ? `<b>${tag}</b>` : ""}</figcaption></figure>`;
}

async function loadCase(pairId) {
  const data = await json(`/api/case/${pairId}`);
  $("caption").textContent = `“${data.caption}”`;
  $("caption-copy").textContent = data.caption;
  $("reference-image").src = data.reference_url;
  $("target-image").src = data.target_url;
  $("reference-id").textContent = data.reference;
  $("target-id").textContent = data.target;
  $("baseline-rank").textContent = `GT ${fmtRank(data.baseline_rank)}`;
  $("improved-rank").textContent = `GT ${fmtRank(data.improved_rank)}`;
  const gain = data.rank_gain;
  $("rank-summary").className = `rank-summary ${gain > 0 ? "gain" : gain < 0 ? "loss" : ""}`;
  $("rank-summary").innerHTML = `<small>Thay đổi thứ hạng ground truth</small><b>${fmtRank(data.baseline_rank)} → ${fmtRank(data.improved_rank)}</b><span>${gain > 0 ? `tăng ${gain} bậc` : gain < 0 ? `giảm ${Math.abs(gain)} bậc` : "không đổi"}</span>`;
  $("baseline-grid").innerHTML = data.baseline_results.map(resultCard).join("");
  $("improved-grid").innerHTML = data.improved_results.map(resultCard).join("");
  $("query-card").style.display = "block";
}

function clearCase() {
  $("case-select").innerHTML = "<option>Không có truy vấn phù hợp</option>";
  $("query-card").style.display = "none";
  $("baseline-grid").innerHTML = "";
  $("improved-grid").innerHTML = "";
}

let timer;
$("status-filter").addEventListener("change", loadCases);
$("case-select").addEventListener("change", event => loadCase(event.target.value));
$("search").addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(loadCases, 250); });
Promise.all([json("/api/summary"), loadCases()]).then(([summary]) => renderSummary(summary)).catch(error => document.body.innerHTML = `<pre class="fatal">Không thể mở demo: ${error.message}</pre>`);
