"use strict";

const classes = { blanket: "Chăn", floor: "Sàn", grass: "Cỏ", rice: "Gạo", sand: "Cát" };
const featureLabels = {
  contrast: "Contrast",
  correlation: "Correlation",
  asm: "ASM",
  energy: "Energy",
  homogeneity: "Homogeneity",
  entropy: "Entropy (nat)",
};
const tabs = [...document.querySelectorAll("[role=tab]")];
const workspace = document.querySelector("#workspace");
const form = document.querySelector("#analysis-form");
const extractionFields = document.querySelector("#extraction-settings");
const angleField = form.elements.angle;
const diagonalOptions = angleField.querySelectorAll("[data-python-only]");
const fileInput = document.querySelector("#file");
const dropzone = document.querySelector("#dropzone");
const preview = document.querySelector("#preview");
const uploadTitle = dropzone.querySelector("strong");
const uploadHint = dropzone.querySelector("small");
const submit = document.querySelector("#run");
const submitText = document.querySelector("#run-text");
const output = document.querySelector("#result");
const title = document.querySelector("#output-title");
const meta = document.querySelector("#output-meta");
const methodNote = document.querySelector("#method-note");
const samples = document.querySelector("#samples");
const decimal = new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 6 });
const percent = new Intl.NumberFormat("vi-VN", { style: "percent", maximumFractionDigits: 2 });

let currentTab = "predict";
let selectedFile = null;
let activeSample = null;
let previewUrl = null;
let activeRequest = null;
let sampleRequest = null;
let fijiConfigured = null;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function sampleUrl(id) {
  return "/samples/" + id + ".png";
}

function placeholder(heading, detail, isError = false) {
  const box = element("div", isError ? "placeholder error" : "placeholder");
  const icon = element("div", "empty-icon");
  icon.setAttribute("aria-hidden", "true");
  icon.append(element("span", "icon chart"));
  box.append(icon, element("strong", "", heading), element("p", "", detail));
  output.replaceChildren(box);
}

function method() {
  return form.elements.method.value;
}

function extractionSettings() {
  const { levels, distance, angle } = form.elements;
  return { levels: levels.valueAsNumber, distance: distance.valueAsNumber, angle: Number(angle.value) };
}

function formatSettings({ levels, distance, angle }) {
  return `G = ${levels} · d = ${distance} · ${angle}°`;
}

function syncSettings() {
  extractionFields.hidden = extractionFields.disabled = currentTab === "predict";
  const usingFiji = method() === "fiji";
  if (usingFiji && angleField.selectedOptions[0]?.hasAttribute("data-python-only")) {
    angleField.value = "0";
  }
  for (const option of diagonalOptions) {
    option.hidden = option.disabled = usingFiji;
  }
}

function refreshForm() {
  submit.disabled = !selectedFile || Boolean(activeRequest);
  submitText.textContent = activeRequest ? "Đang phân tích…" :
    currentTab === "predict" ? "Chạy dự đoán" : "Trích xuất đặc trưng";
  methodNote.hidden = method() !== "fiji" || fijiConfigured !== false;
  output.setAttribute("aria-busy", String(Boolean(activeRequest)));
}

function resetOutput() {
  activeRequest?.abort();
  activeRequest = null;
  const predicting = currentTab === "predict";
  title.textContent = predicting ? "Kết quả dự đoán" : "Đặc trưng GLCM";
  meta.textContent = predicting ? "LDA · 5 lớp" :
    form.checkValidity() ? formatSettings(extractionSettings()) : "Tham số GLCM";
  placeholder("Chưa có kết quả", predicting ?
    "Chọn ảnh và chạy dự đoán để xem lớp kết cấu." :
    "Chọn ảnh và chạy trích xuất để xem các giá trị.");
  refreshForm();
}

function selectTab(tab) {
  if (currentTab === tab.dataset.tab) return;
  currentTab = tab.dataset.tab;
  for (const button of tabs) {
    button.setAttribute("aria-selected", String(button === tab));
    button.tabIndex = button === tab ? 0 : -1;
  }
  workspace.setAttribute("aria-labelledby", tab.id);
  syncSettings();
  resetOutput();
}

for (const tab of tabs) {
  tab.addEventListener("click", () => selectTab(tab));
  tab.addEventListener("keydown", (event) => {
    const targets = { ArrowLeft: tabs[1 - tabs.indexOf(tab)], ArrowRight: tabs[1 - tabs.indexOf(tab)], Home: tabs[0], End: tabs[1] };
    const next = targets[event.key];
    if (!next) return;
    event.preventDefault();
    selectTab(next);
    next.focus();
  });
}

function selectFile(file, sample = null) {
  if (!file) return;
  if (!file.size || file.size > 12 * 1024 * 1024) {
    resetOutput();
    placeholder("Kích thước ảnh không hợp lệ", "Chọn tệp ảnh có dung lượng tối đa 12 MB.", true);
    return;
  }
  selectedFile = file;
  activeSample = sample;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  preview.hidden = true;
  dropzone.classList.remove("has-preview");
  preview.src = previewUrl;
  uploadTitle.textContent = file.name;
  uploadHint.textContent = decimal.format(file.size / 1024) + " KB · Nhấn để đổi ảnh";
  for (const option of samples.children) {
    option.setAttribute("aria-pressed", String(option.dataset.sample === activeSample));
  }
  resetOutput();
}

for (const [id, label] of Object.entries(classes)) {
  const button = element("button", "sample");
  button.type = "button";
  button.dataset.sample = id;
  button.setAttribute("aria-pressed", "false");
  const image = element("img");
  image.src = sampleUrl(id);
  image.alt = "";
  const caption = element("span");
  caption.append(element("strong", "", label), element("small", "", id + "1"));
  button.append(image, caption);
  samples.append(button);
  button.addEventListener("click", async () => {
    sampleRequest?.abort();
    const request = new AbortController();
    sampleRequest = request;
    try {
      const response = await fetch(sampleUrl(id), { signal: request.signal });
      if (!response.ok) throw new Error("Không tải được ảnh mẫu.");
      const blob = await response.blob();
      if (sampleRequest !== request) return;
      selectFile(new File([blob], id + "1-d-p001.png", { type: blob.type }), id);
      dropzone.focus();
    } catch (error) {
      if (error.name === "AbortError" || sampleRequest !== request) return;
      resetOutput();
      placeholder("Không tải được ảnh mẫu", "Hãy thử lại hoặc chọn ảnh trên máy.", true);
    } finally {
      if (sampleRequest === request) sampleRequest = null;
    }
  });
}

preview.addEventListener("load", () => {
  preview.hidden = false;
  dropzone.classList.add("has-preview");
});
preview.addEventListener("error", () => {
  preview.hidden = true;
  dropzone.classList.remove("has-preview");
});
dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  sampleRequest?.abort();
  selectFile(fileInput.files[0]);
  fileInput.value = "";
});
for (const type of ["dragenter", "dragover"]) {
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
}
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragging"));
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragging");
  sampleRequest?.abort();
  selectFile(event.dataTransfer.files[0]);
});
form.addEventListener("change", (event) => {
  if (event.target.name === "method") {
    syncSettings();
    resetOutput();
  }
});
extractionFields.addEventListener("input", resetOutput);

function appendMeasurement(list, name, value, formatter = decimal) {
  const row = element("div");
  row.append(element("dt", "", name), element("dd", "", formatter.format(value)));
  list.append(row);
}

function showPrediction(result) {
  const first = result.ranking[0];
  const winner = element("div", "winner");
  const image = element("img");
  image.src = sampleUrl(first.id);
  image.alt = "";
  const label = element("div");
  label.append(element("small", "", "Lớp dự đoán"), element("strong", "", first.label));
  winner.append(image, label);
  const scores = element("dl", "scores");
  for (const { label: name, probability } of result.ranking) {
    appendMeasurement(scores, name, probability, percent);
  }
  output.replaceChildren(winner, scores, element("p", "", "Xác suất ước tính theo lớp"));
}

function matrixDetails({ matrix, levels, distance, angle }) {
  const details = element("details", "matrix-info");
  const summary = element("summary", "", `Ma trận xác suất GLCM ${levels} × ${levels}`);
  const body = element("div", "matrix-body");
  const canvas = element("canvas");
  canvas.width = matrix.length;
  canvas.height = matrix.length;
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", "Bản đồ nhiệt của ma trận xác suất GLCM");
  const context = canvas.getContext("2d");
  const peak = matrix.reduce((max, row) => Math.max(max, ...row), 0);
  matrix.forEach((row, i) => row.forEach((value, j) => {
    const intensity = value / peak;
    context.fillStyle = `rgb(${Math.round(26 + 182 * intensity)} ${Math.round(47 + 192 * intensity)} ${Math.round(35 + 96 * intensity)})`;
    context.fillRect(j, i, 1, 1);
  }));
  const caption = element("p", "", "Hàng i, cột j · Màu sáng biểu thị xác suất cao hơn.");
  const download = element("a", "matrix-download", "Tải ma trận CSV");
  download.href = "data:text/csv;charset=utf-8," + encodeURIComponent(matrix.map((row) => row.join(",")).join("\n"));
  download.download = `ma-tran-glcm-G${levels}-d${distance}-a${angle}.csv`;
  caption.append(document.createElement("br"), download);
  body.append(canvas, caption);
  details.append(summary, body);
  return details;
}

function showFeatures(result) {
  const list = element("dl", "features");
  for (const [key, label] of Object.entries(featureLabels)) {
    appendMeasurement(list, label, result.features[key]);
  }
  output.replaceChildren(element("p", "", `Ảnh xám · ${result.width} × ${result.height} px`), list);
  if (result.matrix) output.append(matrixDetails(result));
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selectedFile || activeRequest) return;
  const request = new AbortController();
  const engine = method();
  const mode = currentTab;
  const query = mode === "extract" ? "?" + new URLSearchParams(extractionSettings()) : "";
  activeRequest = request;
  refreshForm();
  placeholder("Đang phân tích ảnh", engine === "fiji" ? "Đang chạy plugin GLCM Texture trong Fiji." : "Đang tính GLCM bằng Python.");
  try {
    const response = await fetch(`/api/${mode}/${engine}${query}`, {
      method: "POST",
      body: selectedFile,
      signal: request.signal,
    });
    const result = await response.json();
    if (activeRequest !== request) return;
    if (!response.ok) throw new Error(result.error);
    const name = engine === "python" ? "Python" : "Fiji";
    meta.textContent = mode === "predict" ? name + " · LDA" : name + " · " + formatSettings(result);
    if (mode === "predict") showPrediction(result);
    else showFeatures(result);
  } catch (error) {
    if (error.name !== "AbortError" && activeRequest === request) {
      placeholder("Không thể phân tích ảnh", error.message || "Hãy thử lại.", true);
    }
  } finally {
    if (activeRequest === request) {
      activeRequest = null;
      refreshForm();
    }
  }
});

fetch("/api/status")
  .then((response) => response.json())
  .then((status) => {
    fijiConfigured = status.fijiConfigured;
    refreshForm();
  })
  .catch(() => {});
