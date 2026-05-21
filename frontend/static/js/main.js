const els = {
  root: document.documentElement,
  form: document.getElementById("captionForm"),
  imageInput: document.getElementById("imageInput"),
  dropzone: document.getElementById("dropzone"),
  dropPrompt: document.getElementById("dropPrompt"),
  uploadPreview: document.getElementById("uploadPreview"),
  fileMeta: document.getElementById("fileMeta"),
  fileName: document.getElementById("fileName"),
  fileSize: document.getElementById("fileSize"),
  clearImage: document.getElementById("clearImage"),
  beamWidth: document.getElementById("beamWidth"),
  beamValue: document.getElementById("beamValue"),
  modeLabel: document.getElementById("modeLabel"),
  submitBtn: document.getElementById("submitBtn"),
  submitSpinner: document.getElementById("submitSpinner"),
  submitText: document.getElementById("submitText"),
  emptyState: document.getElementById("emptyState"),
  loadingState: document.getElementById("loadingState"),
  slowNotice: document.getElementById("slowNotice"),
  resultState: document.getElementById("resultState"),
  resultImage: document.getElementById("resultImage"),
  resultFileBadge: document.getElementById("resultFileBadge"),
  captionText: document.getElementById("captionText"),
  timePill: document.getElementById("timePill"),
  modePill: document.getElementById("modePill"),
  copyCaption: document.getElementById("copyCaption"),
  attentionSection: document.getElementById("attentionSection"),
  attentionCanvas: document.getElementById("attentionCanvas"),
  tokenList: document.getElementById("tokenList"),
  saveHeatmap: document.getElementById("saveHeatmap"),
  modelArchitecture: document.getElementById("modelArchitecture"),
  modelVocab: document.getElementById("modelVocab"),
  modelMaxLength: document.getElementById("modelMaxLength"),
  modelBeam: document.getElementById("modelBeam"),
  historyList: document.getElementById("historyList"),
  historyCount: document.getElementById("historyCount"),
  clearHistory: document.getElementById("clearHistory"),
  toastRoot: document.getElementById("toastRoot"),
  themeToggle: document.getElementById("themeToggle"),
  themeIcon: document.getElementById("themeIcon"),
  themeText: document.getElementById("themeText"),
};

let selectedFile = null;
let currentResult = null;
let loading = false;
let slowTimer = null;

function toast(message, type = "error") {
  const node = document.createElement("div");
  node.className = `toast ${type}`;
  node.textContent = message;
  els.toastRoot.appendChild(node);
  setTimeout(() => node.remove(), 4000);
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function setTheme(theme) {
  els.root.dataset.theme = theme;
  els.root.classList.toggle("dark", theme === "dark");
  els.root.classList.toggle("light", theme === "light");
  localStorage.setItem("theme", theme);
  const isDark = theme === "dark";
  els.themeIcon.textContent = isDark ? "light_mode" : "dark_mode";
  els.themeText.textContent = isDark ? "Tối" : "Sáng";
}

function updateBeamLabel() {
  const value = Number(els.beamWidth.value);
  const min = Number(els.beamWidth.min || 1);
  const max = Number(els.beamWidth.max || 5);
  const percent = ((value - min) / (max - min)) * 100;
  els.beamValue.textContent = value;
  els.modeLabel.textContent = value <= 1 ? "Greedy Search" : `Beam Search (width=${value})`;
  els.beamWidth.style.setProperty("--beam-progress", `${percent}%`);
}

function validateFile(file) {
  const validTypes = ["image/jpeg", "image/png", "image/webp"];
  const validExt = /\.(jpe?g|png|webp)$/i.test(file.name);
  if (!validTypes.includes(file.type) && !validExt) {
    toast("Chỉ hỗ trợ ảnh JPG, PNG, WEBP");
    return false;
  }
  if (file.size > 10 * 1024 * 1024) {
    toast("Ảnh quá lớn. Vui lòng chọn ảnh dưới 10MB");
    return false;
  }
  return true;
}

function setFile(file) {
  if (!file || !validateFile(file) || loading) return;
  selectedFile = file;
  const url = URL.createObjectURL(file);
  els.uploadPreview.src = url;
  els.uploadPreview.classList.remove("hidden");
  els.dropPrompt.classList.add("hidden");
  els.fileMeta.classList.remove("hidden");
  els.fileMeta.classList.add("flex");
  els.fileName.textContent = file.name;
  els.fileSize.textContent = formatBytes(file.size);
  els.submitBtn.disabled = false;
}

function clearFile() {
  selectedFile = null;
  els.imageInput.value = "";
  els.uploadPreview.removeAttribute("src");
  els.uploadPreview.classList.add("hidden");
  els.dropPrompt.classList.remove("hidden");
  els.fileMeta.classList.add("hidden");
  els.fileMeta.classList.remove("flex");
  els.submitBtn.disabled = true;
}

function setLoading(isLoading) {
  loading = isLoading;
  els.submitBtn.disabled = isLoading || !selectedFile;
  els.imageInput.disabled = isLoading;
  els.submitSpinner.classList.toggle("hidden", !isLoading);
  els.submitText.textContent = isLoading ? "Đang xử lý..." : "Sinh Mô tả Ảnh";
  els.dropzone.classList.toggle("is-disabled", isLoading);
  els.slowNotice.classList.add("hidden");
  clearTimeout(slowTimer);
  if (isLoading) {
    els.emptyState.classList.add("hidden");
    els.resultState.classList.add("hidden");
    els.loadingState.classList.remove("hidden");
    slowTimer = setTimeout(() => els.slowNotice.classList.remove("hidden"), 15000);
  } else {
    els.loadingState.classList.add("hidden");
  }
}

function typeCaption(text) {
  els.captionText.textContent = "";
  const words = text.split(/\s+/);
  let index = 0;
  const timer = setInterval(() => {
    els.captionText.textContent = words.slice(0, index + 1).join(" ");
    index += 1;
    if (index >= words.length) clearInterval(timer);
  }, 55);
}

function modeText(result) {
  return result.mode === "greedy" ? "Greedy Search" : `Beam Search (width=${result.beam_width})`;
}

function renderAttention(result) {
  const weights = result.attention_weights;
  const tokens = result.caption.replace(/[.!?]$/g, "").split(/\s+/).filter(Boolean);
  els.tokenList.innerHTML = "";

  if (!weights || !Array.isArray(weights) || !weights.length) {
    els.attentionSection.classList.add("hidden");
    els.attentionCanvas.classList.add("hidden");
    tokens.forEach((token) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = token;
      els.tokenList.appendChild(button);
    });
    return;
  }

  els.attentionSection.classList.remove("hidden");
  els.attentionCanvas.classList.remove("hidden");

  const drawIndex = (index) => {
    const tokenWeights = Array.isArray(weights[index]) ? weights[index] : weights[0];
    AttentionMap.draw(els.attentionCanvas, els.resultImage, tokenWeights);
    [...els.tokenList.children].forEach((button, i) => button.classList.toggle("is-active", i === index));
  };

  tokens.forEach((token, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = token;
    button.addEventListener("click", () => drawIndex(index));
    els.tokenList.appendChild(button);
  });

  els.resultImage.onload = () => drawIndex(0);
  if (els.resultImage.complete) drawIndex(0);
}

function renderResult(result, saveHistory = true) {
  currentResult = result;
  els.emptyState.classList.add("hidden");
  els.loadingState.classList.add("hidden");
  els.resultState.classList.remove("hidden");
  els.resultState.classList.add("flex");
  els.resultImage.src = result.image_url;
  els.resultFileBadge.textContent = result.filename || "Ảnh đã upload";
  els.timePill.textContent = `Xử lý trong ${Number(result.process_time || 0).toFixed(2)} giây`;
  els.modePill.textContent = modeText(result);
  els.modelArchitecture.textContent = result.model?.architecture || "EfficientNetB0 + LSTM";
  els.modelVocab.textContent = result.model?.vocab_size || "-";
  els.modelMaxLength.textContent = result.model?.max_length || "-";
  els.modelBeam.textContent = result.model?.beam_width || result.beam_width || "-";
  typeCaption(result.caption || "");
  renderAttention(result);

  if (saveHistory) {
    DemoHistory.add({
      ...result,
      saved_at: Date.now(),
    });
    renderHistory();
  }
}

function truncate(text, size = 40) {
  return text.length > size ? `${text.slice(0, size - 3)}...` : text;
}

function renderHistory() {
  const items = DemoHistory.load();
  els.historyCount.textContent = `${items.length} ảnh đã test`;
  els.historyList.innerHTML = "";
  if (!items.length) {
    const empty = document.createElement("p");
    empty.className = "text-sm text-muted";
    empty.textContent = "Chưa có ảnh nào trong lịch sử.";
    els.historyList.appendChild(empty);
    return;
  }
  items.forEach((item) => {
    const button = document.createElement("button");
    button.className = "history-item group";
    button.type = "button";
    button.innerHTML = `
      <div class="relative h-32 w-full overflow-hidden">
        <img src="${item.image_url}" alt="">
        <div class="absolute inset-0 bg-gradient-to-t from-black/45 to-transparent"></div>
      </div>
      <div class="relative z-10 bg-white p-4 dark:bg-slate-950">
        <p>${truncate(item.caption || "")}</p>
      </div>`;
    button.addEventListener("click", () => renderResult(item, false));
    els.historyList.appendChild(button);
  });
}

async function submitCaption(event) {
  event.preventDefault();
  if (!selectedFile || loading) return;

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("beam_width", els.beamWidth.value);
  setLoading(true);

  try {
    const response = await fetch("/caption", { method: "POST", body: formData });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.error || (response.status >= 500 ? "Đã xảy ra lỗi. Vui lòng thử lại" : "Không thể xử lý ảnh"));
    }
    renderResult(payload);
  } catch (error) {
    const message = error instanceof TypeError
      ? "Không thể kết nối server. Kiểm tra Flask đang chạy chưa"
      : error.message;
    toast(message || "Đã xảy ra lỗi. Vui lòng thử lại");
    els.emptyState.classList.remove("hidden");
  } finally {
    setLoading(false);
  }
}

els.themeToggle.addEventListener("click", () => {
  setTheme(els.root.dataset.theme === "dark" ? "light" : "dark");
});

els.beamWidth.addEventListener("input", updateBeamLabel);
els.imageInput.addEventListener("change", () => setFile(els.imageInput.files[0]));
els.clearImage.addEventListener("click", clearFile);
els.form.addEventListener("submit", submitCaption);
els.copyCaption.addEventListener("click", async () => {
  if (!currentResult?.caption) return;
  await navigator.clipboard.writeText(currentResult.caption);
  toast("Đã sao chép caption", "info");
});
els.saveHeatmap.addEventListener("click", () => {
  const link = document.createElement("a");
  link.download = "attention-heatmap.png";
  link.href = els.attentionCanvas.toDataURL("image/png");
  link.click();
});
els.clearHistory.addEventListener("click", () => {
  DemoHistory.clear();
  renderHistory();
});

["dragenter", "dragover"].forEach((eventName) => {
  els.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    if (!loading) els.dropzone.classList.add("is-dragging");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  els.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropzone.classList.remove("is-dragging");
  });
});
els.dropzone.addEventListener("drop", (event) => {
  if (!loading) setFile(event.dataTransfer.files[0]);
});

setTheme(localStorage.getItem("theme") || "light");
updateBeamLabel();
renderHistory();
