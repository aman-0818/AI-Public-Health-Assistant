const form = document.getElementById("risk-form");
const statusEl = document.getElementById("status");
const submitBtn = document.getElementById("submit-btn");
const useLocationBtn = document.getElementById("use-location");

const emptyState = document.getElementById("empty-state");
const resultContent = document.getElementById("result-content");
const riskCards = document.getElementById("risk-cards");
const densityLevelEl = document.getElementById("density-level");
const nearbyTotalEl = document.getElementById("nearby-total");
const disclaimerTextEl = document.getElementById("disclaimer-text");

const advicePlaceholder = document.getElementById("advice-placeholder");
const adviceContent = document.getElementById("advice-content");

function setStatus(message, isError = false) {
  statusEl.textContent = message || "";
  statusEl.classList.toggle("error", Boolean(isError));
}

function cleanAdviceText(text) {
  if (!text) return "";
  return String(text)
    .replace(/\r\n/g, "\n")
    .replace(/^\s*---+\s*$/gm, "")
    .replace(/\*\*/g, "")
    .replace(/__/g, "")
    .replace(/`/g, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function parseAdviceSections(text) {
  const lines = String(text || "").split("\n");
  const headerRegex = /^\s*(\d+)[.)]\s+(.+?)\s*$/;
  const knownTitleRegex = /^\s*(Quick Risk Summary|Most Important Precautions Today|Early Symptoms To Monitor.*|When To Test Or Visit Hospital|Safety Notes)\s*$/i;
  const sections = [];
  let current = null;
  let nextAutoIndex = 1;

  lines.forEach((line) => {
    const match = line.match(headerRegex);
    if (match) {
      if (current) {
        current.body = current.body.join("\n").trim();
        sections.push(current);
      }
      nextAutoIndex = Math.max(nextAutoIndex, Number(match[1]) + 1);
      current = {
        index: Number(match[1]),
        title: match[2].trim(),
        body: [],
      };
      return;
    }

    const knownMatch = line.match(knownTitleRegex);
    if (knownMatch) {
      if (current) {
        current.body = current.body.join("\n").trim();
        sections.push(current);
      }
      current = {
        index: nextAutoIndex,
        title: knownMatch[1].trim(),
        body: [],
      };
      nextAutoIndex += 1;
      return;
    }

    if (current) {
      current.body.push(line);
    }
  });

  if (current) {
    current.body = current.body.join("\n").trim();
    sections.push(current);
  }

  return sections;
}

function createAdviceCard(section) {
  const card = document.createElement("section");
  card.className = "advice-card";
  card.innerHTML = `
    <div class="advice-card__head">
      <span class="advice-card__index">${section.index}</span>
      <h3>${section.title}</h3>
    </div>
    <p>${section.body || "No details provided."}</p>
  `;
  return card;
}

function renderAdvice(adviceResponse) {
  const baseAdvice = cleanAdviceText(adviceResponse.advice) || "No guidance returned.";
  const sections = parseAdviceSections(baseAdvice);

  adviceContent.innerHTML = "";
  adviceContent.classList.remove("advice--plain");

  if (sections.length >= 2) {
    const grid = document.createElement("div");
    grid.className = "advice-grid";
    sections.sort((a, b) => a.index - b.index).forEach((section) => {
      grid.appendChild(createAdviceCard(section));
    });
    adviceContent.appendChild(grid);
  } else {
    adviceContent.classList.add("advice--plain");
    adviceContent.textContent = baseAdvice;
  }

  if (!adviceResponse.llm_used && adviceResponse.llm_error?.hint) {
    const note = document.createElement("p");
    note.className = "advice-note";
    note.textContent = `Troubleshooting hint: ${adviceResponse.llm_error.hint}`;
    adviceContent.appendChild(note);
  }
}

function percent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function orderedRiskEntries(riskObj) {
  const entries = Object.entries(riskObj || {});
  entries.sort((a, b) => Number(b[1]) - Number(a[1]));
  return entries;
}

function createRiskCard(name, value) {
  const card = document.createElement("div");
  card.className = "risk-card";
  card.innerHTML = `
    <div class="risk-head">
      <span class="risk-name">${name}</span>
      <span class="risk-score">${percent(value)}</span>
    </div>
    <div class="bar-wrap">
      <div class="bar-fill" style="width: 0%"></div>
    </div>
  `;

  const fill = card.querySelector(".bar-fill");
  requestAnimationFrame(() => {
    fill.style.width = `${Math.max(0, Math.min(100, Number(value) * 100))}%`;
  });

  return card;
}

function renderPrediction(payload) {
  const prediction = payload.prediction || {};
  const balancedRisk = prediction.balanced_risk || {};
  const nearby = prediction.nearby_cases_25km || {};
  const entries = orderedRiskEntries(balancedRisk);

  riskCards.innerHTML = "";
  entries.forEach(([disease, score]) => {
    riskCards.appendChild(createRiskCard(disease, score));
  });

  const nearbyTotal = Object.values(nearby).reduce(
    (sum, value) => sum + Number(value || 0),
    0
  );

  densityLevelEl.textContent = (prediction.location_density_level || "-").replace("_", " ");
  nearbyTotalEl.textContent = String(nearbyTotal);
  disclaimerTextEl.textContent = prediction.disclaimer || "";

  emptyState.classList.add("hidden");
  resultContent.classList.remove("hidden");
}

async function fetchAdvice(payload) {
  const response = await fetch("/advice", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await response.json();
  if (!response.ok) {
    const detail = data.detail ? ` (${data.detail})` : "";
    throw new Error((data.error || "Advice request failed") + detail);
  }
  return data;
}

async function predictRisk(input) {
  const response = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Prediction failed");
  }
  return data;
}

function formPayload() {
  const payload = {
    name: document.getElementById("name").value.trim(),
    age: Number(document.getElementById("age").value),
    gender: document.getElementById("gender").value,
    latitude: Number(document.getElementById("latitude").value),
    longitude: Number(document.getElementById("longitude").value),
  };
  return payload;
}

function validatePayload(payload) {
  if (!Number.isFinite(payload.age) || payload.age < 0 || payload.age > 120) {
    throw new Error("Enter a valid age (0-120).");
  }
  if (!Number.isFinite(payload.latitude) || payload.latitude < -90 || payload.latitude > 90) {
    throw new Error("Latitude must be between -90 and 90.");
  }
  if (!Number.isFinite(payload.longitude) || payload.longitude < -180 || payload.longitude > 180) {
    throw new Error("Longitude must be between -180 and 180.");
  }
}

async function handleSubmit(event) {
  event.preventDefault();
  adviceContent.classList.add("hidden");
  advicePlaceholder.classList.remove("hidden");
  advicePlaceholder.textContent = "Generating guidance...";

  let payload;
  try {
    payload = formPayload();
    validatePayload(payload);
  } catch (error) {
    setStatus(error.message || "Invalid input", true);
    return;
  }

  submitBtn.disabled = true;
  setStatus("Analyzing local case patterns...");

  try {
    const predictionResponse = await predictRisk(payload);
    renderPrediction(predictionResponse);

    setStatus("Risk profile ready. Asking Gemini for guidance...");
    const adviceResponse = await fetchAdvice(predictionResponse);

    renderAdvice(adviceResponse);
    adviceContent.classList.remove("hidden");
    advicePlaceholder.classList.add("hidden");

    if (adviceResponse.llm_used) {
      setStatus("Complete. Gemini guidance is included.");
    } else {
      setStatus("Prediction complete. Showing fallback guidance.");
    }
  } catch (error) {
    setStatus(error.message || "Request failed", true);
    advicePlaceholder.textContent = "Unable to generate advice right now.";
  } finally {
    submitBtn.disabled = false;
  }
}

function handleUseLocation() {
  if (!navigator.geolocation) {
    setStatus("Geolocation is not supported in this browser.", true);
    return;
  }
  setStatus("Reading your current location...");
  useLocationBtn.disabled = true;

  navigator.geolocation.getCurrentPosition(
    (pos) => {
      document.getElementById("latitude").value = pos.coords.latitude.toFixed(6);
      document.getElementById("longitude").value = pos.coords.longitude.toFixed(6);
      setStatus("Location filled. You can now analyze risk.");
      useLocationBtn.disabled = false;
    },
    (err) => {
      setStatus(`Location unavailable: ${err.message}`, true);
      useLocationBtn.disabled = false;
    },
    { enableHighAccuracy: true, timeout: 12000, maximumAge: 0 }
  );
}

form.addEventListener("submit", handleSubmit);
useLocationBtn.addEventListener("click", handleUseLocation);
