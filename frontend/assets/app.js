/**
 * frontend/assets/app.js
 * ----------------------
 * Client-side JavaScript for the Disease Risk Sentinel app.
 *
 * FLOW:
 *   1. User fills form (name, age, gender, lat/lon) and clicks "Analyze Risk"
 *   2. handleSubmit() sends POST /predict → gets disease risk percentages
 *   3. renderPrediction() draws animated risk bars on screen
 *   4. fetchAdvice() sends POST /advice → gets Gemini-generated health guidance
 *   5. renderAdvice() parses the guidance into cards and displays them
 *
 * "Use My Current Location" button calls the browser Geolocation API
 * to auto-fill latitude and longitude fields.
 */

// ── DOM element references ──────────────────────────────────────────────────
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

// ── Utility functions ───────────────────────────────────────────────────────

/** Show a status message below the form. Pass isError=true to style it red. */
function setStatus(message, isError = false) {
  statusEl.textContent = message || "";
  statusEl.classList.toggle("error", Boolean(isError));
}

/** Strip markdown symbols (**, __, ```) from Gemini response text. */
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

/**
 * Parse Gemini advice text into structured sections.
 * Looks for numbered headers like "1. Quick Risk Summary" or known title strings.
 * Returns an array of { index, title, body } objects.
 */
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

/** Build a single advice card DOM element from a parsed section object. */
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

/**
 * Render the Gemini advice response into the advice panel.
 * If the response has 2+ sections → shows cards in a grid.
 * Otherwise → shows plain text paragraph.
 */
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

// ── Risk card rendering ─────────────────────────────────────────────────────

/** Convert a 0.0–1.0 float value to a "XX.X%" display string. */
function percent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function orderedRiskEntries(riskObj) {
  const entries = Object.entries(riskObj || {});
  entries.sort((a, b) => Number(b[1]) - Number(a[1]));
  return entries;
}

/**
 * Build a risk card DOM element with an animated progress bar.
 * Bar width animates from 0% → actual value using requestAnimationFrame.
 */
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

/** Render disease risk cards and metadata (density level, nearby cases count) from /predict response. */
function renderPrediction(payload) {
  const prediction = payload.prediction || {};
  const balancedRisk = prediction.balanced_risk || {};
  const nearby = prediction.nearby_cases_25km || {};
  const entries = orderedRiskEntries(balancedRisk);

  // Remove any existing out-of-region warning before re-rendering
  const existingWarning = document.getElementById("out-of-region-warning");
  if (existingWarning) existingWarning.remove();

  // If location is outside training region — show a prominent warning banner
  if (prediction.out_of_region) {
    const warning = document.createElement("div");
    warning.id = "out-of-region-warning";
    warning.className = "out-of-region-warning";
    warning.innerHTML = `
      <strong>⚠️ Location Outside Data Region</strong>
      <p>${prediction.out_of_region_message}</p>
      <p>Predictions below are <strong>not reliable</strong> for your area. This model only covers <strong>Tamil Nadu, India</strong>.</p>
    `;
    // Insert warning before the risk cards
    resultContent.insertBefore(warning, resultContent.firstChild);
  }

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

// ── API calls ───────────────────────────────────────────────────────────────

/** POST to /advice endpoint with the full prediction payload. Returns Gemini guidance text. */
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

/** POST to /predict endpoint with user's {name, age, gender, latitude, longitude}. Returns risk percentages. */
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

// ── Form handling ───────────────────────────────────────────────────────────

/** Read all form field values and return them as a single payload object. */
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

/** Validate the form payload. Throws an Error with a user-friendly message if any required field is missing or invalid. */
function validatePayload(payload) {
  if (!payload.name || payload.name.trim() === "") {
    throw new Error("Name is required.");
  }
  if (!Number.isFinite(payload.age) || payload.age < 0 || payload.age > 120) {
    throw new Error("Enter a valid age (0-120).");
  }
  if (!payload.gender || payload.gender === "") {
    throw new Error("Please select a gender.");
  }
  if (!Number.isFinite(payload.latitude) || payload.latitude < -90 || payload.latitude > 90) {
    throw new Error("Latitude is required and must be between -90 and 90.");
  }
  if (!Number.isFinite(payload.longitude) || payload.longitude < -180 || payload.longitude > 180) {
    throw new Error("Longitude is required and must be between -180 and 180.");
  }
}

/**
 * Main form submit handler.
 * Step 1 → call /predict to get risk percentages and render risk bars.
 * Step 2 → call /advice to get Gemini guidance and render advice cards.
 */
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

    // If out of region, skip Gemini advice (would be misleading)
    if (predictionResponse.prediction && predictionResponse.prediction.out_of_region) {
      advicePlaceholder.textContent = "AI guidance is not available for locations outside Tamil Nadu, India.";
      submitBtn.disabled = false;
      return;
    }

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

/**
 * "Use My Current Location" button handler.
 * Uses the browser Geolocation API to auto-fill latitude and longitude fields.
 */
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
