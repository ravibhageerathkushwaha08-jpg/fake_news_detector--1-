const headlineInput = document.getElementById("headline");
const analyzeBtn = document.getElementById("analyzeBtn");
const stampBadge = document.getElementById("stampBadge");
const stampText = document.getElementById("stampText");
const confidenceText = document.getElementById("confidenceText");
const hint = document.getElementById("hint");

async function analyze() {
  const text = headlineInput.value.trim();

  if (!text) {
    hint.textContent = "Type a headline first.";
    headlineInput.focus();
    return;
  }

  // Reset stamp so the "stamping down" animation can replay
  stampBadge.classList.add("verdict__stamp--hidden");
  stampBadge.classList.remove("verdict__stamp--real", "verdict__stamp--fake");
  confidenceText.textContent = "";

  analyzeBtn.disabled = true;
  hint.textContent = "Reviewing...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await response.json();

    if (!response.ok) {
      hint.textContent = data.error || "Something went wrong.";
      analyzeBtn.disabled = false;
      return;
    }

    stampText.textContent = data.label === "FAKE" ? "FAKE" : "VERIFIED";
    stampBadge.classList.add(
      data.label === "FAKE" ? "verdict__stamp--fake" : "verdict__stamp--real"
    );
    confidenceText.textContent = `Confidence: ${data.confidence}%`;
    hint.textContent = "Press the stamp to get a verdict";

    // trigger the stamp-down animation on the next frame
    requestAnimationFrame(() => {
      stampBadge.classList.remove("verdict__stamp--hidden");
    });
  } catch (err) {
    hint.textContent = "Could not reach the server.";
  } finally {
    analyzeBtn.disabled = false;
  }
}

analyzeBtn.addEventListener("click", analyze);

headlineInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
    analyze();
  }
});
