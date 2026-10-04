/**
 * PRECISION OBSERVABILITY CONSOLE - CLIENT LOGIC
 * High-craft telemetry client for LangGraph state machine tracking & document indexing.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const statTotalDocs = document.getElementById("statTotalDocs");
  const statTotalVectors = document.getElementById("statTotalVectors");
  const statEmbedModel = document.getElementById("statEmbedModel");
  const statRerankerModel = document.getElementById("statRerankerModel");
  const statLlmModel = document.getElementById("statLlmModel");
  const statThreshold = document.getElementById("statThreshold");

  const refreshDocsBtn = document.getElementById("refreshDocsBtn");
  const documentTableBody = document.getElementById("documentTableBody");
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");

  const queryForm = document.getElementById("queryForm");
  const queryInput = document.getElementById("queryInput");
  const submitQueryBtn = document.getElementById("submitQueryBtn");
  const executionTimer = document.getElementById("executionTimer");

  // Circuit Nodes
  const nodeTriage = document.getElementById("nodeTriage");
  const metricTriage = document.getElementById("metricTriage");
  const nodeRetrieve = document.getElementById("nodeRetrieve");
  const metricRetrieve = document.getElementById("metricRetrieve");
  const nodeRerank = document.getElementById("nodeRerank");
  const metricRerank = document.getElementById("metricRerank");
  const nodeGrade = document.getElementById("nodeGrade");
  const metricGrade = document.getElementById("metricGrade");
  const nodeGenerate = document.getElementById("nodeGenerate");
  const metricGenerate = document.getElementById("metricGenerate");
  const nodeRewrite = document.getElementById("nodeRewrite");
  const metricRewrite = document.getElementById("metricRewrite");

  // Gauge & Output
  const gaugeFill = document.getElementById("gaugeFill");
  const confidenceReadout = document.getElementById("confidenceReadout");
  const outputBody = document.getElementById("outputBody");
  const outputLatency = document.getElementById("outputLatency");
  const chunkBadge = document.getElementById("chunkBadge");
  const chunksList = document.getElementById("chunksList");

  let systemThreshold = 0.70;

  // =========================================================================
  // Telemetry & Stats Fetching
  // =========================================================================
  async function fetchStats() {
    try {
      const res = await fetch("/api/stats");
      if (!res.ok) return;
      const data = await res.json();

      statTotalDocs.textContent = data.total_documents || 0;
      statTotalVectors.textContent = (data.total_vectors || 0).toLocaleString();
      const embDevice = (data.embedding_device || "cuda").toUpperCase();
      statEmbedModel.textContent = `${data.embedding_model || "BAAI/bge-small-en-v1.5"} [${embDevice}]`;
      if (statRerankerModel) {
        const isFlashRank = (data.reranker_provider || "").toLowerCase() === "flashrank";
        const rrDevice = isFlashRank ? "ONNX" : (data.reranker_device || "cpu").toUpperCase();
        statRerankerModel.textContent = `${data.reranker_model || "ms-marco-MiniLM-L-12-v2"} [${rrDevice}]`;
      }
      statLlmModel.textContent = `${data.llm_provider.toUpperCase()} (${data.llm_model})`;

      systemThreshold = data.confidence_threshold || 0.70;
      statThreshold.textContent = systemThreshold.toFixed(2);
    } catch (err) {
      console.warn("[Telemetry] Failed to load stats:", err);
    }
  }

  // =========================================================================
  // Document Ledger
  // =========================================================================
  async function fetchDocuments() {
    try {
      documentTableBody.innerHTML = `<tr><td colspan="4" class="table-empty mono">Scanning storage repository...</td></tr>`;
      const res = await fetch("/api/documents");
      if (!res.ok) return;
      const docs = await res.json();

      if (!docs || docs.length === 0) {
        documentTableBody.innerHTML = `<tr><td colspan="4" class="table-empty mono">No documents found in data/. Drop files above to index.</td></tr>`;
        return;
      }

      documentTableBody.innerHTML = "";
      docs.forEach(doc => {
        const tr = document.createElement("tr");

        const statusBadge = doc.is_indexed
          ? `<span class="badge-status badge-indexed">INDEXED</span>`
          : `<span class="badge-status badge-unindexed">UNINDEXED</span>`;

        const actionBtn = doc.is_indexed
          ? `<button class="btn-action" data-file="${doc.filename}" data-reindex="true">RE-INDEX</button>`
          : `<button class="btn-action" data-file="${doc.filename}" data-reindex="false">INGEST</button>`;

        tr.innerHTML = `
          <td>
            <span class="doc-name">${escapeHtml(doc.filename)}</span>
            <span class="doc-hash mono" title="${doc.doc_hash}">SHA: ${doc.doc_hash_short}</span>
          </td>
          <td class="mono">${doc.size_kb} KB</td>
          <td>${statusBadge}</td>
          <td>${actionBtn}</td>
        `;

        documentTableBody.appendChild(tr);
      });

      // Bind action buttons
      documentTableBody.querySelectorAll(".btn-action").forEach(btn => {
        btn.addEventListener("click", () => {
          const filename = btn.getAttribute("data-file");
          const force = btn.getAttribute("data-reindex") === "true";
          ingestDocument(filename, force, btn);
        });
      });

    } catch (err) {
      documentTableBody.innerHTML = `<tr><td colspan="4" class="table-empty mono text-red">Error loading ledger: ${err.message}</td></tr>`;
    }
  }

  async function ingestDocument(filename, forceReindex, buttonElem) {
    const originalText = buttonElem.textContent;
    buttonElem.textContent = "INDEXING...";
    buttonElem.disabled = true;

    try {
      const res = await fetch("/api/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename, force_reindex: forceReindex })
      });
      let data;
      try {
        data = await res.json();
      } catch (parseErr) {
        data = { message: `HTTP ${res.status}: ${res.statusText}` };
      }

      if (res.ok) {
        await fetchStats();
        await fetchDocuments();
      } else {
        alert(`Ingestion failed: ${data.detail || data.message || "Unknown error"}`);
      }
    } catch (err) {
      alert(`Network error: ${err.message}`);
    } finally {
      buttonElem.textContent = originalText;
      buttonElem.disabled = false;
    }
  }

  // =========================================================================
  // Drag and Drop File Upload
  // =========================================================================
  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));

  dropzone.addEventListener("drop", async (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      await handleUploadFiles(e.dataTransfer.files);
    }
  });

  fileInput.addEventListener("change", async () => {
    if (fileInput.files.length > 0) {
      await handleUploadFiles(fileInput.files);
      fileInput.value = "";
    }
  });

  async function handleUploadFiles(files) {
    for (const file of files) {
      const formData = new FormData();
      formData.append("file", file);

      try {
        const res = await fetch("/api/upload?auto_ingest=true", {
          method: "POST",
          body: formData
        });
        if (!res.ok) {
          console.error("Upload error for file:", file.name);
        }
      } catch (err) {
        console.error("Upload network error:", err);
      }
    }
    await fetchStats();
    await fetchDocuments();
  }

  // =========================================================================
  // Query Console & State Machine Execution
  // =========================================================================
  queryForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = queryInput.value.trim();
    if (!question) return;

    submitQueryBtn.disabled = true;
    resetCircuitState();

    // Start stopwatch
    let startTime = Date.now();
    executionTimer.textContent = "IN-FLIGHT (0.0s)";
    const timerInterval = setInterval(() => {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      executionTimer.textContent = `IN-FLIGHT (${elapsed}s)`;
    }, 100);

    // Set initial triage node active
    if (nodeTriage) {
      nodeTriage.className = "circuit-node active";
      metricTriage.textContent = "CLASSIFYING...";
    }
    nodeRetrieve.className = "circuit-node";
    metricRetrieve.textContent = "STANDBY";
    if (nodeRerank) {
      nodeRerank.className = "circuit-node";
      metricRerank.textContent = "STANDBY";
    }

    try {
      const res = await fetch("/api/query/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question })
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Query execution failed.");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let streamedAnswer = "";
      let completed = false;

      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });

        let boundary = buffer.indexOf("\n\n");
        while (boundary !== -1) {
          const block = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          const dataLine = block.split("\n").find(line => line.startsWith("data:"));
          if (dataLine) {
            const event = JSON.parse(dataLine.slice(5).trim());
            if (event.event === "token") {
              streamedAnswer += event.token;
              renderStreamingAnswer(streamedAnswer);
              await new Promise(resolve => requestAnimationFrame(resolve));
            } else if (event.event === "complete") {
              renderExecutionTrace(event.trace);
              completed = true;
            } else if (event.event === "error") {
              throw new Error(event.detail || "Query execution failed.");
            }
          }
          boundary = buffer.indexOf("\n\n");
        }

        if (done) break;
      }

      clearInterval(timerInterval);
      if (!completed) throw new Error("Query stream ended before completion.");

    } catch (err) {
      clearInterval(timerInterval);
      executionTimer.textContent = "EXECUTION ERROR";
      outputBody.innerHTML = `<p class="mono text-red">ERROR: ${escapeHtml(err.message)}</p>`;
    } finally {
      submitQueryBtn.disabled = false;
    }
  });

  function resetCircuitState() {
    [nodeTriage, nodeRetrieve, nodeRerank, nodeGrade, nodeGenerate, nodeRewrite].forEach(n => {
      if (n) n.className = "circuit-node";
    });
    if (metricTriage) metricTriage.textContent = "IDLE";
    metricRetrieve.textContent = "IDLE";
    if (metricRerank) metricRerank.textContent = "IDLE";
    metricGrade.textContent = "SCORE: --";
    metricGenerate.textContent = "IDLE";
    metricRewrite.textContent = "RETRY: 0/2";

    gaugeFill.style.width = "0%";
    confidenceReadout.textContent = `0.00 / ${systemThreshold.toFixed(2)} [IN-FLIGHT]`;
    outputBody.classList.remove("is-streaming");
    outputBody.innerHTML = `<p class="mono text-muted">Executing LangGraph reasoning loops...</p>`;
    outputLatency.textContent = "-- ms";
    chunksList.innerHTML = `<div class="chunk-empty mono">Fetching chunks...</div>`;
    chunkBadge.textContent = "0 CHUNKS";
  }

  function renderExecutionTrace(trace) {
    executionTimer.textContent = `COMPLETED (${(trace.total_latency_ms / 1000).toFixed(2)}s)`;
    outputLatency.textContent = `${trace.total_latency_ms} ms`;

    // 1. Update Circuit Blocks
    const triageStep = trace.steps_trace ? trace.steps_trace.find(s => s.node === "triage") : null;
    if (nodeTriage) {
      nodeTriage.className = "circuit-node pass";
      metricTriage.textContent = triageStep && triageStep.intent ? triageStep.intent.toUpperCase() : "RESOLVED";
    }

    const scoreVal = trace.confidence_score;
    const isPass = scoreVal >= trace.threshold;

    const isDirect = triageStep && triageStep.intent === "direct";
    if (isDirect) {
      nodeRetrieve.className = "circuit-node";
      metricRetrieve.textContent = "BYPASSED";
      if (nodeRerank) {
        nodeRerank.className = "circuit-node";
        metricRerank.textContent = "BYPASSED";
      }
      nodeGrade.className = "circuit-node pass";
      metricGrade.textContent = "BYPASS (1.00)";
      nodeRewrite.className = "circuit-node branch-node";
      metricRewrite.textContent = "BYPASSED";
      nodeGenerate.className = "circuit-node pass";
      metricGenerate.textContent = "DIRECT ANSWER";
    } else {
      const retStep = trace.steps_trace ? trace.steps_trace.find(s => s.node === "retrieve") : null;
      const rerankStep = trace.steps_trace ? trace.steps_trace.find(s => s.node === "rerank") : null;

      nodeRetrieve.className = "circuit-node pass";
      if (retStep && retStep.details) {
        const match = retStep.details.match(/\d+/);
        metricRetrieve.textContent = match ? `FOUND ${match[0]} CANDIDATES` : `FOUND ${(trace.retrieved_chunks || []).length} CHUNKS`;
      } else {
        metricRetrieve.textContent = `FOUND ${(trace.retrieved_chunks || []).length} CHUNKS`;
      }

      if (nodeRerank) {
        nodeRerank.className = "circuit-node pass";
        if (rerankStep) {
          metricRerank.textContent = `TOP ${(trace.retrieved_chunks || []).length} RESCORED`;
        } else {
          metricRerank.textContent = `TOP ${(trace.retrieved_chunks || []).length} PRUNED`;
        }
      }

      nodeGrade.className = `circuit-node ${isPass ? "pass" : "warn"}`;
      metricGrade.textContent = `SCORE: ${scoreVal.toFixed(2)}`;

      if (trace.retries_count > 0) {
        nodeRewrite.className = "circuit-node branch-node warn";
        metricRewrite.textContent = `RETRY: ${trace.retries_count}/2`;
      } else {
        nodeRewrite.className = "circuit-node branch-node";
        metricRewrite.textContent = "RETRY: 0/2 (NONE)";
      }

      nodeGenerate.className = `circuit-node ${isPass ? "pass" : "warn"}`;
      metricGenerate.textContent = isPass ? "GROUNDED ANSWER" : "REFUSAL TRIGGERED";
    }

    // 2. Animate Confidence Gauge

    const percent = Math.min(100, Math.max(0, scoreVal * 100));
    gaugeFill.style.width = `${percent}%`;
    const statusText = isPass ? "[PASS]" : "[BELOW GATE]";
    confidenceReadout.textContent = `${scoreVal.toFixed(2)} / ${trace.threshold.toFixed(2)} ${statusText}`;

    // 3. Render Grounded Answer with Formatted Citations
    renderAnswerText(trace.final_answer);

    // 4. Render Raw Chunk Inspector
    renderChunks(trace.retrieved_chunks);
  }

  function renderStreamingAnswer(answer) {
    outputBody.classList.add("is-streaming");
    outputBody.textContent = answer;
  }

  function renderAnswerText(answer) {
    outputBody.classList.remove("is-streaming");
    if (!answer) {
      outputBody.innerHTML = `<p class="mono text-muted">No response generated.</p>`;
      return;
    }

    // Convert inline citations like [Source: filename, Page X] into styled citation pills
    const citationRegex = /\[Source:\s*([^,\]]+)(?:,\s*Page\s*([^\]]+))?\]/gi;
    let formatted = escapeHtml(answer).replace(citationRegex, (match, src, page) => {
      const pageInfo = page ? `p. ${page.trim()}` : "p. 1";
      return `<span class="citation-pill mono" title="${match}">⚲ ${src.trim()} [${pageInfo}]</span>`;
    });

    // Convert newlines to paragraphs
    const paragraphs = formatted.split("\n\n").map(p => `<p>${p.replace(/\n/g, "<br>")}</p>`).join("");
    outputBody.innerHTML = paragraphs;
  }

  function renderChunks(chunks) {
    const list = chunks || [];
    chunkBadge.textContent = `${list.length} CHUNKS`;

    if (list.length === 0) {
      chunksList.innerHTML = `<div class="chunk-empty mono">No chunks retrieved for this query.</div>`;
      return;
    }


    chunksList.innerHTML = "";
    chunks.forEach((c, idx) => {
      const cosineStr = c.score !== null && c.score !== undefined ? `Cosine: ${c.score.toFixed(4)}` : "Cosine: N/A";
      const rerankStr = c.rerank_score !== null && c.rerank_score !== undefined ? ` | Rerank: ${c.rerank_score.toFixed(4)}` : "";
      const pageStr = (c.page_numbers && c.page_numbers.length) ? `Pages: [${c.page_numbers.join(", ")}]` : "Page: 1";

      const card = document.createElement("div");
      card.className = "chunk-card";
      card.innerHTML = `
        <div class="chunk-meta-bar">
          <span>#${idx + 1} | ${escapeHtml(c.source_file)} (${pageStr})</span>
          <div>
            <span class="text-ice">${cosineStr}</span>
            <span class="text-emerald mono font-bold">${rerankStr}</span>
          </div>
        </div>
        <div class="chunk-text mono">${escapeHtml(c.text)}</div>
      `;
      chunksList.appendChild(card);
    });
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Bind refresh
  refreshDocsBtn.addEventListener("click", () => {
    fetchStats();
    fetchDocuments();
  });

  // Initial Load
  fetchStats();
  fetchDocuments();
});
