const chat = document.getElementById("chat");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const newChatBtn = document.getElementById("new-chat");

let threadId = newThreadId();

function newThreadId() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  return Date.now().toString(36) + Math.random().toString(36).slice(2);
}

// Render text safely: plain text, with ``` fenced blocks shown as code
function renderText(container, text) {
  text.split("```").forEach((part, i) => {
    if (i % 2 === 1) {
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      code.textContent = part.replace(/^[a-zA-Z0-9_-]*\n/, "").trim();
      pre.appendChild(code);
      container.appendChild(pre);
    } else if (part.trim()) {
      const div = document.createElement("div");
      div.className = "text";
      div.textContent = part.trim();
      container.appendChild(div);
    }
  });
}

function addRow(dl, label, value) {
  const dt = document.createElement("dt");
  dt.textContent = label;
  const dd = document.createElement("dd");
  dd.textContent = value;
  dl.append(dt, dd);
}

function scrollDown() {
  chat.scrollTop = chat.scrollHeight;
}

function addMessage(role, text, details) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  renderText(bubble, text);
  wrap.appendChild(bubble);

  if (details) {
    const box = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "How this answer was produced";
    const dl = document.createElement("dl");
    addRow(dl, "Route", details.route);
    addRow(dl, "Search query", details.search_query || "(none)");
    addRow(dl, "Sources", details.sources.length ? details.sources.join(", ") : "(none)");
    box.append(summary, dl);
    wrap.appendChild(box);
  }

  chat.appendChild(wrap);
  scrollDown();
  return wrap;
}

function addTyping() {
  const wrap = document.createElement("div");
  wrap.className = "msg bot typing";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.append(document.createElement("span"), document.createElement("span"), document.createElement("span"));
  wrap.appendChild(bubble);
  chat.appendChild(wrap);
  scrollDown();
  return wrap;
}

function welcome() {
  addMessage("bot", "Hi! Ask me about Kubernetes jobs, cron jobs, autoscaling (HPA/VPA), monitoring or work queues.");
}

async function send(question) {
  addMessage("user", question);
  const typing = addTyping();
  sendBtn.disabled = true;
  input.disabled = true;

  try {
    const resp = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, thread_id: threadId }),
    });
    if (!resp.ok) throw new Error("Server returned " + resp.status);
    const data = await resp.json();
    typing.remove();
    addMessage("bot", data.answer, {
      route: data.route,
      search_query: data.search_query,
      sources: data.sources || [],
    });
  } catch (err) {
    typing.remove();
    addMessage("bot", "Sorry, something went wrong: " + err.message);
  } finally {
    sendBtn.disabled = false;
    input.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  input.style.height = "auto";
  send(question);
});

// Enter sends, Shift+Enter makes a new line
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

// Grow the textarea as you type
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 140) + "px";
});

newChatBtn.addEventListener("click", () => {
  threadId = newThreadId();   // new thread = fresh memory on the backend
  chat.replaceChildren();
  welcome();
  input.focus();
});

welcome();
input.focus();