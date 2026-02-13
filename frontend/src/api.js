const ENV_API_BASE = (import.meta.env.VITE_API_BASE || "").trim();
const STORED_API_BASE =
  typeof window !== "undefined"
    ? String(window.localStorage.getItem("ameb_backend_base_url") || "").trim()
    : "";

export const API_BASE =
  STORED_API_BASE ||
  ENV_API_BASE ||
  (typeof window !== "undefined" && window.location?.origin
    ? String(window.location.origin).trim()
    : "http://127.0.0.1:8008");

export async function fetchProjects(params = {}) {
  const url = new URL(`${API_BASE}/projects`);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      url.searchParams.set(key, value);
    }
  });
  const res = await fetch(url);
  return res.json();
}


export async function generateProjectSetcard(projectId, force = false) {
  const url = new URL(`${API_BASE}/projects/${projectId}/setcard`);
  if (force) url.searchParams.set("force", "1");
  const res = await fetch(url, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function createProject(payload) {
  const res = await fetch(`${API_BASE}/projects`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function updateProject(projectId, payload) {
  const res = await fetch(`${API_BASE}/projects/${projectId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function uploadProjectScreenshot(projectId, file, url) {
  const data = new FormData();
  if (file) data.append("file", file);
  if (url) data.append("url", url);
  const res = await fetch(`${API_BASE}/projects/${projectId}/screenshot`, {
    method: "POST",
    body: data,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function exportProjects() {
  const res = await fetch(`${API_BASE}/projects/export`);
  if (!res.ok) throw new Error(await res.text());
  return res.text();
}

export async function importProjects(file) {
  const data = new FormData();
  data.append("file", file);
  const res = await fetch(`${API_BASE}/projects/import`, {
    method: "POST",
    body: data,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchProjectCopyStatus(projectId) {
  const res = await fetch(`${API_BASE}/projects/${projectId}/copy-status`);
  return res.json();
}

export async function reimportProject(projectId, payload = {}) {
  const res = await fetch(`${API_BASE}/projects/${projectId}/reimport`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function runProjectExportCmd(projectId, payload = {}) {
  const res = await fetch(`${API_BASE}/projects/${projectId}/export-cmd`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchProjectTagStatus(projectId) {
  const res = await fetch(`${API_BASE}/projects/${projectId}/tag-status`);
  return res.json();
}

export async function tagProjectMissing(projectId) {
  const res = await fetch(`${API_BASE}/projects/${projectId}/tag-missing`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function retagProject(projectId) {
  const res = await fetch(`${API_BASE}/projects/${projectId}/retag-all`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function regenerateProjectEmbeddings(projectId) {
  const res = await fetch(`${API_BASE}/projects/${projectId}/embeddings/regenerate`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function regenerateEmbeddingsAll() {
  const res = await fetch(`${API_BASE}/embeddings/regenerate-all`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function restartServer() {
  const res = await fetch(`${API_BASE}/server/restart`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function testLlmTags(file, settings) {
  const data = new FormData();
  if (file) data.append("file", file);
  if (settings) data.append("settings_json", JSON.stringify(settings));
  const res = await fetch(`${API_BASE}/llm/test-tags`, { method: "POST", body: data });
  return handleJson(res);
}

export async function migrateAsset(assetId, payload) {
  const res = await fetch(`${API_BASE}/assets/${assetId}/migrate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchMigrateStatus(assetId) {
  const res = await fetch(`${API_BASE}/assets/${assetId}/migrate-status`);
  return res.json();
}

export async function deleteAsset(assetId) {
  const res = await fetch(`${API_BASE}/assets/${assetId}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteProjectAssets(projectId) {
  const res = await fetch(`${API_BASE}/projects/${projectId}/assets`, { method: "DELETE" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteProject(projectId) {
  const res = await fetch(`${API_BASE}/projects/${projectId}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
export async function openProject(id, target = "auto") {
  const url = new URL(`${API_BASE}/projects/${id}/open`);
  if (target) url.searchParams.set("target", target);
  const res = await fetch(url, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchAssets(params = {}) {
  const url = new URL(`${API_BASE}/assets`);
  if (params.semantic === "0") {
    url.searchParams.set("semantic", "0");
  }
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      url.searchParams.set(key, value);
    }
  });
  const res = await fetch(url);
  return res.json();
}

export async function fetchAssetTypes() {
  const res = await fetch(`${API_BASE}/assets/types`);
  return res.json();
}

export async function fetchProjectStats(params = {}) {
  const url = new URL(`${API_BASE}/projects/stats`);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      url.searchParams.set(key, value);
    }
  });
  const res = await fetch(url);
  return res.json();
}

export async function fetchAsset(assetId) {
  const res = await fetch(`${API_BASE}/assets/${assetId}`, { cache: "no-store" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function uploadAsset(file, projectId) {
  const data = new FormData();
  data.append("file", file);
  if (projectId) data.append("project_id", projectId);
  const res = await fetch(`${API_BASE}/assets/upload`, {
    method: "POST",
    body: data,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function exportTags({ hashType, projectId }) {
  const url = new URL(`${API_BASE}/tags/export`);
  if (hashType) url.searchParams.set("hash_type", hashType);
  if (projectId) url.searchParams.set("project_id", projectId);
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return res.text();
}

export async function importTags({ file, hashType, projectId, mode }) {
  const url = new URL(`${API_BASE}/tags/import`);
  if (hashType) url.searchParams.set("hash_type", hashType);
  if (projectId) url.searchParams.set("project_id", projectId);
  if (mode) url.searchParams.set("mode", mode);
  const data = new FormData();
  data.append("file", file);
  const res = await fetch(url, { method: "POST", body: data });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function clearAllTags() {
  const res = await fetch(`${API_BASE}/tags/clear`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function generateTags(id) {
  const res = await fetch(`${API_BASE}/assets/${id}/generate-tags`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function mergeAssetTags({ assetIds, tags }) {
  const res = await fetch(`${API_BASE}/assets/tags/merge`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ asset_ids: assetIds, tags }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function updateTags(id, tags) {
  const res = await fetch(`${API_BASE}/assets/${id}/tags`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tags }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchSettings() {
  const res = await fetch(`${API_BASE}/settings`);
  return res.json();
}

export async function updateSettings(payload) {
  const res = await fetch(`${API_BASE}/settings`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function resetDatabase() {
  const res = await fetch(`${API_BASE}/admin/reset-db`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchTasks() {
  const res = await fetch(`${API_BASE}/tasks`, { cache: "no-store" });
  return handleJson(res);
}

export async function fetchQueueStatus() {
  const res = await fetch(`${API_BASE}/queue/status`, { cache: "no-store" });
  return handleJson(res);
}

export async function enqueueOpenAiRecovery({ flow, taskId, limit = 300, staleMinutes = 180 } = {}) {
  const url = new URL(`${API_BASE}/openai/recover-enqueue`);
  if (flow) url.searchParams.set("flow", flow);
  if (taskId !== undefined && taskId !== null && String(taskId) !== "") url.searchParams.set("task_id", String(taskId));
  url.searchParams.set("limit", String(limit));
  url.searchParams.set("stale_minutes", String(staleMinutes));
  const res = await fetch(url, { method: "POST" });
  return handleJson(res);
}

export async function cancelTask(taskId) {
  const res = await fetch(`${API_BASE}/tasks/${taskId}/cancel`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function cancelAllTasks() {
  const res = await fetch(`${API_BASE}/tasks/cancel-all`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteTask(taskId) {
  const res = await fetch(`${API_BASE}/tasks/${taskId}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function cleanupTasks() {
  const res = await fetch(`${API_BASE}/tasks/cleanup`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}


export async function translateProjectNameTags(projectId) {
  const res = await fetch(`${API_BASE}/projects/${projectId}/name-tags`, { method: "POST" });
  return handleJson(res);
}

export async function translateProjectNameTagsMissing(projectId) {
  const res = await fetch(`${API_BASE}/projects/${projectId}/name-tags-missing`, { method: "POST" });
  return handleJson(res);
}

export async function nameTagsProjectSimple(projectId) {
  const res = await fetch(`${API_BASE}/projects/${projectId}/name-tags-simple`, { method: "POST" });
  return handleJson(res);
}

export async function nameTagsProjectSimpleMissing(projectId) {
  const res = await fetch(`${API_BASE}/projects/${projectId}/name-tags-simple-missing`, { method: "POST" });
  return handleJson(res);
}

export async function translateProjectTags(projectId) {
  const res = await fetch(`${API_BASE}/projects/${projectId}/translate-tags`, { method: "POST" });
  return handleJson(res);
}

export async function translateProjectTagsMissing(projectId) {
  const res = await fetch(`${API_BASE}/projects/${projectId}/translate-tags-missing`, { method: "POST" });
  return handleJson(res);
}

export async function translateAllNameTags() {
  const res = await fetch(`${API_BASE}/projects/name-tags-all`, { method: "POST" });
  return handleJson(res);
}

export async function translateAllNameTagsMissing() {
  const res = await fetch(`${API_BASE}/projects/name-tags-all-missing`, { method: "POST" });
  return handleJson(res);
}

export async function nameTagsAllSimple() {
  const res = await fetch(`${API_BASE}/projects/name-tags-all-simple`, { method: "POST" });
  return handleJson(res);
}

export async function nameTagsAllSimpleMissing() {
  const res = await fetch(`${API_BASE}/projects/name-tags-all-simple-missing`, { method: "POST" });
  return handleJson(res);
}

export async function translateAllTags() {
  const res = await fetch(`${API_BASE}/projects/translate-tags-all`, { method: "POST" });
  return handleJson(res);
}

export async function translateAllTagsMissing() {
  const res = await fetch(`${API_BASE}/projects/translate-tags-all-missing`, { method: "POST" });
  return handleJson(res);
}
export async function tagMissingAllProjects() {
  const res = await fetch(`${API_BASE}/projects/tag-missing-all`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}


async function handleJson(res) {
  if (!res.ok) {
    let detail = "";
    try {
      const data = await res.json();
      detail = data?.detail ? String(data.detail) : JSON.stringify(data);
    } catch {
      detail = await res.text();
    }
    throw new Error(detail || res.statusText);
  }
  return res.json();
}

