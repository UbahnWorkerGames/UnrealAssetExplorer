import React, { useEffect, useMemo, useRef, useState } from "react";
import logo64 from "./assets/logo64.png";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import {
  API_BASE,
  createProject,
  fetchAsset,
  fetchAssets,
  fetchAssetTypes,
  fetchProjects,
  fetchProjectStats,
  fetchTasks,
  fetchQueueStatus,
  enqueueOpenAiRecovery,
  cancelTask,
  cancelAllTasks,
  deleteTask,
  cleanupTasks,
  updateTags,
  generateProjectSetcard,
  fetchSettings,
  fetchProjectCopyStatus,
  fetchProjectTagStatus,
  fetchMigrateStatus,
  generateTags,
  exportProjects,
  importProjects,
  deleteAsset,
  deleteProject,
  deleteProjectAssets,
  migrateAsset,
  openProject,
  regenerateEmbeddingsAll,
  testLlmTags,
  regenerateProjectEmbeddings,
  retagProject,
  tagProjectMissing,
  tagMissingAllProjects,
  translateProjectNameTags,
  translateProjectNameTagsMissing,
  translateAllNameTags,
  translateAllNameTagsMissing,
  nameTagsProjectSimple,
  nameTagsAllSimple,
  nameTagsProjectSimpleMissing,
  nameTagsAllSimpleMissing,
  translateProjectTags,
  translateAllTags,
  translateProjectTagsMissing,
  translateAllTagsMissing,
  exportTags,
  importTags,
  clearAllTags,
  mergeAssetTags,
  updateProject,
  uploadProjectScreenshot,
  updateSettings,
  uploadAsset,
  reimportProject,
  runProjectExportCmd,
  restartServer,
  resetDatabase,
} from "./api.js";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faCopy,
  faEye,
  faFolderOpen,
  faHardDrive,
  faImages,
  faLightbulb,
} from "@fortawesome/free-regular-svg-icons";
const views = ["assets", "projects", "upload", "tasks", "settings"];
const DEFAULT_TAG_PROMPT = `Generate as many relevant asset tags as make sense (single words, lowercase, no duplicates) in {language}.
Return ONLY a JSON object with a 'tags' array of strings, ordered by relevance (best match first).
Name: {name}
Class: {asset_class}
Description: {description}
Existing tags: {existing_tags}`;
const isTrue = (value) => String(value).toLowerCase() === "true" || value === true;
const taskLabelMap = {
  embeddings_all: "Rebuild semantic (all)",
  embeddings_project: "Rebuild semantic (project)",
  tag_project_missing: "Tag missing",
  tag_project_retag: "Tag all",
  tag_missing_all: "Tag missing (all)",
  name_tags_all: "Asset title to tags (all)",
  name_tags_project: "Asset title to tags (project)",
  name_tags_all_missing: "Asset title to tags missing (all)",
  name_tags_project_missing: "Asset title to tags missing (project)",
  name_tags_all_simple: "Name to tags (all)",
  name_tags_project_simple: "Name to tags (project)",
  tags_translate_all: "Translate tags (all)",
  tags_translate_project: "Translate tags (project)",
  name_tags_all_simple_missing: "Name to tags missing (all)",
  name_tags_project_simple_missing: "Name to tags missing (project)",
  tags_translate_all_missing: "Translate tags missing (all)",
  tags_translate_project_missing: "Translate tags missing (project)",
};
const normalizeProjectArtStyle = (project) => {
  if (!project) return "regular";
  let style = (project.art_style || "regular").toLowerCase();
  if (style === "lowpoly") style = "low poly";
  if (style !== "regular" && style !== "") return style;
  const name = (project.name || "").toLowerCase();
  if (name.includes("low poly") || name.includes("lowpoly")) return "low poly";
  if (name.includes("stylized") || name.includes("stylised")) return "stylized";
  return "regular";
};
const formatTaskLabel = (kind) => taskLabelMap[kind] || kind;

function TagPill({ label, onClick }) {
  return (
    <button className="tag-pill" type="button" onClick={onClick}>
      {label}
    </button>
  );
}
function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return "";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let idx = 0;
  while (value >= 1024 && idx < units.length - 1) {
    value /= 1024;
    idx += 1;
  }
  return `${value.toFixed(1)} ${units[idx]}`;
}
function resolveApiUrl(path) {
  if (!path) return "";
  if (/^https?:\/\//i.test(path)) return path;
  return `${API_BASE}${String(path).startsWith("/") ? path : `/${path}`}`;
}

function getProjectCoverUrl(project) {
  const explicit = resolveApiUrl(project?.screenshot_url);
  if (explicit) return explicit;
  const folderPath = String(project?.folder_path || "").trim();
  if (!folderPath) return "";
  const normalized = folderPath.replace(/\\/g, "/").replace(/\/+$/, "");
  const slug = normalized.split("/").pop();
  if (!slug) return "";
  return resolveApiUrl(`/media/projects/${slug}/screenshot.jpg`);
}
function truncateText(value, maxLength) {
  if (!value) return "";
  if (value.length <= maxLength) return value;
  return `${value.slice(0, Math.max(0, maxLength - 3)).trimEnd()}...`;
}
function formatDuration(ms) {
  if (!Number.isFinite(ms) || ms <= 0) return "";
  const total = Math.round(ms / 1000);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}
function AssetCard({ asset, project, onSelect, onContextMenu, tileSize, isSelected }) {
  const nanite = asset?.meta?.mesh?.nanite_enabled;
  const displayName = asset?.meta?.package ? asset.meta.package.split("/").pop() : asset.name;
  const listImage = asset.anim_thumb || asset.thumb_image;
  const imageSize = tileSize ? Math.max(140, tileSize) : 200;
  const displayTags = (asset?.display_tags && asset.display_tags.length ? asset.display_tags : asset?.tags || []).slice(0, 4);
  return (
    <button
      className={`asset-card${isSelected ? " selected" : ""}`}
      type="button"
      onClick={onSelect}
      onContextMenu={(e) => {
        if (onContextMenu) {
          e.preventDefault();
          onContextMenu(e);
        }
      }}
      style={{ minHeight: imageSize + 90 }}
    >
      <div className="asset-image" style={{ height: imageSize }}>
        {nanite && <span className="asset-pill">Nanite</span>}
        {listImage ? (
          <img src={resolveApiUrl(listImage)} alt={asset.name} />
        ) : (
          <div className="asset-placeholder">No preview</div>
        )}
      </div>
      <div className="asset-body">
        <div className="asset-title">{displayName}</div>
        <div className="asset-meta">
          {project?.name || "Unassigned"}
        </div>
        {project?.art_style && <div className="asset-style">{project.art_style}</div>}
        {project?.project_era && <div className="asset-era">{project.project_era}</div>}
        {displayTags.length > 0 && (
          <div className="asset-tags">
            {displayTags.map((tag) => (
              <span key={tag} className="asset-tag" title={tag}>
                {tag}
              </span>
            ))}
          </div>
        )}
      </div>
    </button>
  );
}
export default function App() {
  const [view, setView] = useState("assets");
  const viewRef = useRef("assets");
  const [aboutOpen, setAboutOpen] = useState(false);
  const aboutRef = useRef(null);
  const [showImportHelper, setShowImportHelper] = useState(() => {
    if (typeof localStorage === "undefined") return true;
    return localStorage.getItem("ameb_hide_importhelper") !== "1";
  });
  const uploadToastIdsRef = useRef(new Map());
  const importToastIdsRef = useRef(new Map());
  const copyToastIdsRef = useRef(new Map());
  const tagToastIdsRef = useRef(new Map());
  const [projects, setProjects] = useState([]);
  const [assets, setAssets] = useState([]);
  const [queryInput, setQueryInput] = useState("");
  const [selectedProjects, setSelectedProjects] = useState([]);
  const [assetTypes, setAssetTypes] = useState([]);
  const [selectedTypes, setSelectedTypes] = useState([]);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(54);
  const loadMoreRef = useRef(null);
  const lastAssetScrollYRef = useRef(null);
  const [hasMore, setHasMore] = useState(true);
  const [totalCount, setTotalCount] = useState(0);
  const [totalAll, setTotalAll] = useState(0);
  const [tileSize, setTileSize] = useState(200);
  const [refreshKey, setRefreshKey] = useState(0);
  const [projectName, setProjectName] = useState("");
  const [projectLink, setProjectLink] = useState("");
  const [projectTags, setProjectTags] = useState("");
  const [projectArtStyle, setProjectArtStyle] = useState("");
  const [projectScreenshotUrl, setProjectScreenshotUrl] = useState("");
  const [projectScreenshotFile, setProjectScreenshotFile] = useState(null);
  const [projectSourcePath, setProjectSourcePath] = useState("");
  const [projectSourceFolder, setProjectSourceFolder] = useState("");
  const [projectFullCopy, setProjectFullCopy] = useState(false);
  const [createIsAi, setCreateIsAi] = useState(false);
  const [projectImportFile, setProjectImportFile] = useState(null);
  const [showCreateProject, setShowCreateProject] = useState(false);
  const [copyStatus, setCopyStatus] = useState(null);
  const [projectSearch, setProjectSearch] = useState("");
  const [showEmptyProjects, setShowEmptyProjects] = useState(false);
  const [projectSortKey, setProjectSortKey] = useState(() => {
    if (typeof localStorage === "undefined") return "name";
    return localStorage.getItem("ameb_project_sort_key") || "name";
  });
  const [projectSortDir, setProjectSortDir] = useState(() => {
    if (typeof localStorage === "undefined") return "asc";
    return localStorage.getItem("ameb_project_sort_dir") || "asc";
  });
  const [editingProjectId, setEditingProjectId] = useState(null);
  const [editName, setEditName] = useState("");
  const [editLink, setEditLink] = useState("");
  const [editTags, setEditTags] = useState("");
  const [editArtStyle, setEditArtStyle] = useState("");
  const [editSourcePath, setEditSourcePath] = useState("");
  const [editSourceFolder, setEditSourceFolder] = useState("");
  const [editIsAi, setEditIsAi] = useState(false);
  const [editFullCopy, setEditFullCopy] = useState(false);
  const [editScreenshotUrl, setEditScreenshotUrl] = useState("");
  const [editScreenshotFile, setEditScreenshotFile] = useState(null);
  const [settingsTagIncludeTypes, setSettingsTagIncludeTypes] = useState([]);
  const [settingsTagExcludeTypes, setSettingsTagExcludeTypes] = useState([]);
  const [settingsExportIncludeTypes, setSettingsExportIncludeTypes] = useState([]);
  const [settingsExportExcludeTypes, setSettingsExportExcludeTypes] = useState([]);
  const [assetTypeCatalog, setAssetTypeCatalog] = useState([]);
  const [assetTypeCatalogInput, setAssetTypeCatalogInput] = useState("");
  const [projectPreview, setProjectPreview] = useState(null);
  const [projectStats, setProjectStats] = useState({});
  const [projectStatsSummary, setProjectStatsSummary] = useState(null);
  const [llmTestImage, setLlmTestImage] = useState(null);
  const [llmTestResult, setLlmTestResult] = useState(null);
  const [projectImported, setProjectImported] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem("ameb_project_imported") || "{}");
    } catch {
      return {};
    }
  });
  const [projectTagStatus, setProjectTagStatus] = useState({});
  const [tagExportHash, setTagExportHash] = useState("blake3");
  const [tagExportProjectId, setTagExportProjectId] = useState("");
  const [tagImportHash, setTagImportHash] = useState("blake3");
  const [tagImportProjectId, setTagImportProjectId] = useState("");
  const [tagImportFile, setTagImportFile] = useState(null);
  const [tagImportMode, setTagImportMode] = useState("replace");
  const [uploadFiles, setUploadFiles] = useState([]);
  const [uploadProject, setUploadProject] = useState("");
  const [uploadProgress, setUploadProgress] = useState(null);
  const [showResetModal, setShowResetModal] = useState(false);
  const [resetConfirmInput, setResetConfirmInput] = useState("");
  const [projectFilterSearch, setProjectFilterSearch] = useState("");
  const [tasks, setTasks] = useState([]);
  const [tasksLoading, setTasksLoading] = useState(false);
  const [queueStatus, setQueueStatus] = useState(null);
  const [queueStatusError, setQueueStatusError] = useState("");
  const [queueStatusUpdatedAt, setQueueStatusUpdatedAt] = useState(0);
  const tasksLoadedRef = useRef(false);
  const [selectedAssetId, setSelectedAssetId] = useState(null);
  const [selectedAssetIds, setSelectedAssetIds] = useState([]);
  const [editableTags, setEditableTags] = useState([]);
  const [tagInput, setTagInput] = useState("");
  const [selectedAsset, setSelectedAsset] = useState(null);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [isFetchingAssets, setIsFetchingAssets] = useState(false);
  const isFetchingAssetsRef = useRef(false);
  const pageRequestRef = useRef(false);
  const lastLoadedPageRef = useRef(0);
  const [detailImagePreview, setDetailImagePreview] = useState(null);
  const [pendingNext, setPendingNext] = useState(false);
  const prevTagStatusRef = useRef({});
  const assetsRequestRef = useRef(0);
  const [apiKeyInput, setApiKeyInput] = useState("");
  const [openaiKeyInput, setOpenaiKeyInput] = useState("");
  const [openrouterKeyInput, setOpenrouterKeyInput] = useState("");
  const [groqKeyInput, setGroqKeyInput] = useState("");
  const [hasApiKey, setHasApiKey] = useState(false);
  const [hasOpenAiKey, setHasOpenAiKey] = useState(false);
  const [hasOpenRouterKey, setHasOpenRouterKey] = useState(false);
  const [hasGroqKey, setHasGroqKey] = useState(false);
  const [selectedArtStyles, setSelectedArtStyles] = useState(["regular", "stylized", "low poly"]);
  const [selectedEras, setSelectedEras] = useState([]);
  const [projectAiFilter, setProjectAiFilter] = useState("all");
  const [projectArtStyleFilter, setProjectArtStyleFilter] = useState("__all__");
  const [projectEraFilter, setProjectEraFilter] = useState("__all__");
  const [naniteFilter, setNaniteFilter] = useState("all");
  const [collisionFilter, setCollisionFilter] = useState("all");
  const [useSemanticSearch, setUseSemanticSearch] = useState(() => {
    if (typeof window === "undefined") return true;
    const stored = window.localStorage.getItem("useSemanticSearch");
    if (stored === null) return true;
    return stored === "true";
  });
  const [contentDir, setContentDir] = useState("");
  const [overwriteExisting, setOverwriteExisting] = useState(false);
  const [migrateStatus, setMigrateStatus] = useState(null);
  const [savedViews, setSavedViews] = useState([]);
  const [selectedViewId, setSelectedViewId] = useState("");
  const projectArtStyles = useMemo(() => {
    const set = new Set();
    projects.forEach((project) => {
      const value = String(project.art_style || "").trim();
      if (value) set.add(value);
    });
    return Array.from(set).sort((a, b) =>
      a.localeCompare(b, undefined, { sensitivity: "base" })
    );
  }, [projects]);
  const hasEmptyProjectArtStyle = useMemo(
    () => projects.some((project) => !String(project.art_style || "").trim()),
    [projects]
  );
  const projectEraOptions = useMemo(() => {
    const set = new Set();
    projects.forEach((project) => {
      const value = String(project.project_era || "").trim();
      if (value) set.add(value);
    });
    return Array.from(set).sort((a, b) =>
      a.localeCompare(b, undefined, { sensitivity: "base" })
    );
  }, [projects]);
  const hasEmptyProjectEra = useMemo(
    () => projects.some((project) => !String(project.project_era || "").trim()),
    [projects]
  );
  const eraOptions = useMemo(() => {
    const set = new Set();
    projects.forEach((project) => {
      const value = String(project.project_era || "").trim().toLowerCase();
      if (value) set.add(value);
    });
    return Array.from(set).sort();
  }, [projects]);
  useEffect(() => {
    if (projectArtStyleFilter === "__all__") return;
    if (
      projectArtStyleFilter !== "__none__" &&
      !projectArtStyles.includes(projectArtStyleFilter)
    ) {
      setProjectArtStyleFilter("__all__");
    }
  }, [projectArtStyleFilter, projectArtStyles]);
  useEffect(() => {
    if (projectEraFilter === "__all__") return;
    if (projectEraFilter !== "__none__" && !projectEraOptions.includes(projectEraFilter)) {
      setProjectEraFilter("__all__");
    }
  }, [projectEraFilter, projectEraOptions]);
  const filteredProjects = useMemo(() => {
    const query = projectSearch.trim().toLowerCase();
    return projects.filter((project) => {
      const isAi = Boolean(project.is_ai_generated);
      if (projectAiFilter === "only" && !isAi) return false;
      if (projectAiFilter === "exclude" && isAi) return false;
      const artStyle = String(project.art_style || "").trim();
      if (projectArtStyleFilter === "__none__" && artStyle) {
        return false;
      }
      if (
        projectArtStyleFilter !== "__all__" &&
        projectArtStyleFilter !== "__none__" &&
        artStyle.toLowerCase() !== String(projectArtStyleFilter || "").trim().toLowerCase()
      ) {
        return false;
      }
      const eraValue = String(project.project_era || "").trim();
      if (projectEraFilter === "__none__" && eraValue) {
        return false;
      }
      if (
        projectEraFilter !== "__all__" &&
        projectEraFilter !== "__none__" &&
        eraValue.toLowerCase() !== String(projectEraFilter || "").trim().toLowerCase()
      ) {
        return false;
      }
      if (showEmptyProjects) {
        const stats = projectStats[project.id] || {};
        const total = Number(stats.total_all ?? stats.total ?? 0);
        if (total > 0) return false;
      }
      if (!query) return true;
      const nameMatch = (project.name || "").toLowerCase().includes(query);
      const tagsMatch = (project.tags || []).some((tag) =>
        String(tag || "").toLowerCase().includes(query)
      );
      const catMatch = String(project.category_name ?? project.category ?? "").toLowerCase().includes(query);
      const descMatch = String(project.description || "").toLowerCase().includes(query);
      return nameMatch || tagsMatch || catMatch || descMatch;
    });
  }, [
    projects,
    projectSearch,
    projectAiFilter,
    projectArtStyleFilter,
    projectEraFilter,
    projectStats,
    showEmptyProjects,
  ]);
  const sortedProjects = useMemo(() => {
    const dir = projectSortDir === "desc" ? -1 : 1;
    const list = [...filteredProjects];
    list.sort((a, b) => {
      if (projectSortKey === "id") {
        const ai = Number(a.id || 0);
        const bi = Number(b.id || 0);
        return (ai - bi) * dir;
      }
      if (projectSortKey === "source_folder") {
        const af = String(a.source_folder || a.folder_path || "");
        const bf = String(b.source_folder || b.folder_path || "");
        return af.localeCompare(bf, undefined, { numeric: true, sensitivity: "base" }) * dir;
      }
      const an = String(a.name || "");
      const bn = String(b.name || "");
      return an.localeCompare(bn, undefined, { numeric: true, sensitivity: "base" }) * dir;
    });
    return list;
  }, [filteredProjects, projectSortKey, projectSortDir]);
  const projectStatsAggregate = useMemo(() => {
    let totalItems = 0;
    let totalSourceBytes = 0;
    const typeTotals = {};
    projects.forEach((project) => {
      totalSourceBytes += Number(project.source_size_bytes || 0);
    });
    Object.values(projectStats || {}).forEach((stats) => {
      totalItems += Number(stats?.total || 0);
      const types = stats?.types || {};
      Object.entries(types).forEach(([type, count]) => {
        typeTotals[type] = (typeTotals[type] || 0) + Number(count || 0);
      });
    });
    const typeList = Object.entries(typeTotals).sort((a, b) => b[1] - a[1]);
    return { totalItems, totalSourceBytes, typeList };
  }, [projects, projectStats]);
  const tagStatsAggregate = useMemo(() => {
    let totalAssets = 0;
    let taggedAssets = 0;
    let tagAssignmentsTotal = 0;
    let uniqueTagsTotal = 0;

    Object.values(projectStats || {}).forEach((stats) => {
      totalAssets += Number(stats?.total || 0);
      taggedAssets += Number(stats?.tagged || 0);
      tagAssignmentsTotal += Number(stats?.tag_assignments_total || 0);
    });

    if (projectStatsSummary && Number.isFinite(Number(projectStatsSummary.unique_tags_total))) {
      uniqueTagsTotal = Number(projectStatsSummary.unique_tags_total || 0);
    } else {
      uniqueTagsTotal = Object.values(projectStats || {}).reduce(
        (acc, stats) => acc + Number(stats?.unique_tags_count || 0),
        0
      );
    }

    const assetsWithoutTags =
      projectStatsSummary && Number.isFinite(Number(projectStatsSummary.assets_without_tags))
        ? Number(projectStatsSummary.assets_without_tags || 0)
        : Math.max(0, totalAssets - taggedAssets);
    const avgTagsPerTaggedAsset = taggedAssets ? tagAssignmentsTotal / taggedAssets : 0;

    return {
      totalAssets,
      taggedAssets,
      assetsWithoutTags,
      tagAssignmentsTotal,
      uniqueTagsTotal,
      avgTagsPerTaggedAsset,
    };
  }, [projectStats, projectStatsSummary]);
  const [refreshingProjectSizes, setRefreshingProjectSizes] = useState(false);
  const [restartingServer, setRestartingServer] = useState(false);
  const [settings, setSettings] = useState({
    provider: "openai",
    base_url: "http://127.0.0.1:11434",
    api_key: "",
    openai_base_url: "https://api.openai.com",
    openrouter_base_url: "https://openrouter.ai/api",
    groq_base_url: "https://api.groq.com/openai",
    ollama_base_url: "http://127.0.0.1:11434",
    model: "",
    openai_model: "",
    openrouter_model: "",
    groq_model: "",
    ollama_model: "",
    import_base_url: "http://127.0.0.1:9090",
    skip_export_if_on_server: true,
    export_overwrite_zips: false,
    export_default_image_count: 1,
    export_static_mesh_image_count: 1,
    export_skeletal_mesh_image_count: 1,
    export_material_image_count: 1,
    export_blueprint_image_count: 1,
    export_niagara_image_count: 1,
    export_anim_sequence_image_count: 4,
    export_capture360_discard_frames: 2,
    export_upload_after_export: true,
    export_upload_path_template: "/assets/upload",
    export_check_path_template: "/assets/exists?hash={hash}&hash_type=blake3",
    ue_cmd_path: "",
    ue_cmd_extra_args: "",
    tag_language: "english",
    tag_image_size: 512,
    tag_image_quality: 80,
    default_full_project_copy: false,
    tag_include_types: "",
    tag_exclude_types: "",
    tag_missing_min_tags: 1,
    tag_use_batch_mode: false,
    tag_batch_max_assets: 500,
    tag_batch_project_concurrency: 3,
    tag_translate_enabled: false,
    tag_display_limit: 0,
    generate_embeddings_on_import: true,
    sidebar_width: 280,
    purge_assets_on_startup: false,
    use_temperature: false,
    temperature: 1,
    llm_min_interval_seconds: 0,
  });
  const llmReady = useMemo(() => {
    const provider = (settings.provider || "").toLowerCase();
    if (!provider || provider === "ollama") return true;
    if (provider === "openai") return hasOpenAiKey;
    if (provider === "openrouter") return hasOpenRouterKey;
    if (provider === "groq") return hasGroqKey;
    return hasApiKey;
  }, [settings.provider, hasApiKey, hasOpenAiKey, hasOpenRouterKey, hasGroqKey]);
  const llmReadyTitle = llmReady ? "" : "LLM key required for active provider";
  const activeProviderLabel = useMemo(() => {
    const provider = (settings.provider || "").toLowerCase();
    if (provider === "openai") return "OpenAI";
    if (provider === "openrouter") return "OpenRouter";
    if (provider === "groq") return "Groq";
    if (provider === "ollama") return "Ollama";
    if (provider) return provider.toUpperCase();
    return "LLM";
  }, [settings.provider]);
  const llmActionGroupLabel = useMemo(() => {
    if (activeProviderLabel === "OpenAI") return "OpenAI";
    return `AI (${activeProviderLabel})`;
  }, [activeProviderLabel]);
  function resolveViewFromHash() {
    if (typeof window === "undefined") return null;
    const hash = window.location.hash.replace("#", "").trim();
    return views.includes(hash) ? hash : null;
  }
  function handleViewClick(nextView, preserveProjectSelection = false) {
    if (typeof window !== "undefined") {
      window.location.hash = nextView;
    }
    setAboutOpen(false);
    setView(nextView);
    if (nextView === "assets") {
      if (selectedAssetId || selectedAsset) {
        closeDetail();
      }
      if (!preserveProjectSelection) {
        setSelectedProjects(projects.map((project) => String(project.id)));
      }
      setSelectedAssetId(null);
      setSelectedAsset(null);
    }
  }
  useEffect(() => {
    if (typeof window === "undefined") return undefined;
    const applyHash = () => {
      const next = resolveViewFromHash();
      if (next && next !== viewRef.current) {
        setView(next);
      }
    };
    applyHash();
    const onHashChange = () => applyHash();
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);
  useEffect(() => {
    viewRef.current = view;
    if (typeof window === "undefined") return;
    const hash = `#${view}`;
    if (window.location.hash !== hash) {
      window.history.replaceState(null, "", hash);
    }
  }, [view]);
  useEffect(() => {
    if (typeof localStorage === "undefined") return;
    localStorage.setItem("ameb_hide_importhelper", showImportHelper ? "0" : "1");
  }, [showImportHelper]);
  useEffect(() => {
    if (typeof localStorage === "undefined") return;
    localStorage.setItem("ameb_project_sort_key", projectSortKey);
    localStorage.setItem("ameb_project_sort_dir", projectSortDir);
  }, [projectSortKey, projectSortDir]);
  useEffect(() => {
    if (typeof document === "undefined") return undefined;
    if (!aboutOpen) return undefined;
    const onDocClick = (event) => {
      if (!aboutRef.current) return;
      if (!aboutRef.current.contains(event.target)) {
        setAboutOpen(false);
      }
    };
    const onKeyDown = (event) => {
      if (event.key === "Escape") {
        setAboutOpen(false);
      }
    };
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [aboutOpen]);
  useEffect(() => {
    if (typeof window === "undefined") return undefined;
    if (!API_BASE) return undefined;
    const url = `${API_BASE.replace(/\/$/, "")}/events`;
    const evt = new EventSource(url);
    const onUpload = (event) => {
      try {
        const payload = JSON.parse(event.data || "{}");
        if (payload.type && payload.type !== "upload") return;
        const batchId = payload.batch_id ?? "default";
        const name = payload.name || "asset";
        const current = payload.current ?? 0;
        const total = payload.total ?? 0;
        const percent = payload.percent ?? 0;
        const label =
          total > 0
            ? `Imported ${current}/${total} (${percent}%) — ${name}`
            : `Imported ${name}`;
        const map = uploadToastIdsRef.current;
        const existing = map.get(batchId);
        if (existing) {
          toast.update(existing, { render: label });
          if (total > 0 && current >= total) {
            toast.update(existing, { autoClose: 2000 });
            map.delete(batchId);
          }
        } else {
          const id = toast.info(label, { autoClose: false });
          map.set(batchId, id);
        }
      } catch (err) {
        console.error(err);
      }
    };
    const onProjectsImport = (event) => {
      try {
        const payload = JSON.parse(event.data || "{}");
        const importId = payload.import_id || "projects_import";
        const current = payload.current ?? 0;
        const total = payload.total ?? 0;
        const created = payload.created ?? 0;
        const skipped = payload.skipped ?? 0;
        const errors = payload.errors ?? 0;
        const status = payload.status || "running";
        const label =
          total > 0
            ? `Import ${current}/${total} (created ${created}, skipped ${skipped}, errors ${errors})`
            : `Import running (created ${created}, skipped ${skipped}, errors ${errors})`;
        const map = importToastIdsRef.current;
        const existing = map.get(importId);
        if (existing) {
          toast.update(existing, { render: label });
          if (status === "done") {
            toast.update(existing, {
              render: `Import done: ${created} created, ${skipped} skipped, ${errors} errors`,
              autoClose: 4000,
              type: errors > 0 ? "warning" : "success",
            });
            map.delete(importId);
          }
        } else {
          const id = toast.info(label, { autoClose: false });
          map.set(importId, id);
        }
      } catch (err) {
        console.error(err);
      }
    };
    const onTagsImport = (event) => {
      try {
        const payload = JSON.parse(event.data || "{}");
        const importId = payload.import_id || "tags_import";
        const current = payload.current ?? 0;
        const total = payload.total ?? 0;
        const updated = payload.updated ?? 0;
        const missing = payload.missing ?? 0;
        const errors = payload.errors ?? 0;
        const status = payload.status || "running";
        const label =
          total > 0
            ? `Tags ${current}/${total} (updated ${updated}, missing ${missing}, errors ${errors})`
            : `Tags running (updated ${updated}, missing ${missing}, errors ${errors})`;
        const map = importToastIdsRef.current;
        const existing = map.get(importId);
        if (existing) {
          toast.update(existing, { render: label });
          if (status === "done") {
            toast.update(existing, {
              render: `Tags done: ${updated} updated, ${missing} missing, ${errors} errors`,
              autoClose: 4000,
              type: errors > 0 ? "warning" : "success",
            });
            map.delete(importId);
          }
        } else {
          const id = toast.info(label, { autoClose: false });
          map.set(importId, id);
        }
      } catch (err) {
        console.error(err);
      }
    };
    evt.addEventListener("upload", onUpload);
    evt.addEventListener("projects_import", onProjectsImport);
    evt.addEventListener("tags_import", onTagsImport);
    evt.onerror = () => {
      evt.close();
    };
    return () => {
      evt.removeEventListener("upload", onUpload);
      evt.removeEventListener("projects_import", onProjectsImport);
      evt.removeEventListener("tags_import", onTagsImport);
      evt.close();
    };
  }, []);
  const projectMap = useMemo(() => {
    const map = new Map();
    projects.forEach((project) => map.set(project.id, project));
    return map;
  }, [projects]);
  const uniqueSelectedProjects = useMemo(
    () => Array.from(new Set(selectedProjects)),
    [selectedProjects]
  );
  const projectFiltersActive =
    uniqueSelectedProjects.length > 0 && uniqueSelectedProjects.length < projects.length;
  const visibleAssets = useMemo(() => {
    if (!projectFiltersActive) return assets;
    const allowed = new Set(uniqueSelectedProjects.map((id) => String(id)));
    return assets.filter((asset) => allowed.has(String(asset.project_id)));
  }, [assets, projectFiltersActive, uniqueSelectedProjects]);
  function useDebounce(value, delayMs) {
    const [debounced, setDebounced] = useState(value);
    useEffect(() => {
      const handle = setTimeout(() => setDebounced(value), delayMs);
      return () => clearTimeout(handle);
    }, [value, delayMs]);
    return debounced;
  }
  const debouncedQuery = useDebounce(queryInput, 400);
  const effectiveQuery = debouncedQuery;
  const trimmedQuery = effectiveQuery.trim();
  const filterParams = useMemo(() => {
    const typeFiltersActive = selectedTypes.length && selectedTypes.length !== assetTypes.length;
    const styleFiltersActive = selectedArtStyles.length && selectedArtStyles.length < 3;
    const eraFiltersActive = selectedEras.length && selectedEras.length < eraOptions.length;
    const filteredProjectIds = uniqueSelectedProjects.filter((projectId) => {
      const project = projects.find((item) => String(item.id) === projectId);
      if (styleFiltersActive) {
        const style = normalizeProjectArtStyle(project);
        if (!selectedArtStyles.includes(style)) return false;
      }
      if (eraFiltersActive) {
        const era = String(project?.project_era || "").trim().toLowerCase();
        if (era && !selectedEras.includes(era)) return false;
      }
      return true;
    });
    const allProjectsSelected =
      projects.length > 0 && uniqueSelectedProjects.length >= projects.length;
    const useProjectFilter = projectFiltersActive || styleFiltersActive || eraFiltersActive;
    return {
      query: trimmedQuery && (!useSemanticSearch || trimmedQuery.length >= 3) ? trimmedQuery : undefined,
      project_ids:
        useProjectFilter && filteredProjectIds.length
          ? filteredProjectIds.join(",")
          : undefined,
      types: typeFiltersActive ? selectedTypes.join(",") : undefined,
      tag: undefined,
      semantic: useSemanticSearch ? "1" : "0",
      nanite: naniteFilter === "with" ? "1" : naniteFilter === "without" ? "0" : undefined,
      collision: collisionFilter === "with" ? "1" : collisionFilter === "without" ? "0" : undefined,
      page,
      page_size: pageSize,
    };
  }, [
    trimmedQuery,
    uniqueSelectedProjects,
    selectedTypes,
    selectedArtStyles,
    selectedEras,
    eraOptions.length,
    assetTypes.length,
    projects,
    page,
    pageSize,
    useSemanticSearch,
    naniteFilter,
    collisionFilter,
  ]);
  const statsParams = useMemo(() => {
    const typeFiltersActive = selectedTypes.length && selectedTypes.length !== assetTypes.length;
    const styleFiltersActive = selectedArtStyles.length && selectedArtStyles.length < 3;
    const eraFiltersActive = selectedEras.length && selectedEras.length < eraOptions.length;
    const filteredProjectIds = uniqueSelectedProjects.filter((projectId) => {
      const project = projects.find((item) => String(item.id) === projectId);
      if (styleFiltersActive) {
        const style = normalizeProjectArtStyle(project);
        if (!selectedArtStyles.includes(style)) return false;
      }
      if (eraFiltersActive) {
        const era = String(project?.project_era || "").trim().toLowerCase();
        if (era && !selectedEras.includes(era)) return false;
      }
      return true;
    });
    const projectFiltersActive =
      uniqueSelectedProjects.length > 0 && uniqueSelectedProjects.length < projects.length;
    const useProjectFilter = projectFiltersActive || styleFiltersActive || eraFiltersActive;
    return {
      query: trimmedQuery || undefined,
      tag: undefined,
      project_ids:
        useProjectFilter && filteredProjectIds.length
          ? filteredProjectIds.join(",")
          : undefined,
      types: typeFiltersActive ? selectedTypes.join(",") : undefined,
      nanite: naniteFilter === "with" ? "1" : naniteFilter === "without" ? "0" : undefined,
      collision: collisionFilter === "with" ? "1" : collisionFilter === "without" ? "0" : undefined,
    };
  }, [
    trimmedQuery,
    selectedTypes,
    assetTypes.length,
    naniteFilter,
    collisionFilter,
    selectedArtStyles,
    selectedEras,
    eraOptions.length,
    uniqueSelectedProjects,
    projects,
    refreshKey,
  ]);
  useEffect(() => {
    fetchProjects()
      .then((data) => {
        setProjects(data);
        setSelectedProjects(data.map((project) => String(project.id)));
      })
      .catch(console.error);
    fetchAssetTypes()
      .then((data) => {
        const types = (data.items || []).filter(Boolean);
        setAssetTypes(types);
        setSelectedTypes(types);
      })
      .catch(console.error);
    fetchSettings().then((data) => {
      const clean = { ...data };
      if (clean.api_key) {
        clean.api_key = "";
      }
      if (clean.use_temperature !== undefined) {
        clean.use_temperature = String(clean.use_temperature).toLowerCase() === "true";
      }
      if (!clean.base_url) {
        clean.base_url = "http://127.0.0.1:11434";
      }
      if (!clean.import_base_url) {
        clean.import_base_url = "http://127.0.0.1:9090";
      }
      if (!clean.openai_base_url) {
        clean.openai_base_url = "https://api.openai.com";
      }
      if (!clean.openrouter_base_url) {
        clean.openrouter_base_url = "https://openrouter.ai/api";
      }
      if (!clean.groq_base_url) {
        clean.groq_base_url = "https://api.groq.com/openai";
      }
      if (!clean.ollama_base_url) {
        clean.ollama_base_url = "http://127.0.0.1:11434";
      }
      if (clean.skip_export_if_on_server !== undefined) {
        const raw = String(clean.skip_export_if_on_server).toLowerCase();
        clean.skip_export_if_on_server = raw === "true" || raw === "1" || raw === "yes" || raw === "on";
      }
      if (clean.tag_translate_enabled !== undefined) {
        const raw = String(clean.tag_translate_enabled).toLowerCase();
        clean.tag_translate_enabled = raw === "true" || raw === "1" || raw === "yes" || raw === "on";
      }
      if (clean.default_full_project_copy !== undefined) {
        const raw = String(clean.default_full_project_copy).toLowerCase();
        clean.default_full_project_copy = raw === "true" || raw === "1" || raw === "yes" || raw === "on";
      }
      if (clean.tag_use_batch_mode !== undefined) {
        const raw = String(clean.tag_use_batch_mode).toLowerCase();
        clean.tag_use_batch_mode = raw === "true" || raw === "1" || raw === "yes" || raw === "on";
      }
      if (clean.generate_embeddings_on_import !== undefined) {
        const raw = String(clean.generate_embeddings_on_import).toLowerCase();
        clean.generate_embeddings_on_import = raw === "true" || raw === "1" || raw === "yes" || raw === "on";
      }
      if (clean.sidebar_width !== undefined && clean.sidebar_width !== "") {
        const parsedWidth = Number(clean.sidebar_width);
        if (!Number.isNaN(parsedWidth)) {
          clean.sidebar_width = parsedWidth;
        }
      }
      if (clean.temperature !== undefined && clean.temperature !== "") {
        const parsedTemp = Number(clean.temperature);
        if (!Number.isNaN(parsedTemp)) {
          clean.temperature = parsedTemp;
        }
      }
      if (clean.tag_display_limit !== undefined && clean.tag_display_limit !== "") {
        const parsedLimit = Number(clean.tag_display_limit);
        if (!Number.isNaN(parsedLimit)) {
          clean.tag_display_limit = parsedLimit;
        }
      }
      if (clean.tag_missing_min_tags !== undefined && clean.tag_missing_min_tags !== "") {
        const parsedMin = Number(clean.tag_missing_min_tags);
        if (!Number.isNaN(parsedMin)) {
          clean.tag_missing_min_tags = parsedMin;
        }
      }
      if (clean.tag_image_size !== undefined && clean.tag_image_size !== "") {
        const parsedSize = Number(clean.tag_image_size);
        if (!Number.isNaN(parsedSize)) {
          clean.tag_image_size = parsedSize;
        }
      }
      if (clean.tag_image_quality !== undefined && clean.tag_image_quality !== "") {
        const parsedQuality = Number(clean.tag_image_quality);
        if (!Number.isNaN(parsedQuality)) {
          clean.tag_image_quality = parsedQuality;
        }
      }
      if (clean.tag_batch_max_assets !== undefined && clean.tag_batch_max_assets !== "") {
        const parsedBatch = Number(clean.tag_batch_max_assets);
        if (!Number.isNaN(parsedBatch)) {
          clean.tag_batch_max_assets = Math.max(1, Math.min(50000, parsedBatch));
        }
      }
      if (clean.tag_batch_project_concurrency !== undefined && clean.tag_batch_project_concurrency !== "") {
        const parsedConcurrency = Number(clean.tag_batch_project_concurrency);
        if (!Number.isNaN(parsedConcurrency)) {
          clean.tag_batch_project_concurrency = parsedConcurrency;
        }
      }
      if (clean.purge_assets_on_startup !== undefined) {
        clean.purge_assets_on_startup = String(clean.purge_assets_on_startup).toLowerCase() === "true";
      }
      const includeTypes = String(data.tag_include_types || "")
        .split(",")
        .map((entry) => entry.trim())
        .filter(Boolean);
      const excludeTypes = String(data.tag_exclude_types || "")
        .split(",")
        .map((entry) => entry.trim())
        .filter(Boolean);
      const exportIncludeTypes = String(data.export_include_types || "")
        .split(",")
        .map((entry) => entry.trim())
        .filter(Boolean);
      const hasExportExcludeTypes = Object.prototype.hasOwnProperty.call(data, "export_exclude_types");
      let exportExcludeTypes = String(data.export_exclude_types || "")
        .split(",")
        .map((entry) => entry.trim())
        .filter(Boolean);
      if (!hasExportExcludeTypes && !exportExcludeTypes.length) {
        exportExcludeTypes = ["Material", "MaterialInstance", "MaterialInstanceConstant"];
      }
      const catalogTypes = String(data.asset_type_catalog || "")
        .split(",")
        .map((entry) => entry.trim())
        .filter(Boolean);
      setSettingsTagIncludeTypes(includeTypes);
      setSettingsTagExcludeTypes(excludeTypes);
      setSettingsExportIncludeTypes(exportIncludeTypes);
      setSettingsExportExcludeTypes(exportExcludeTypes);
      setAssetTypeCatalog(catalogTypes);
      setHasApiKey(Boolean(data.has_api_key));
      setHasOpenAiKey(Boolean(data.has_openai_api_key));
      setHasOpenRouterKey(Boolean(data.has_openrouter_api_key));
      setHasGroqKey(Boolean(data.has_groq_api_key));
      setApiKeyInput("");
      setOpenaiKeyInput("");
      setOpenrouterKeyInput("");
      setGroqKeyInput("");
      setProjectFullCopy(Boolean(clean.default_full_project_copy));
      setSettings((prev) => ({ ...prev, ...clean }));
    });
    const storedViews = window.localStorage.getItem("assetViews");
    if (storedViews) {
      try {
        const parsed = JSON.parse(storedViews);
        if (Array.isArray(parsed)) {
          setSavedViews(parsed);
        }
      } catch (err) {
        console.error(err);
      }
    }
  }, []);
  useEffect(() => {
    if (!selectedEras.length && eraOptions.length) {
      setSelectedEras(eraOptions);
    }
  }, [eraOptions, selectedEras.length]);
  useEffect(() => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem("useSemanticSearch", String(useSemanticSearch));
    }
  }, [useSemanticSearch]);
  useEffect(() => {
    let active = true;
    const requestId = ++assetsRequestRef.current;
    toast.info("Loading assets...");
    isFetchingAssetsRef.current = true;
    setIsFetchingAssets(true);
    setIsLoadingMore(page > 1);
    fetchAssets(filterParams)
      .then((data) => {
        if (!active || requestId !== assetsRequestRef.current) return;
        if (page === 1) {
          setAssets(data.items || []);
        } else {
          setAssets((prev) => {
            const next = data.items || [];
            if (!next.length) return prev;
            const seen = new Set(prev.map((item) => item.id));
            const merged = prev.slice();
            next.forEach((item) => {
              if (!seen.has(item.id)) {
                seen.add(item.id);
                merged.push(item);
              }
            });
            return merged;
          });
        }
        if (pendingNext) {
          if ((data.items || []).length) {
            const nextItem = data.items[0];
            if (nextItem) {
              setSelectedAssetId(nextItem.id);
              setSelectedAsset(nextItem);
            }
          } else {
            toast.info("No more assets");
          }
          setPendingNext(false);
        }
        setTotalCount(data.total || 0);
        setTotalAll(data.total_all || 0);
        const totalForQuery = data.total || 0;
        const nextHasMore = totalForQuery > page * pageSize;
        setHasMore(nextHasMore);
        console.info("assets: loaded", {
          page,
          pageSize,
          total: data.total || 0,
          items: (data.items || []).length,
          hasMore: nextHasMore,
          view,
          selectedAssetId,
        });
        lastLoadedPageRef.current = page;
        pageRequestRef.current = false;
        toast.dismiss();
        setIsLoadingMore(false);
        isFetchingAssetsRef.current = false;
        setIsFetchingAssets(false);
      })
      .catch((err) => {
        if (active && requestId === assetsRequestRef.current) {
          console.error(err);
          toast.error("Failed to load assets");
          setIsLoadingMore(false);
          setPendingNext(false);
          pageRequestRef.current = false;
          isFetchingAssetsRef.current = false;
          setIsFetchingAssets(false);
        }
      });
    return () => {
      active = false;
    };
  }, [filterParams, page, pageSize, refreshKey]);
  useEffect(() => {
    if (!selectedAssetId) {
      setSelectedAsset(null);
      return;
    }
    if (selectedAssetId && window.history?.pushState) {
      const url = new URL(window.location.href);
      url.searchParams.set("asset", String(selectedAssetId));
      window.history.pushState({ assetId: selectedAssetId }, "", url.toString());
    }
    window.scrollTo({ top: 0, behavior: "auto" });
    const cached = assets.find((asset) => asset.id === selectedAssetId);
    if (cached) {
      setSelectedAsset(cached);
    }
    fetchAsset(selectedAssetId)
      .then((data) => {
        setSelectedAsset(data);
      })
      .catch(console.error);
  }, [selectedAssetId]);
  useEffect(() => {
    if (!selectedAssetId || selectedAsset) return;
    const cached = assets.find((asset) => asset.id === selectedAssetId);
    if (cached) {
      setSelectedAsset(cached);
    }
  }, [assets, selectedAssetId, selectedAsset]);
  useEffect(() => {
    if (!selectedAsset) return;
    setEditableTags(Array.isArray(selectedAsset.tags) ? selectedAsset.tags : []);
  }, [selectedAsset]);
  useEffect(() => {
    if (!selectedAssetId) return;
    if (assets.length && !assets.some((asset) => asset.id === selectedAssetId)) {
      closeDetail();
    }
  }, [assets, selectedAssetId]);
  useEffect(() => {
    const handlePopState = (event) => {
      const stateAssetId = event.state?.assetId;
      if (!stateAssetId) {
        closeDetail();
      }
    };
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);
  useEffect(() => {
    const url = new URL(window.location.href);
    const assetParam = url.searchParams.get("asset");
    const projectParam = url.searchParams.get("project_content");
    if (assetParam) {
      const parsed = Number(assetParam);
      if (!Number.isNaN(parsed)) {
        setSelectedAssetId(parsed);
      }
    }
    if (projectParam) {
      const parsed = Number(projectParam);
      if (!Number.isNaN(parsed)) {
        setView("assets");
        setSelectedProjects([String(parsed)]);
        resetPaging(true, true);
      }
    }
  }, []);
  const selectedIndex = selectedAssetId
    ? visibleAssets.findIndex((asset) => asset.id === selectedAssetId)
    : -1;
  useEffect(() => {
    const onKeyDown = (e) => {
      const tag = (e.target?.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select" || e.target?.isContentEditable) {
        return;
      }
      if ((e.key === "Backspace" || e.key === "Escape") && selectedAssetId) {
        e.preventDefault();
        closeDetail();
        return;
      }
      if (selectedAssetId && view === "assets") {
        if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
          e.preventDefault();
          handlePrevDetail();
          return;
        }
        if (e.key === "ArrowRight" || e.key === "ArrowDown") {
          e.preventDefault();
          handleNextDetail();
        }
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [selectedAssetId, view, selectedIndex, visibleAssets, isLoadingMore, totalCount, page]);
  useEffect(() => {
    if (!projectFiltersActive) return;
    if (!selectedAssetId) return;
    const allowed = new Set(selectedProjects.map((id) => String(id)));
    if (!allowed.has(String(selectedAsset?.project_id || ""))) {
      closeDetail();
    }
  }, [projectFiltersActive, selectedProjects, selectedAssetId, selectedAsset]);
  useEffect(() => {
    if (view !== "assets" || selectedAssetId || !hasMore) return;
    const target = loadMoreRef.current;
    if (!target) return;
    const observer = new IntersectionObserver(
      (entries) => {
        const entry = entries[0];
        if (!entry || !entry.isIntersecting) return;
        if (!isAssetGridVisible()) return;
        if (isLoadingMore || isFetchingAssetsRef.current) return;
        if (pageRequestRef.current) return;
        pageRequestRef.current = true;
        setIsLoadingMore(true);
        setPage((prev) => prev + 1);
      },
      { root: null, rootMargin: "200px", threshold: 0 }
    );
    observer.observe(target);
    return () => observer.disconnect();
  }, [view, selectedAssetId, hasMore, isLoadingMore, page, assets.length]);
  useEffect(() => {
    if (view !== "assets") {
      lastAssetScrollYRef.current = null;
    }
  }, [view]);
  useEffect(() => {
    if (view !== "assets") {
      setSelectedAssetIds([]);
    }
  }, [view]);
  useEffect(() => {
    if (selectedAssetId) {
      setSelectedAssetIds([]);
    }
  }, [selectedAssetId]);
  useEffect(() => {
    if (view !== "assets") return;
    if (selectedAssetId !== null) return;
    if (lastAssetScrollYRef.current === null) return;
    const targetY = lastAssetScrollYRef.current;
    requestAnimationFrame(() => {
      window.scrollTo({ top: targetY, behavior: "auto" });
    });
  }, [view, selectedAssetId]);
  useEffect(() => {
    if (view !== "assets" || selectedAssetId || !hasMore || isFetchingAssetsRef.current) return;
    const getScrollHeight = () =>
      Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);
    const handleScroll = () => {
      if (isLoadingMore || isFetchingAssetsRef.current) return;
      if (!isAssetGridVisible()) return;
      if (lastLoadedPageRef.current < page) return;
      if (pageRequestRef.current) return;
      const scrollBottom = window.innerHeight + window.scrollY;
      const threshold = getScrollHeight() - 200;
      if (scrollBottom >= threshold) {
        console.info("assets: scroll threshold hit", {
          scrollBottom,
          threshold,
          page,
          hasMore,
          isLoadingMore,
        });
        pageRequestRef.current = true;
        setIsLoadingMore(true);
        setPage((prev) => prev + 1);
      }
    };
    window.addEventListener("scroll", handleScroll);
    handleScroll();
    return () => window.removeEventListener("scroll", handleScroll);
  }, [view, selectedAssetId, hasMore, isLoadingMore]);
  useEffect(() => {
    if (view !== "assets" || selectedAssetId || !hasMore || isLoadingMore || isFetchingAssetsRef.current) return;
    if (lastLoadedPageRef.current < page) return;
    if (assets.length === 0) return;
    const getScrollHeight = () =>
      Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);
    const maybeLoadMore = () => {
      if (!isAssetGridVisible()) return;
      const scrollHeight = getScrollHeight();
      if (scrollHeight <= window.innerHeight + 200) {
        if (pageRequestRef.current) return;
        console.info("assets: short page load", {
          scrollHeight,
          viewport: window.innerHeight,
          page,
          hasMore,
        });
        pageRequestRef.current = true;
        setIsLoadingMore(true);
        setPage((prev) => prev + 1);
      }
    };
    maybeLoadMore();
    window.addEventListener("resize", maybeLoadMore);
    return () => window.removeEventListener("resize", maybeLoadMore);
  }, [view, selectedAssetId, hasMore, isLoadingMore, assets.length, page]);
  useEffect(() => {
    fetchProjectStats(statsParams)
      .then((data) => {
        const map = {};
        (data.items || []).forEach((item) => {
          map[item.project_id] = item;
        });
        setProjectStats(map);
        setProjectStatsSummary(data.summary || null);
      })
      .catch(console.error);
  }, [statsParams]);
  useEffect(() => {
    if (!migrateStatus?.assetId) return;
    if (migrateStatus.status === "done" || migrateStatus.status === "error") return;
    const handle = setInterval(() => {
      fetchMigrateStatus(migrateStatus.assetId)
        .then((data) => {
          setMigrateStatus({ ...data, assetId: migrateStatus.assetId });
        })
        .catch(() => {
          setMigrateStatus((prev) => prev);
        });
    }, 1000);
    return () => clearInterval(handle);
  }, [migrateStatus]);
  useEffect(() => {
    const activeIds = Object.entries(projectTagStatus)
      .filter(([, status]) => status?.status === "running" || status?.status === "queued")
      .map(([id]) => Number(id));
    if (!activeIds.length) return;
    const handle = setInterval(() => {
      activeIds.forEach((id) => {
        fetchProjectTagStatus(id)
          .then((data) => {
            setProjectTagStatus((prev) => {
              const prevStatus = prev[id] || {};
              const merged = {
                ...data,
                done: Math.max(prevStatus.done || 0, data.done || 0),
                total: Math.max(prevStatus.total || 0, data.total || 0),
              };
              return { ...prev, [id]: merged };
            });
          })
          .catch(() => {});
      });
    }, 1000);
    return () => clearInterval(handle);
  }, [projectTagStatus]);
  useEffect(() => {
    const prev = prevTagStatusRef.current || {};
    Object.entries(projectTagStatus).forEach(([id, status]) => {
      if (!status) return;
      const key = String(id);
      const prevStatus = prev[id]?.status;
      const done = Number(status.done || 0);
      const total = Number(status.total || 0);
      const errors = Number(status.errors || 0);
      const label = status.message
        ? String(status.message)
        : `Project ${id}: tags ${done}/${total} (errors: ${errors})`;
      const existing = tagToastIdsRef.current.get(key);
      if (status.status === "running" || status.status === "queued") {
        if (existing) {
          toast.update(existing, { render: label, autoClose: false });
        } else {
          const tid = toast.info(label, { autoClose: false });
          tagToastIdsRef.current.set(key, tid);
        }
        return;
      }
      if (status.status === "done" && prevStatus !== "done") {
        if (existing) {
          toast.update(existing, { render: label, type: "success", autoClose: 2500 });
          tagToastIdsRef.current.delete(key);
        } else {
          toast.success(label);
        }
        setRefreshKey((k) => k + 1);
        return;
      }
      if (status.status === "error" || status.status === "canceled") {
        const finalLabel = status.status === "canceled" ? `Project ${id}: tagging canceled` : label;
        if (existing) {
          toast.update(existing, { render: finalLabel, type: "error", autoClose: 3000 });
          tagToastIdsRef.current.delete(key);
        } else {
          toast.error(finalLabel);
        }
      }
    });
    prevTagStatusRef.current = projectTagStatus;
  }, [projectTagStatus]);
  function resetPaging(clearDetail = false, clearList = false) {
    setPage(1);
    setHasMore(true);
    pageRequestRef.current = false;
    if (clearList) {
      setAssets([]);
      setIsLoadingMore(false);
      setPendingNext(false);
    }
    if (clearDetail) {
      setSelectedAssetId(null);
      setSelectedAsset(null);
    }
  }
  function isAssetGridVisible() {
    const grid = document.querySelector(".asset-grid");
    if (!grid) return false;
    if (grid.offsetParent === null) return false;
    const style = window.getComputedStyle(grid);
    return style.display !== "none" && style.visibility !== "hidden";
  }
  function persistViews(next) {
    setSavedViews(next);
    window.localStorage.setItem("assetViews", JSON.stringify(next));
  }
  function handleSaveView() {
    const name = window.prompt("View name", "") || "";
    if (!name.trim()) return;
    const view = {
      id: `${Date.now()}`,
      name: name.trim(),
      query: queryInput,
      tag: "",
      selectedProjects,
      selectedTypes,
      selectedArtStyles,
      selectedEras,
      naniteFilter,
      collisionFilter,
      tileSize,
    };
    persistViews([view, ...savedViews]);
    setSelectedViewId(view.id);
    toast.info("View saved");
  }
  function handleApplyView(view) {
    if (!view) return;
    setQueryInput(view.query || "");
    if (!view.query && view.tag) {
      setQueryInput(view.tag);
    }
    setSelectedProjects(view.selectedProjects || []);
    setSelectedTypes(view.selectedTypes || []);
    setSelectedArtStyles(view.selectedArtStyles || ["regular", "stylized", "low poly"]);
    setSelectedEras(view.selectedEras || eraOptions);
    setNaniteFilter(view.naniteFilter || "all");
    setCollisionFilter(view.collisionFilter || "all");
    if (view.tileSize) {
      setTileSize(view.tileSize);
    }
    setSelectedViewId(view.id);
    setSelectedAssetId(null);
    setSelectedAsset(null);
    resetPaging(true, true);
    toast.info(`View loaded: ${view.name}`);
  }
  function handleDeleteView(viewId) {
    const next = savedViews.filter((view) => view.id !== viewId);
    persistViews(next);
    if (selectedViewId === viewId) {
      setSelectedViewId("");
    }
  }
  function handleSelectViewChange(value) {
    setSelectedViewId(value);
    const selected = savedViews.find((view) => view.id === value);
    if (selected) {
      handleApplyView(selected);
    }
  }
  function handleProviderChange(nextProvider) {
    setSettings((prev) => ({ ...prev, provider: nextProvider }));
  }
  async function handleCreateProject(e) {
    e.preventDefault();
    const tags = projectTags
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean);
    try {
      const project = await createProject({
        name: projectName,
        link: projectLink,
        tags,
        art_style: projectArtStyle || "regular",
        source_path: projectSourcePath || undefined,
        source_folder: projectSourceFolder || undefined,
        full_project_copy: projectFullCopy,
        is_ai_generated: createIsAi,
      });
      if (projectScreenshotFile || projectScreenshotUrl) {
        await uploadProjectScreenshot(project.id, projectScreenshotFile, projectScreenshotUrl);
      }
      if (project.copy_started) {
        setCopyStatus({ status: "queued", copied: 0, total: 0, projectId: project.id });
      }
      const fresh = await fetchProjects();
      setProjects(fresh);
      setSelectedProjects(fresh.map((project) => String(project.id)));
      setProjectName("");
      setProjectLink("");
      setProjectTags("");
      setProjectArtStyle("");
      setProjectScreenshotUrl("");
      setProjectScreenshotFile(null);
      setProjectSourcePath("");
      setProjectSourceFolder("");
      setProjectFullCopy(Boolean(settings.default_full_project_copy));
      setCreateIsAi(false);
      setShowCreateProject(false);
      toast.info("Project created");
    } catch (err) {
      console.error(err);
      toast.error("Project creation failed");
    }
  }
  async function handleResetDatabase() {
    try {
      const result = await resetDatabase();
      const backupPath = result?.backup_path;
      toast.success(backupPath ? `Database reset. Backup: ${backupPath}` : "Database reset");
      const freshProjects = await fetchProjects();
      setProjects(freshProjects);
      setSelectedProjects(freshProjects.map((project) => String(project.id)));
      setAssets([]);
      setSelectedAssetId(null);
      setSelectedAsset(null);
      setTotalCount(0);
      setTotalAll(0);
    } catch (err) {
      console.error(err);
      toast.error("Database reset failed");
    }
  }
  async function handleRefreshProjectSizes() {
    if (refreshingProjectSizes) return;
    setRefreshingProjectSizes(true);
    try {
      const fresh = await fetchProjects({ include_sizes: "1" });
      const statsData = await fetchProjectStats(statsParams);
      const statsMap = {};
      (statsData.items || []).forEach((item) => {
        statsMap[item.project_id] = item;
      });
      setProjects(fresh);
      setSelectedProjects(fresh.map((project) => String(project.id)));
      setProjectStats(statsMap);
      setProjectStatsSummary(statsData.summary || null);
      toast.success("Source sizes refreshed");
    } catch (err) {
      console.error(err);
      toast.error("Failed to refresh source sizes");
    } finally {
      setRefreshingProjectSizes(false);
    }
  }
  async function handleRestartServer() {
    if (restartingServer) return;
    if (!window.confirm("Restart backend server now?")) return;
    setRestartingServer(true);
    try {
      await restartServer();
      toast.warn("Server restart requested. UI will reconnect in a few seconds.");
    } catch (err) {
      console.error(err);
      toast.error(`Server restart failed: ${err.message || "unknown error"}`);
      setRestartingServer(false);
      return;
    }
    window.setTimeout(() => setRestartingServer(false), 8000);
  }
  async function handleUpload(e) {
    e.preventDefault();
    try {
      if (!uploadFiles.length) return;
      const total = uploadFiles.length;
      const hw = typeof navigator !== "undefined" ? Number(navigator.hardwareConcurrency || 4) : 4;
      const concurrency = Math.max(1, Math.min(4, Math.floor(hw / 2)));
        const progressToastId = toast.info(
          `Uploading ${total} assets... (parallel ${concurrency})`,
          { autoClose: false }
        );
        setUploadProgress({ current: 0, total, percent: 0 });
        let completed = 0;
        let active = 0;
        let index = 0;
        const errors = [];
        const successes = [];
        await new Promise((resolve) => {
          const next = () => {
            if (index >= total && active === 0) {
              resolve();
              return;
            }
            while (active < concurrency && index < total) {
              const file = uploadFiles[index++];
              active += 1;
              uploadAsset(file, uploadProject || undefined)
                .then(() => {
                  successes.push(file.name || "asset");
                  completed += 1;
                  setUploadProgress({
                    current: completed,
                    total,
                    percent: Math.round((completed / total) * 100),
                  });
                  toast.update(progressToastId, {
                    render: `Imported ${completed}/${total}`,
                  });
                })
                .catch((err) => {
                  errors.push(err);
                })
                .finally(() => {
                active -= 1;
                next();
              });
          }
        };
        next();
      });
      setUploadFiles([]);
      setUploadProject("");
      resetPaging();
      setUploadProgress(null);
        if (errors.length) {
          toast.error(`Upload finished with ${errors.length} error(s)`);
          toast.dismiss(progressToastId);
        } else {
          toast.update(progressToastId, {
            render: `Imported ${successes.length} assets`,
            autoClose: 2000,
          });
        }
      } catch (err) {
      console.error(err);
      setUploadProgress(null);
      toast.error("Upload failed");
    }
  }
  async function handleAddTag() {
    const value = tagInput.trim().toLowerCase();
    if (!value) return;
    if (editableTags.includes(value)) {
      setTagInput("" );
      return;
    }
    const next = [...editableTags, value];
    setEditableTags(next);
    setTagInput("");
    if (!selectedAsset) return;
    try {
      await mergeAssetTags({ assetIds: [selectedAsset.id], tags: [value] });
      setSelectedAsset((prev) => (prev ? { ...prev, tags: next } : prev));
      setAssets((prev) => prev.map((a) => (a.id === selectedAsset.id ? { ...a, tags: next } : a)));
    } catch (err) {
      console.error(err);
      toast.error("Tag merge failed");
    }
  }
  async function handleRemoveTag(tag) {
    const next = editableTags.filter((t) => t !== tag);
    setEditableTags(next);
    await handleSaveTags(next);
  }
  async function handleSaveTags(tags) {
    if (!selectedAsset) return;
    try {
      const result = await updateTags(selectedAsset.id, tags);
      if (result?.tags) {
        setSelectedAsset((prev) => (prev ? { ...prev, tags: result.tags } : prev));
        setAssets((prev) => prev.map((a) => (a.id === selectedAsset.id ? { ...a, tags: result.tags } : a)));
      }
    } catch (err) {
      console.error(err);
      toast.error("Tag update failed");
    }
  }
  async function handleGenerateTags(assetId) {
    try {
      const result = await generateTags(assetId);
      if (result?.asset) {
        setAssets((prev) => prev.map((asset) => (asset.id === assetId ? result.asset : asset)));
        setSelectedAsset(result.asset);
      } else if (result?.tags) {
        setAssets((prev) =>
          prev.map((asset) => (asset.id === assetId ? { ...asset, tags: result.tags } : asset))
        );
        setSelectedAsset((prev) => (prev ? { ...prev, tags: result.tags } : prev));
      }
      await new Promise((resolve) => setTimeout(resolve, 200));
      const refreshed = await fetchAsset(assetId);
      setAssets((prev) => prev.map((asset) => (asset.id === assetId ? refreshed : asset)));
      setSelectedAsset(refreshed);
      setRefreshKey((key) => key + 1);
      toast.info("Tags updated");
    } catch (err) {
      console.error(err);
      toast.error(`Tag generation failed: ${err.message || "unknown error"}`);
    }
  }
  async function handleTagMissing(projectId) {
    try {
      await tagProjectMissing(projectId);
      setProjectTagStatus((prev) => ({ ...prev, [projectId]: { status: "queued", done: 0, total: 0, errors: 0 } }));
    } catch (err) {
      toast.error(`Tag missing failed: ${err.message || "unknown error"}`);
    }
  }
  async function handleRetagAll(projectId) {
    if (!window.confirm("Retag all assets? Existing tags will be replaced.")) return;
    try {
      await retagProject(projectId);
      setProjectTagStatus((prev) => ({ ...prev, [projectId]: { status: "queued", done: 0, total: 0, errors: 0 } }));
    } catch (err) {
      toast.error(`Retag failed: ${err.message || "unknown error"}`);
    }
  }
  async function handleRegenerateProjectEmbeddings(projectId) {
    try {
      await regenerateProjectEmbeddings(projectId);
      toast.info("Regenerating semantic embeddings...");
    } catch (err) {
      toast.error(`Embedding regen failed: ${err.message || "unknown error"}`);
    }
  }
  async function handleRegenerateEmbeddingsAll() {
    if (!window.confirm("Regenerate embeddings for all assets?")) return;
    try {
      await regenerateEmbeddingsAll();
      toast.info("Semantic rebuild scheduled for next server restart.");
    } catch (err) {
      toast.error(`Embedding regen failed: ${err.message || "unknown error"}`);
    }
  }
  async function handleTagMissingAllProjects() {
    const missingAssets = Number(tagStatsAggregate?.assetsWithoutTags || 0);
    const totalAssets = Number(tagStatsAggregate?.totalAssets || 0);
    const confirmMessage = missingAssets > 0
      ? `Tag missing assets for all projects? This targets about ${missingAssets} assets without tags${totalAssets > 0 ? ` (of ${totalAssets} total)` : ""} and may take a while.`
      : "Tag missing assets for all projects? This can take a while.";
    if (!window.confirm(confirmMessage)) return;
    try {
      await tagMissingAllProjects();
      toast.info("Tagging missing assets for all projects... Embeddings are not generated. Use Rebuild semantic if needed.");
    } catch (err) {
      toast.error(`Tag missing failed: ${err.message || "unknown error"}`);
    }
  }
  async function handleTranslateNameTagsAll() {
    if (!window.confirm("Translate asset names to tags (LLM) for all projects?")) return;
    try {
      await translateAllNameTags();
      toast.info("Translating asset names to tags (all projects)...");
    } catch (err) {
      toast.error(`Translate names failed: ${err.message || "unknown error"}`);
    }
  }
  async function handleTranslateNameTagsAllMissing() {
    if (!window.confirm("Translate asset names to tags (LLM) only for assets not processed yet?")) return;
    try {
      await translateAllNameTagsMissing();
      toast.info("Translating missing asset names to tags (all projects)...");
    } catch (err) {
      toast.error(`Translate names missing failed: ${err.message || "unknown error"}`);
    }
  }
  async function handleNameTagsAllSimple() {
    if (!window.confirm("Generate tags from asset names for all projects? (no translation)")) return;
    try {
      await nameTagsAllSimple();
      toast.info("Generating tags from names (all projects)...");
    } catch (err) {
      toast.error(`Name->tags failed: ${err.message || "unknown error"}`);
    }
  }
  async function handleNameTagsAllSimpleMissing() {
    if (!window.confirm("Generate tags from names only for assets with too few tags?")) return;
    try {
      await nameTagsAllSimpleMissing();
      toast.info("Generating missing name-based tags (all projects)...");
    } catch (err) {
      toast.error(`Name->tags missing failed: ${err.message || "unknown error"}`);
    }
  }
  async function handleTranslateAllTags() {
    if (!window.confirm("Translate existing tags for all projects (LLM)?")) return;
    try {
      await translateAllTags();
      toast.info("Translating existing tags (all projects)...");
    } catch (err) {
      toast.error(`Translate tags failed: ${err.message || "unknown error"}`);
    }
  }
  async function handleTranslateAllTagsMissing() {
    if (!window.confirm("Translate tags only for assets without translated tags?")) return;
    try {
      await translateAllTagsMissing();
      toast.info("Translating missing tags (all projects)...");
    } catch (err) {
      toast.error(`Translate tags missing failed: ${err.message || "unknown error"}`);
    }
  }
  async function handleDeleteTask(taskId) {
    try {
      await deleteTask(taskId);
      setTasks((prev) => prev.filter((item) => item.id !== taskId));
    } catch (err) {
      toast.error(`Delete task failed: ${err.message || "unknown error"}`);
    }
  }
  async function handleCancelTask(taskId) {
    try {
      await cancelTask(taskId);
      const data = await fetchTasks();
      setTasks(data.items || []);
    } catch (err) {
      toast.error(`Cancel failed: ${err.message || "unknown error"}`);
    }
  }

  async function handleCleanupTasks() {
    try {
      await cleanupTasks();
      const data = await fetchTasks();
      setTasks(data.items || []);
    } catch (err) {
      toast.error(`Cleanup failed: ${err.message || "unknown error"}`);
    }
  }

  async function handleTranslateNameTagsProject(projectId) {
    try {
      await translateProjectNameTags(projectId);
      toast.info("Translating asset names to tags...");
    } catch (err) {
      toast.error(`Translate names failed: ${err.message || "unknown error"}`);
    }
  }
  async function handleTranslateNameTagsProjectMissing(projectId) {
    try {
      await translateProjectNameTagsMissing(projectId);
      toast.info("Translating missing asset names to tags...");
    } catch (err) {
      toast.error(`Translate names missing failed: ${err.message || "unknown error"}`);
    }
  }

  async function handleRepairOpenAiQueue() {
    try {
      const task123 = tasks.find((t) => Number(t.id) === 123);
      const openaiTaskId = task123 ? 123 : null;
      const res = await enqueueOpenAiRecovery({
        flow: "translate_name_tags",
        taskId: openaiTaskId,
        limit: 500,
        staleMinutes: 120,
      });
      toast.info(`Recovery queued (task ${res.task_id})`);
    } catch (err) {
      toast.error(`Recovery enqueue failed: ${err.message || "unknown error"}`);
    }
  }
  async function handleNameTagsProjectSimple(projectId) {
    try {
      await nameTagsProjectSimple(projectId);
      toast.info("Generating tags from names...");
    } catch (err) {
      toast.error(`Name->tags failed: ${err.message || "unknown error"}`);
    }
  }
  async function handleNameTagsProjectSimpleMissing(projectId) {
    try {
      await nameTagsProjectSimpleMissing(projectId);
      toast.info("Generating missing name-based tags...");
    } catch (err) {
      toast.error(`Name->tags missing failed: ${err.message || "unknown error"}`);
    }
  }
  async function handleTranslateTagsProject(projectId) {
    try {
      await translateProjectTags(projectId);
      toast.info("Translating existing tags...");
    } catch (err) {
      toast.error(`Translate tags failed: ${err.message || "unknown error"}`);
    }
  }
  async function handleTranslateTagsProjectMissing(projectId) {
    try {
      await translateProjectTagsMissing(projectId);
      toast.info("Translating missing tags...");
    } catch (err) {
      toast.error(`Translate tags missing failed: ${err.message || "unknown error"}`);
    }
  }
  async function handleTestLlmTags() {
    try {
      const overrideSettings = { ...settings };
      if (apiKeyInput.trim()) overrideSettings.api_key = apiKeyInput.trim();
      if (openaiKeyInput.trim()) overrideSettings.openai_api_key = openaiKeyInput.trim();
      if (openrouterKeyInput.trim()) overrideSettings.openrouter_api_key = openrouterKeyInput.trim();
      if (groqKeyInput.trim()) overrideSettings.groq_api_key = groqKeyInput.trim();
      ["api_key", "openai_api_key", "openrouter_api_key", "groq_api_key"].forEach((key) => {
        const raw = overrideSettings[key];
        if (!raw) {
          delete overrideSettings[key];
          return;
        }
        if (String(raw).replace(/\*/g, "").trim().length === 0) {
          delete overrideSettings[key];
        }
      });
      const result = await testLlmTags(llmTestImage, overrideSettings);
      setLlmTestResult(result);
      const tags = (result?.tags || []).join(", ");
      toast.info(tags ? `Test tags: ${tags}` : "Test complete (no tags)");
    } catch (err) {
      toast.error(`LLM test failed: ${err.message || "unknown error"}`);
    }
  }
  async function handleUpdateProject(projectId) {
    const tags = editTags
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean);
    try {
      await updateProject(projectId, {
        name: editName,
        link: editLink,
        tags,
        art_style: editArtStyle,
        source_path: editSourcePath || undefined,
        source_folder: editSourceFolder || undefined,
        full_project_copy: editFullCopy,
        is_ai_generated: editIsAi,
      });
      if (editScreenshotFile || (editScreenshotUrl || "").trim()) {
        await uploadProjectScreenshot(projectId, editScreenshotFile, (editScreenshotUrl || "").trim());
      }
      const fresh = await fetchProjects();
      setProjects(fresh);
      setSelectedProjects(fresh.map((project) => String(project.id)));
      setEditingProjectId(null);
      setEditScreenshotUrl("");
      setEditScreenshotFile(null);
      toast.info("Project saved");
    } catch (err) {
      console.error(err);
      toast.error(`Project save failed: ${err.message || "unknown error"}`);
    }
  }
  function startEditProject(project) {
    setEditingProjectId(project.id);
    setEditName(project.name || "");
    setEditLink(project.link || "");
    setEditTags((project.tags || []).join(", "));
    setEditArtStyle(project.art_style || "");
    setEditSourcePath(project.source_path || "");
    setEditSourceFolder(project.source_folder || "");
    setEditIsAi(Boolean(project.is_ai_generated));
    setEditFullCopy(Boolean(project.full_project_copy));
    setEditScreenshotUrl("");
    setEditScreenshotFile(null);
  }
  async function handleSaveSettings(e) {
    e.preventDefault();
    try {
      const payload = {
        ...settings,
        tag_include_types: settingsTagIncludeTypes.join(","),
        tag_exclude_types: settingsTagExcludeTypes.join(","),
        export_include_types: settingsExportIncludeTypes.join(","),
        export_exclude_types: settingsExportExcludeTypes.join(","),
        asset_type_catalog: assetTypeCatalog.join(","),
      };
      if (payload.skip_export_if_on_server !== undefined) {
        payload.skip_export_if_on_server = payload.skip_export_if_on_server ? "true" : "false";
      }
      if (payload.export_overwrite_zips !== undefined) {
        payload.export_overwrite_zips = payload.export_overwrite_zips ? "true" : "false";
      }
      if (payload.export_upload_after_export !== undefined) {
        payload.export_upload_after_export = payload.export_upload_after_export ? "true" : "false";
      }
      if (payload.default_full_project_copy !== undefined) {
        payload.default_full_project_copy = payload.default_full_project_copy ? "true" : "false";
      }
      if (payload.tag_use_batch_mode !== undefined) {
        payload.tag_use_batch_mode = payload.tag_use_batch_mode ? "true" : "false";
      }
      if (payload.tag_batch_max_assets !== undefined && payload.tag_batch_max_assets !== "") {
        const parsedBatch = Number(payload.tag_batch_max_assets);
        if (Number.isNaN(parsedBatch)) {
          delete payload.tag_batch_max_assets;
        } else {
          payload.tag_batch_max_assets = Math.max(1, Math.min(50000, parsedBatch));
        }
      }
      if (payload.tag_batch_project_concurrency !== undefined && payload.tag_batch_project_concurrency !== "") {
        const parsedConcurrency = Number(payload.tag_batch_project_concurrency);
        if (Number.isNaN(parsedConcurrency)) {
          delete payload.tag_batch_project_concurrency;
        } else {
          payload.tag_batch_project_concurrency = parsedConcurrency;
        }
      }
      if (apiKeyInput.trim()) {
        payload.api_key = apiKeyInput.trim();
      } else {
        delete payload.api_key;
      }
      if (openaiKeyInput.trim()) {
        payload.openai_api_key = openaiKeyInput.trim();
      } else {
        delete payload.openai_api_key;
      }
      if (openrouterKeyInput.trim()) {
        payload.openrouter_api_key = openrouterKeyInput.trim();
      } else {
        delete payload.openrouter_api_key;
      }
      if (groqKeyInput.trim()) {
        payload.groq_api_key = groqKeyInput.trim();
      } else {
        delete payload.groq_api_key;
      }
      payload.openai_base_url = settings.openai_base_url;
      payload.openrouter_base_url = settings.openrouter_base_url;
      payload.groq_base_url = settings.groq_base_url;
      payload.ollama_base_url = settings.ollama_base_url;
      payload.openai_model = settings.openai_model;
      payload.openrouter_model = settings.openrouter_model;
      payload.groq_model = settings.groq_model;
      payload.ollama_model = settings.ollama_model;
      await updateSettings(payload);
      if (apiKeyInput.trim()) {
        setHasApiKey(true);
        setApiKeyInput("");
      }
      if (openaiKeyInput.trim()) {
        setHasOpenAiKey(true);
        setOpenaiKeyInput("");
      }
      if (openrouterKeyInput.trim()) {
        setHasOpenRouterKey(true);
        setOpenrouterKeyInput("");
      }
      if (groqKeyInput.trim()) {
        setHasGroqKey(true);
        setGroqKeyInput("");
      }
      toast.info("Settings saved");
    } catch (err) {
      console.error(err);
      toast.error("Settings save failed");
    }
  }
  async function handleMigrateAsset(assetId) {
    const asset = assets.find((item) => item.id === assetId) || selectedAsset;
    if (!asset) return;
    const snapshotId = asset.hash_main_blake3 || assetId;
    const mode = overwriteExisting ? "override" : "skip";
    const base = (settings.import_base_url || "http://127.0.0.1:9090").replace(/\/$/, "");
    const url = `${base}/asset-import?id=${encodeURIComponent(snapshotId)}&mode=${mode}`;
    try {
      const res = await fetch(url);
      const data = await res.json().catch(() => ({}));
      if (!res.ok || data.ok === false) {
        throw new Error(data.error || `HTTP ${res.status}`);
      }
      setTimeout(() => {
        handleSelectAsset(assetId, { silent: true });
      }, 200);
    } catch (err) {
      console.error(err);
      toast.error(`Migrate failed: ${err.message || "unknown error"}`);
    }
  }
  async function handleSelectAsset(assetId, options = {}) {
    const asset = assets.find((item) => item.id === assetId) || selectedAsset;
    if (!asset) return;
    let objectPath = asset.meta?.object_path || "";
    if (!objectPath && asset.meta?.package) {
      const pkg = String(asset.meta.package);
      const name = pkg.split("/").pop();
      if (name) {
        objectPath = `${pkg}.${name}`;
      }
    }
    if (!objectPath) {
      toast.error("No object path in asset meta");
      return;
    }
    const base = (settings.import_base_url || "http://127.0.0.1:9090").replace(/\/$/, "");
    const url = `${base}/asset-select?path=${encodeURIComponent(objectPath)}`;
    const silent = options?.silent;
    try {
      const res = await fetch(url);
      const data = await res.json().catch(() => ({}));
      if (!res.ok || data.ok === false) {
        throw new Error(data.error || `HTTP ${res.status}`);
      }
      if (!silent) {
        toast.info("Selected in editor");
      }
    } catch (err) {
      console.error(err);
      toast.error(`Select failed: ${err.message || "unknown error"}`);
    }
  }
  function handleExportAssetZip(assetId) {
    const url = `${API_BASE}/download/${assetId}.zip?layout=project`;
    window.open(url, "_blank", "noopener");
    toast.info("Export started");
  }
  async function handleDeleteAsset(assetId) {
    if (!window.confirm("Delete asset from database? Files will remain.")) return;
    try {
      await deleteAsset(assetId);
      setAssets((prev) => prev.filter((asset) => asset.id !== assetId));
      setSelectedAssetId(null);
      setSelectedAsset(null);
      toast.info("Asset deleted");
    } catch (err) {
      console.error(err);
      toast.error("Asset delete failed");
    }
  }
  async function handleDeleteProjectAssets(projectId) {
    if (!window.confirm("Delete all assets in this project from database? Files will remain.")) return;
    try {
      await deleteProjectAssets(projectId);
      setSelectedAssetId(null);
      setSelectedAsset(null);
      resetPaging();
      toast.info("Project assets deleted");
    } catch (err) {
      console.error(err);
      toast.error("Project assets delete failed");
    }
  }
  function markProjectImported(projectId) {
    setProjectImported((prev) => {
      const next = { ...prev, [projectId]: true };
      localStorage.setItem("ameb_project_imported", JSON.stringify(next));
      return next;
    });
  }
  async function handleDeleteProject(projectId) {
    if (!window.confirm("Delete project and all its assets from the database? Files will remain.")) return;
    try {
      await deleteProject(projectId);
      const fresh = await fetchProjects();
      setProjects(fresh);
      setSelectedProjects(fresh.map((project) => String(project.id)));
      resetPaging(true, true);
      setRefreshKey((key) => key + 1);
      resetPaging();
      toast.info("Project deleted");
    } catch (err) {
      console.error(err);
      toast.error("Project delete failed");
    }
  }
  async function handleOpenProject(project) {
    toast.info("Opening folder...");
    try {
      await openProject(project.id);
    } catch (err) {
      console.error(err);
      toast.error("Open folder failed");
    }
  }
  async function handleOpenProjectSource(project) {
    toast.info("Opening source folder...");
    try {
      await openProject(project.id, "source");
    } catch (err) {
      console.error(err);
      toast.error("Open source folder failed");
    }
  }
  async function handleExportProjects() {
    try {
      const csv = await exportProjects();
      const blob = new Blob([csv], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "projects.csv";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      toast.info("Projects exported");
    } catch (err) {
      console.error(err);
      toast.error(`Export failed: ${err.message || "unknown error"}`);
    }
  }
  async function handleImportProjects() {
    if (!projectImportFile) {
      toast.error("Select a CSV file first");
      return;
    }
    try {
      const result = await importProjects(projectImportFile);
      if (result?.task_id) {
        toast.info(`Import queued (task ${result.task_id})`);
      } else {
        const skipped = result.skipped ?? 0;
        toast.info(
          `Imported projects: ${result.created}, skipped: ${skipped}, errors: ${result.errors}`
        );
      }
      setProjectImportFile(null);
      if (!result?.task_id) {
        const fresh = await fetchProjects();
        setProjects(fresh);
        setSelectedProjects(fresh.map((project) => String(project.id)));
      }
    } catch (err) {
      console.error(err);
      toast.error(`Import failed: ${err.message || "unknown error"}`);
    }
  }
  async function handleGenerateSetcard(projectId) {
    try {
      const res = await generateProjectSetcard(projectId, true);
      if (res.setcard_url) {
        setProjectPreview(resolveApiUrl(res.setcard_url));
      }
      toast.info("Setcard generated");
    } catch (err) {
      console.error(err);
      toast.error("Setcard generation failed");
    }
  }
  async function handleReimportProject(project) {
    try {
      let payload = {};
      if (!project.source_path && !project.source_folder) {
        const sourceInput = window.prompt("Source content/pack path", "") || "";
        if (!sourceInput.trim()) {
          toast.error("Source path missing");
          return;
        }
        payload = { source_path: sourceInput.trim() };
      }
      await reimportProject(project.id, payload);
      markProjectImported(project.id);
      setCopyStatus({ status: "queued", copied: 0, total: 0, projectId: project.id });
      toast.info("Reimport started");
    } catch (err) {
      console.error(err);
      toast.error(`Reimport failed: ${err.message || "unknown error"}`);
    }
  }
  async function handleRunExportCmd(project) {
    try {
      if (!settings.ue_cmd_path) {
        toast.error("Set Unreal command path in Settings");
        return;
      }
      const res = await runProjectExportCmd(project.id, {});
      toast.info("Export command started");
      if (res?.command) {
        console.log("Export command:", res.command);
      }
    } catch (err) {
      console.error(err);
      toast.error(`Export command failed: ${err.message || "unknown error"}`);
    }
  }
  async function handleExportTags() {
    try {
      const csv = await exportTags({
        hashType: tagExportHash,
        projectId: tagExportProjectId || undefined,
      });
      const blob = new Blob([csv], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `tags_${tagExportHash}${tagExportProjectId ? `_p${tagExportProjectId}` : ""}.csv`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      toast.info("Tags exported");
    } catch (err) {
      console.error(err);
      toast.error(`Export failed: ${err.message || "unknown error"}`);
    }
  }
  async function handleImportTags() {
    if (!tagImportFile) {
      toast.error("Select a CSV file");
      return;
    }
    try {
      const result = await importTags({
        file: tagImportFile,
        hashType: tagImportHash,
        projectId: tagImportProjectId || undefined,
        mode: tagImportMode,
      });
      if (result?.task_id) {
        toast.info(`Tags import queued (task ${result.task_id})`);
      } else {
        toast.info(`Imported tags: ${result.updated}, missing: ${result.missing}`);
      }
      setTagImportFile(null);
      if (!result?.task_id) {
        resetPaging();
      }
    } catch (err) {
      console.error(err);
      toast.error(`Import failed: ${err.message || "unknown error"}`);
    }
  }
  async function handleClearAllTags() {
    if (!window.confirm("Delete ALL tags for ALL assets? This cannot be undone.")) return;
    try {
      const result = await clearAllTags();
      if (result?.task_id) {
        toast.info(`Tag clear queued (task ${result.task_id})`);
      } else {
        toast.info("Tags cleared");
      }
      resetPaging();
    } catch (err) {
      console.error(err);
      toast.error(`Clear tags failed: ${err.message || "unknown error"}`);
    }
  }
  function toggleAssetSelection(assetId) {
    setSelectedAssetIds((prev) => {
      if (prev.includes(assetId)) {
        return prev.filter((id) => id !== assetId);
      }
      return [...prev, assetId];
    });
  }
  function handleAssetClick(asset, event) {
    if (event && (event.ctrlKey || event.metaKey || event.shiftKey)) {
      event.preventDefault();
      event.stopPropagation();
      toggleAssetSelection(asset.id);
      return;
    }
    setSelectedAssetIds([]);
    lastAssetScrollYRef.current = window.scrollY;
    setSelectedAssetId(asset.id);
    setSelectedAsset(asset);
  }
  async function handleAssetContextMenu(asset, event) {
    event.preventDefault();
    let ids = selectedAssetIds;
    if (!ids.includes(asset.id)) {
      ids = [asset.id];
      setSelectedAssetIds(ids);
    }
    const input = window.prompt(`Tag hinzufügen für ${ids.length} Asset(s)`);
    if (!input) return;
    const tags = input
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean);
    if (!tags.length) return;
    try {
      const result = await mergeAssetTags({ assetIds: ids, tags });
      toast.info(`Tagged ${result.updated} assets (missing ${result.missing}, errors ${result.errors})`);
      setSelectedAssetIds([]);
      resetPaging();
    } catch (err) {
      console.error(err);
      toast.error(`Tag update failed: ${err.message || "unknown error"}`);
    }
  }
  const selectedProject = selectedAsset ? projectMap.get(selectedAsset.project_id) : null;
  const projectNameWrap = Math.max(10, Number(settings.project_name_wrap_chars ?? 90) || 90);
  function closeDetail() {
    setSelectedAssetId(null);
    setSelectedAsset(null);
    if (window.history?.replaceState) {
      const url = new URL(window.location.href);
      url.searchParams.delete("asset");
      window.history.replaceState({}, "", url.toString());
    }
  }
  function handlePrevDetail() {
    if (selectedIndex <= 0) return;
    setSelectedAssetId(visibleAssets[selectedIndex - 1].id);
  }
  function handleNextDetail() {
    if (selectedIndex < 0) return;
    if (selectedIndex < visibleAssets.length - 1) {
      setSelectedAssetId(visibleAssets[selectedIndex + 1].id);
      return;
    }
    if (!isLoadingMore && !pageRequestRef.current && visibleAssets.length < totalCount) {
      pageRequestRef.current = true;
      setPendingNext(true);
      setIsLoadingMore(true);
      setPage((prev) => prev + 1);
    }
  }
  function handleTagClick(tag) {
    if (!tag) return;
    setQueryInput(tag);
    if (!useSemanticSearch) {
      resetPaging(true, true);
    }
    handleViewClick("assets", true);
  }

  function handleViewProjectAssets(projectId) {
    setSelectedProjects([String(projectId)]);
    setSelectedAssetId(null);
    setSelectedAsset(null);
    resetPaging(true, true);
    if (window.history?.pushState) {
      const url = new URL(window.location.href);
      url.searchParams.delete("asset");
      url.searchParams.set("project_content", String(projectId));
      window.history.pushState({ projectContent: projectId }, "", url.toString());
    }
    handleViewClick("assets", true);
  }
  function formatApproxSize(meta) {
    const size = meta?.mesh?.approx_size_cm;
    if (!size) return "";
    const x = Number(size.x).toFixed(1);
    const y = Number(size.y).toFixed(1);
    const z = Number(size.z).toFixed(1);
    return `${x} x ${y} x ${z} cm`;
  }
    function formatProjectBytes(bytes) {
    if (bytes === undefined || bytes === null) return "";
    const value = Number(bytes);
    if (!Number.isFinite(value) || value <= 0) return "0.00 GB";
    return `${(value / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  }
function formatSizeGb(bytes) {
    const value = Number(bytes);
    if (!Number.isFinite(value) || value <= 0) return "0.00";
    return (value / (1024 * 1024 * 1024)).toFixed(2);
  }
  function toggleAllProjects(checked) {
    if (checked) {
      setSelectedProjects(projects.map((project) => String(project.id)));
    } else {
      setSelectedProjects([]);
    }
    resetPaging(true, true);
  }
  function toggleProject(id) {
    setSelectedProjects((prev) => {
      const value = String(id);
      if (prev.includes(value)) {
        return prev.filter((item) => item !== value);
      }
      return [...prev, value];
    });
    resetPaging(true, true);
  }
  function toggleAllTypes(checked) {
    if (checked) {
      setSelectedTypes(assetTypes.slice());
    } else {
      setSelectedTypes([]);
    }
    resetPaging(true, true);
  }
  function toggleType(value) {
    setSelectedTypes((prev) => {
      if (prev.includes(value)) {
        return prev.filter((item) => item !== value);
      }
      return [...prev, value];
    });
    resetPaging(true, true);
  }
  function toggleTypeList(value, list, setList, otherList, setOtherList) {
    const exists = list.includes(value);
    if (exists) {
      setList(list.filter((item) => item !== value));
    } else {
      setList([...list, value]);
      if (otherList.includes(value)) {
        setOtherList(otherList.filter((item) => item !== value));
      }
    }
  }
  function addAssetTypeCatalog() {
    const value = assetTypeCatalogInput.trim();
    if (!value) return;
    if (!assetTypeCatalog.includes(value)) {
      const next = [...assetTypeCatalog, value];
      setAssetTypeCatalog(next);
      if (!assetTypes.includes(value)) {
        setAssetTypes((prev) => [...prev, value]);
        setSelectedTypes((prev) => (prev.includes(value) ? prev : [...prev, value]));
      }
    }
    setAssetTypeCatalogInput("");
  }
  function removeAssetTypeCatalog(value) {
    setAssetTypeCatalog(assetTypeCatalog.filter((item) => item !== value));
  }
  function refreshTypes() {
    fetchAssetTypes()
      .then((data) => {
        const types = (data.items || []).filter(Boolean);
        setAssetTypes(types);
        setSelectedTypes(types);
      })
      .catch(console.error);
  }
  function handleUploadFiles(files) {
    const filtered = Array.from(files).filter((file) => file.name.toLowerCase().endsWith(".zip"));
    setUploadFiles(filtered);
  }
  useEffect(() => {
    if (view !== "tasks") return;
    let active = true;
    const load = () => {
      if (!tasksLoadedRef.current) setTasksLoading(true);
      fetchTasks()
        .then((data) => {
          if (!active) return;
          const items = data.items || [];
          setTasks(items);
        })
        .catch(() => {})
        .finally(() => {
          if (active) {
            tasksLoadedRef.current = true;
            setTasksLoading(false);
          }
        });
    };
    load();
    const handle = setInterval(load, 3000);
    return () => {
      active = false;
      clearInterval(handle);
    };
  }, [view]);

  useEffect(() => {
    let active = true;
    const loadQueue = () => {
      fetchQueueStatus()
        .then((data) => {
          if (!active) return;
          setQueueStatus(data || null);
          setQueueStatusError("");
          setQueueStatusUpdatedAt(Date.now());
        })
        .catch((err) => {
          if (!active) return;
          setQueueStatusError(String(err?.message || "queue status unavailable"));
        });
    };
    loadQueue();
    const handle = setInterval(loadQueue, 3000);
    return () => {
      active = false;
      clearInterval(handle);
    };
  }, []);

  useEffect(() => {
    if (!copyStatus?.projectId) return;
    if (copyStatus.status === "done" || copyStatus.status === "error") return;
    const handle = setInterval(() => {
      fetchProjectCopyStatus(copyStatus.projectId)
        .then((data) => {
          setCopyStatus({ ...data, projectId: copyStatus.projectId });
        })
        .catch(() => {
          setCopyStatus((prev) => prev);
        });
    }, 1000);
    return () => clearInterval(handle);
  }, [copyStatus]);
  useEffect(() => {
    if (!copyStatus) return;
    const projectKey = String(copyStatus.projectId ?? "default");
    const copied = Number(copyStatus.copied ?? 0);
    const total = Number(copyStatus.total ?? 0);
    const percent = total > 0 ? Math.round((copied / total) * 100) : 0;
    const existingId = copyToastIdsRef.current.get(projectKey);
    const progressLabel = `Copying content ${copied}/${total} (${percent}%)`;
    if (copyStatus.status === "done") {
      const doneLabel = total > 0 ? `Content copied (${total} files)` : "Content copied";
      if (existingId) {
        toast.update(existingId, {
          render: doneLabel,
          type: "success",
          isLoading: false,
          autoClose: 2500,
        });
        copyToastIdsRef.current.delete(projectKey);
      } else {
        toast.success(doneLabel);
      }
      fetchProjects()
        .then((fresh) => setProjects(fresh))
        .catch(() => {});
      if (copyStatus.projectId) {
        fetchProjectStats(copyStatus.projectId)
          .then((stats) => setProjectStats((prev) => ({ ...prev, [copyStatus.projectId]: stats })))
          .catch(() => {});
      }
      setCopyStatus(null);
      return;
    }
    if (copyStatus.status === "error") {
      const errorLabel = copyStatus.error || "Copy failed";
      if (existingId) {
        toast.update(existingId, {
          render: errorLabel,
          type: "error",
          isLoading: false,
          autoClose: 5000,
        });
        copyToastIdsRef.current.delete(projectKey);
      } else {
        toast.error(errorLabel);
      }
      setCopyStatus(null);
      return;
    }
    if (existingId) {
      toast.update(existingId, { render: progressLabel, isLoading: true, autoClose: false });
    } else {
      const id = toast.info(progressLabel, {
        isLoading: true,
        autoClose: false,
        closeOnClick: false,
        draggable: false,
      });
      copyToastIdsRef.current.set(projectKey, id);
    }
  }, [copyStatus]);
  function toggleArtStyle(style) {
    setSelectedArtStyles((prev) => {
      if (prev.includes(style)) {
        return prev.filter((item) => item !== style);
      }
      return [...prev, style];
    });
    resetPaging(true, true);
  }
  const queueAgeMs = queueStatusUpdatedAt ? Date.now() - queueStatusUpdatedAt : 0;
  const queueIsStale = queueStatusUpdatedAt > 0 && queueAgeMs > 15000;
  const queueTopLabel = queueStatusError
    ? `Queue offline: ${queueStatusError}`
    : queueStatus
    ? `Queue ${queueStatus.worker_busy ? "busy" : "idle"}${
        queueStatus.worker_active_task_id ? ` #${queueStatus.worker_active_task_id}` : ""
      } | queued ${queueStatus.tasks?.queued || 0} | OpenAI pending ${
        queueStatus.openai_batches?.pending || 0
      } ready ${queueStatus.openai_batches?.ready || 0}${queueIsStale ? " | stale" : ""}`
    : "Queue loading...";
  const startupImport = queueStatus?.startup_import || null;
  const startupImportRunning = Boolean(startupImport?.running);
  const startupImportTotal = Number(startupImport?.total || 0);
  const startupImportDone = Number(startupImport?.done || 0);
  const startupImportPercent = startupImportTotal > 0 ? Math.min(100, Math.round((startupImportDone / startupImportTotal) * 100)) : 0;
  const startupImportText = startupImportRunning
    ? startupImportTotal > 0
      ? `startup import ${startupImportDone}/${startupImportTotal}`
      : "startup import scanning"
    : "";
  const queueTopLabelWithStartup = startupImportText ? `${queueTopLabel} | ${startupImportText}` : queueTopLabel;
  return (
    <div className="app-shell" style={{ "--sidebar-width": `${settings.sidebar_width || 280}px` }}>
      <nav className="navbar navbar-dark bg-dark navbar-expand">
        <div className="container-fluid">
          <div className="navbar-brand navbar-brand-logo" onClick={() => handleViewClick("assets")}>
            <img src={logo64} alt="Asset Admin" className="brand-logo" />
          </div>
          <div className="navbar-nav me-auto">
            {views.map((key) => (
              <a
                key={key}
                className={`nav-link btn btn-link ${view === key ? "active" : ""}`}
                href={`#${key}`}
                onClick={(e) => {
                  e.preventDefault();
                  handleViewClick(key);
                }}
              >
                {key}
              </a>
            ))}
            <div className={`nav-item nav-dropdown ${aboutOpen ? "open" : ""}`} ref={aboutRef}>
              <button
                className="nav-link btn btn-link nav-dropdown-toggle"
                type="button"
                aria-haspopup="true"
                aria-expanded={aboutOpen}
                onClick={() => setAboutOpen((prev) => !prev)}
              >
                About
              </button>
              <div className="nav-dropdown-menu">
                <a
                  className="nav-dropdown-item"
                  href="https://patreon.com/UbahnWorkerGames"
                  target="_blank"
                  rel="noreferrer"
                  onClick={() => setAboutOpen(false)}
                >
                  Patreon
                </a>
                <a
                  className="nav-dropdown-item"
                  href="https://github.com/ubahnworkergames"
                  target="_blank"
                  rel="noreferrer"
                  onClick={() => setAboutOpen(false)}
                >
                  GitHub
                </a>
              </div>
            </div>
          </div>
          <div className={`navbar-text small ${queueStatusError ? "text-warning" : queueIsStale ? "text-warning" : "text-light"}`}>
            {queueTopLabelWithStartup}
            {startupImportRunning && (
              <div className="startup-import-status">
                <div className="startup-import-label">
                  {`Startup import ${startupImportDone}/${startupImportTotal || "?"}${startupImport?.current_flow ? ` (${startupImport.current_flow})` : ""}`}
                </div>
                <div className="startup-import-bar">
                  <div className="startup-import-bar-fill" style={{ width: `${startupImportPercent}%` }} />
                </div>
              </div>
            )}
          </div>
        </div>
      </nav>
      {view === "assets" && (
        <section className="panel panel-assets">
            <div className="asset-layout">
              <aside className="asset-sidebar">
                <div className="sidebar-block">
                  <div className="sidebar-title">Saved search</div>
                  <select
                    className="form-select"
                    value={selectedViewId}
                    onChange={(e) => handleSelectViewChange(e.target.value)}
                  >
                    <option value="">Select saved view</option>
                    {savedViews.map((view) => (
                      <option key={view.id} value={view.id}>
                        {view.name}
                      </option>
                    ))}
                  </select>
                  <div className="detail-header-actions">
                    <button className="btn btn-outline-dark btn-sm" type="button" onClick={handleSaveView}>
                      Save
                    </button>
                    <button
                      className="btn btn-outline-dark btn-sm"
                      type="button"
                      disabled={!selectedViewId}
                      onClick={() => handleDeleteView(selectedViewId)}
                    >
                      Delete
                    </button>
                  </div>
                </div>
                <div className="sidebar-block">
                  <div className="sidebar-title">Search</div>
                  <input
                    className="form-control"
                    placeholder="Search name, description, tags..."
                    value={queryInput}
                    onChange={(e) => {
                      setQueryInput(e.target.value);
                      resetPaging(true, true);
                    }}
                  />
                  <label className="filter-item">
                    <input
                      type="checkbox"
                      checked={useSemanticSearch}
                      onChange={(e) => {
                        setUseSemanticSearch(e.target.checked);
                        resetPaging(true, true);
                      }}
                    />
                    Semantic search
                  </label>
                </div>
                <div className="sidebar-block">
                  <div className="sidebar-title">Overview</div>
                  <div className="asset-count">All assets: {totalAll}</div>
                  <div className="asset-count">Matches: {totalCount}</div>
                </div>
              <div className="sidebar-block">
                <div className="sidebar-title">Item size</div>
                <input
                  type="range"
                  className="form-range"
                  min="160"
                  max="280"
                  value={tileSize}
                  onChange={(e) => setTileSize(Number(e.target.value))}
                />
                <div className="asset-count">Current size: {tileSize}px</div>
              </div>
              <div className="sidebar-block">
                <div className="sidebar-title">Filters</div>
                <div className="filter-group">
                  <div className="filter-label">Type</div>
                  <label className="filter-item">
                    <input
                      type="checkbox"
                      checked={selectedTypes.length === assetTypes.length && assetTypes.length > 0}
                      onChange={(e) => toggleAllTypes(e.target.checked)}
                    />
                    Toggle all
                  </label>
                  <div className="filter-list">
                    {assetTypes.length ? (
                      assetTypes.map((assetType) => (
                        <label key={assetType} className="filter-item">
                          <input
                            type="checkbox"
                            checked={selectedTypes.includes(assetType)}
                            onChange={() => toggleType(assetType)}
                          />
                          {assetType}
                        </label>
                      ))
                    ) : (
                      <div className="asset-count">No types found</div>
                    )}
                  </div>
                </div>
                <div className="filter-group">
                  <div className="filter-label">Nanite</div>
                  <select
                    className="form-select"
                    value={naniteFilter}
                    onChange={(e) => {
                      setNaniteFilter(e.target.value);
                      resetPaging(true, true);
                    }}
                  >
                    <option value="all">All</option>
                    <option value="with">With</option>
                    <option value="without">Without</option>
                  </select>
                </div>
                <div className="filter-group">
                  <div className="filter-label">Collision</div>
                  <select
                    className="form-select"
                    value={collisionFilter}
                    onChange={(e) => {
                      setCollisionFilter(e.target.value);
                      resetPaging(true, true);
                    }}
                  >
                    <option value="all">All</option>
                    <option value="with">With</option>
                    <option value="without">Without</option>
                  </select>
                </div>
                <div className="filter-group">
                  <div className="filter-label">Art style</div>
                  <label className="filter-item">
                    <input
                      type="checkbox"
                      checked={selectedArtStyles.length === 3}
                      onChange={(e) => {
                        setSelectedArtStyles(
                          e.target.checked ? ["regular", "stylized", "low poly"] : []
                        );
                        resetPaging(true, true);
                      }}
                    />
                    Toggle all
                  </label>
                  <div className="filter-list">
                    {["regular", "stylized", "low poly"].map((style) => (
                      <label key={style} className="filter-item">
                        <input
                          type="checkbox"
                          checked={selectedArtStyles.includes(style)}
                          onChange={() => toggleArtStyle(style)}
                        />
                        {style}
                      </label>
                    ))}
                  </div>
                </div>
                <div className="filter-group">
                  <div className="filter-label">Era</div>
                  <label className="filter-item">
                    <input
                      type="checkbox"
                      checked={selectedEras.length === eraOptions.length && eraOptions.length > 0}
                      onChange={(e) => {
                        setSelectedEras(e.target.checked ? eraOptions : []);
                        resetPaging(true, true);
                      }}
                    />
                    Toggle all
                  </label>
                  <div className="filter-list">
                    {eraOptions.length ? (
                      eraOptions.map((era) => (
                        <label key={era} className="filter-item">
                          <input
                            type="checkbox"
                            checked={selectedEras.includes(era)}
                            onChange={() => {
                              setSelectedEras((prev) =>
                                prev.includes(era)
                                  ? prev.filter((value) => value !== era)
                                  : [...prev, era]
                              );
                              resetPaging(true, true);
                            }}
                          />
                          {era}
                        </label>
                      ))
                    ) : (
                      <div className="asset-count">No eras found</div>
                    )}
                  </div>
                </div>
                <div className="filter-group">
                  <div className="filter-label">Projects</div>
                  <input
                    className="form-control form-control-sm"
                    placeholder="Filter projects..."
                    value={projectFilterSearch}
                    onChange={(e) => setProjectFilterSearch(e.target.value)}
                  />
                  <label className="filter-item">
                    <input
                      type="checkbox"
                      checked={selectedProjects.length === projects.length && projects.length > 0}
                      onChange={(e) => toggleAllProjects(e.target.checked)}
                    />
                    Toggle all
                  </label>
                  <div className="filter-list">
                    {projects
                      .filter((project) => {
                        const stats = projectStats[project.id] || {};
                        const total = Number(stats.total_all ?? stats.total ?? 0);
                        if (showEmptyProjects && total > 0) return false;
                        if (selectedEras.length && eraOptions.length) {
                          const era = String(project.project_era || "").trim().toLowerCase();
                          if (era && !selectedEras.includes(era)) return false;
                        }
                        const query = projectFilterSearch.trim().toLowerCase();
                        if (!query) return true;
                        const nameMatch = (project.name || "").toLowerCase().includes(query);
                        const tagMatch = (project.tags || []).some((tag) =>
                          String(tag || "").toLowerCase().includes(query)
                        );
                        return nameMatch || tagMatch;
                      })
                      .map((project) => (
                        <div key={project.id} className="filter-item">
                          <label className="project-filter-label">
                            <button
                              className="icon-btn icon-folder"
                              type="button"
                              onClick={() => handleOpenProject(project)}
                              title="Open folder"
                              aria-label="Open folder"
                            >
                              <FontAwesomeIcon icon={faFolderOpen} />
                            </button>
                            <input
                              type="checkbox"
                              checked={selectedProjects.includes(String(project.id))}
                              onChange={() => toggleProject(project.id)}
                            />
                            <span className="project-filter-name" style={{ maxWidth: `${projectNameWrap}ch` }}>
                              {project.name}
                            </span>
                            <span className="filter-meta">
                              {projectStats[project.id]?.matched ?? 0}/{projectStats[project.id]?.total ?? 0}
                            </span>
                          </label>
                        </div>
                      ))}
                  </div>
                </div>
                  <div className="filter-group">
                  <button
                    className="btn btn-outline-dark btn-sm"
                    type="button"
                    onClick={() => {
                      refreshTypes();
                      setRefreshKey((key) => key + 1);
                    }}
                    >
                      Refresh list
                  </button>
                  </div>
                </div>
              </aside>
              <div className={`asset-main${selectedAssetId ? " detail-open" : ""}`}>
                {selectedAssetId ? (
                  <div className="asset-detail-panel">
                    {!selectedAsset ? (
                      <div className="asset-detail-loading">Loading asset...</div>
                    ) : (
                      <>
                        <div className="detail-header">
                          <div>
                            <div className="detail-title">
                              {selectedAsset.meta?.package
                                ? selectedAsset.meta.package.split("/").pop()
                                : selectedAsset.name}
                            </div>
                            <div className="detail-subtitle detail-subtitle-actions">
                              <span>{selectedProject?.name || "Unassigned"}</span>
                              {selectedProject && (
                                <button
                                  className="btn btn-outline-dark btn-xs"
                                  type="button"
                                  onClick={() => handleViewProjectAssets(selectedProject.id)}
                                >
                                  View project assets
                                </button>
                              )}
                            </div>
                          </div>
                          <div className="detail-header-actions">
                            <button
                              className="btn btn-outline-dark btn-sm"
                              onClick={() => {
                                closeDetail();
                              }}
                            >
                              Back to list
                            </button>
                            <button
                              className="btn btn-outline-dark btn-sm"
                              onClick={handlePrevDetail}
                              disabled={selectedIndex <= 0}
                            >
                              Prev
                            </button>
                              <button
                                className="btn btn-outline-dark btn-sm"
                                onClick={handleNextDetail}
                                disabled={selectedIndex < 0 || (selectedIndex >= visibleAssets.length - 1 && visibleAssets.length >= totalCount)}
                              >
                                Next
                              </button>
                            <span className="detail-counter">
                              {selectedIndex >= 0 ? `${selectedIndex + 1} / ${totalCount || visibleAssets.length}` : ""}
                            </span>
                            <span className="detail-id">ID {selectedAsset.id}</span>
                            <button
                              className="btn btn-outline-dark btn-sm"
                              onClick={() => handleExportAssetZip(selectedAsset.id)}
                            >
                              Export
                            </button>
                            <button
                              className="btn btn-outline-danger btn-sm"
                              onClick={() => handleDeleteAsset(selectedAsset.id)}
                            >
                                Delete asset
                            </button>
                          </div>
                        </div>
                        <div className="detail-body">
                            <div className="detail-left">
                              <div className="detail-preview">
                                {selectedAsset.anim_detail ? (
                                  <img
                                    src={resolveApiUrl(selectedAsset.anim_detail)}
                                    alt={selectedAsset.name}
                                    onClick={() =>
                                      setDetailImagePreview(
                                        resolveApiUrl(
                                          selectedAsset.anim_detail || selectedAsset.full_image || selectedAsset.detail_image
                                        )
                                      )
                                    }
                                  />
                                ) : selectedAsset.detail_image ? (
                                  <img
                                    src={resolveApiUrl(selectedAsset.detail_image)}
                                    alt={selectedAsset.name}
                                    onClick={() =>
                                      setDetailImagePreview(
                                        resolveApiUrl(selectedAsset.full_image || selectedAsset.detail_image)
                                      )
                                    }
                                  />
                                ) : (
                                  <div className="asset-placeholder">No preview</div>
                                )}
                              </div>
                            </div>
                          <div className="detail-info">
                            {(selectedProject?.art_style || selectedAsset?.project_art_style) && (
                              <div className="detail-row">
                                <span>Art style</span>
                                <span>{selectedProject?.art_style || selectedAsset?.project_art_style}</span>
                              </div>
                            )}
                            {selectedProject?.project_era && (
                              <div className="detail-row">
                                <span>Era</span>
                                <span>{selectedProject.project_era}</span>
                              </div>
                            )}
                            {(selectedAsset.meta?.class || selectedAsset.type) && (
                              <div className="detail-row">
                                <span>Class</span>
                                <span>{selectedAsset.meta?.class || selectedAsset.type}</span>
                              </div>
                            )}
                            {(selectedAsset.meta?.disk_bytes_total || selectedAsset.size_bytes) && (
                              <div className="detail-row">
                                <span>Size</span>
                                <span>
                                  {formatBytes(selectedAsset.meta?.disk_bytes_total || selectedAsset.size_bytes)}
                                  {formatApproxSize(selectedAsset.meta) ? ` (~${formatApproxSize(selectedAsset.meta)})` : ""}
                                </span>
                              </div>
                            )}
                            {selectedAsset.meta?.mesh?.vertices !== undefined && (
                              <div className="detail-row">
                                <span>Vertices</span>
                                <span>{selectedAsset.meta?.mesh?.vertices}</span>
                              </div>
                            )}
                            {selectedAsset.meta?.mesh?.triangles !== undefined && (
                              <div className="detail-row">
                                <span>Triangles</span>
                                <span>{selectedAsset.meta?.mesh?.triangles}</span>
                              </div>
                            )}
                            {selectedAsset.meta?.mesh?.nanite_enabled !== undefined && (
                              <div className="detail-row">
                                <span>Nanite</span>
                                <span>{selectedAsset.meta?.mesh?.nanite_enabled ? "Yes" : "No"}</span>
                              </div>
                            )}
                            {selectedAsset.meta?.mesh?.collision_complexity && (
                              <div className="detail-row">
                                <span>Collision</span>
                                <span>{selectedAsset.meta?.mesh?.collision_complexity}</span>
                              </div>
                            )}
                            {selectedAsset.project_era && (
                              <div className="detail-row">
                                <span>Era</span>
                                <span>{selectedAsset.project_era}</span>
                              </div>
                            )}
                            <div className="detail-row">
                              <span>Tags</span>
                              <span className="tag-edit-wrap">
                                {editableTags.map((tag) => (
                                  <span key={`edit-${tag}`} className="tag-pill tag-pill-edit">
                                    <button
                                      className="tag-pill-button"
                                      type="button"
                                      onClick={() => handleTagClick(tag)}
                                    >
                                      {tag}
                                    </button>
                                    <button type="button" onClick={() => handleRemoveTag(tag)}>x</button>
                                  </span>
                                ))}
                                <input
                                  className="form-control form-control-sm tag-edit-input"
                                  placeholder="Add tag"
                                  value={tagInput}
                                  onChange={(e) => setTagInput(e.target.value)}
                                  onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); handleAddTag(); } }}
                                />
                                <button className="btn btn-outline-dark btn-sm" type="button" onClick={handleAddTag}>Add</button>
                              </span>
                            </div>
                            {contentDir ? (
                              <div className="detail-row">
                                <span>Content dir</span>
                                <span>{contentDir}</span>
                              </div>
                            ) : null}
                            <div className="detail-row detail-action-row">
                              <span>Actions</span>
                              <span>
                                <button
                                  className="btn btn-outline-dark btn-sm"
                                  onClick={() => handleMigrateAsset(selectedAsset.id)}
                                >
                                  Migrate
                                </button>
                                <button
                                  className="btn btn-outline-dark btn-sm"
                                  onClick={() => handleSelectAsset(selectedAsset.id)}
                                >
                                  Select
                                </button>
                                <button
                                  className="btn btn-outline-dark btn-sm"
                                  onClick={() => handleGenerateTags(selectedAsset.id)}
                                >
                                  Generate tags
                                </button>
                              </span>
                            </div>
                            <div className="detail-files detail-files-secondary">
                              <div className="detail-files-title">
                                Files ({(selectedAsset.meta?.files_on_disk || []).length})
                              </div>
                              {selectedAsset.path_warning && (
                                <div className="detail-warning">
                                  Warning: Files span multiple root folders ({(selectedAsset.path_roots || []).join(", ")}).
                                  Export will include extra folders.
                                </div>
                              )}
                              <pre className="detail-files-box">
                                {(selectedAsset.meta?.files_on_disk || []).length
                                  ? selectedAsset.meta.files_on_disk.join("\n")
                                  : "No files"}
                              </pre>
                            </div>
                            {migrateStatus?.assetId === selectedAsset.id && (
                              <div className="copy-bar">
                                <div
                                  className="copy-bar-fill"
                                  style={{
                                    width:
                                      migrateStatus.total > 0
                                        ? `${Math.round((migrateStatus.copied / migrateStatus.total) * 100)}%`
                                        : "0%",
                                  }}
                                />
                              </div>
                            )}
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                ) : (
                  <>
                      <div
                        className="asset-grid"
                        style={{
                          gridTemplateColumns: `repeat(auto-fill, minmax(${tileSize}px, 1fr))`,
                          display: selectedAssetId ? "none" : "grid",
                        }}
                      >
                        {visibleAssets.map((asset) => {
                          const project = projectMap.get(asset.project_id);
                          return (
                            <AssetCard
                              key={asset.id}
                              asset={asset}
                              project={project}
                              tileSize={tileSize}
                              isSelected={selectedAssetIds.includes(asset.id)}
                              onSelect={(e) => handleAssetClick(asset, e)}
                              onContextMenu={(e) => handleAssetContextMenu(asset, e)}
                            />
                          );
                        })}
                      </div>
                    <div ref={loadMoreRef} style={{ height: 1 }} />
                    {isLoadingMore && !selectedAssetId && <div className="load-more">Loading more...</div>}
                  </>
                )}
              </div>
          </div>
        </section>
      )}
      {view === "upload" && (
        <section className="panel">
          <div className="upload-panel">
            <div className="section-title">Upload</div>
            <form className="upload-form" onSubmit={handleUpload}>
              <div className="upload-row">
                <label className="form-label">Project (optional)</label>
                <select
                  className="form-select"
                  value={uploadProject}
                  onChange={(e) => setUploadProject(e.target.value)}
                >
                  <option value="">Unassigned</option>
                  {projects.map((project) => (
                    <option key={project.id} value={project.id}>
                      {project.name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="upload-row">
                <label className="form-label">ZIP files</label>
                <input
                  className="form-control"
                  type="file"
                  accept=".zip"
                  multiple
                  onChange={(e) => handleUploadFiles(e.target.files || [])}
                />
              </div>
              {uploadFiles.length > 0 && (
                <div className="upload-row upload-meta">
                  <span>{uploadFiles.length} file(s) selected</span>
                  {uploadProgress && (
                    <span>{uploadProgress.current}/{uploadProgress.total} ({uploadProgress.percent}%)</span>
                  )}
                </div>
              )}
              <div className="upload-row">
                <button className="btn btn-dark" type="submit" disabled={!uploadFiles.length}>
                  Upload
                </button>
              </div>
            </form>
          </div>
        </section>
      )}
      {view === "projects" && (
        <section className="panel">
          <div className="project-layout">
              <div className="project-list">
                <div className="project-list-header">
                <div className="project-list-title">
                  {showImportHelper && (
                    <div className="btn-group">
                      <a
                        className="btn btn-outline-success btn-sm"
                        href="https://patreon.com/UbahnWorkerGames"
                        target="_blank"
                        rel="noreferrer"
                      >
                        Get Importhelper
                      </a>
                      <button
                        className="btn btn-outline-danger btn-sm importhelper-dismiss"
                        type="button"
                        title="Hide permanently"
                        aria-label="Hide permanently"
                        onClick={() => {
                          if (typeof window !== "undefined") {
                            const ok = window.confirm("Hide this button permanently?");
                            if (!ok) return;
                          }
                          setShowImportHelper(false);
                        }}
                      >
                        ×
                      </button>
                    </div>
                  )}
                  <div className="project-stats-grid">
                    <div className="project-stats-box">
                    <div className="project-stats-actions">
                      <button
                        className="btn btn-outline-warning btn-sm"
                        type="button"
                        onClick={handleRestartServer}
                        disabled={restartingServer}
                        title="Restart backend once"
                      >
                        {restartingServer ? "Restarting..." : "Restart server"}
                      </button>
                      <button
                        className="btn btn-outline-dark btn-sm"
                        type="button"
                        onClick={handleRefreshProjectSizes}
                        disabled={refreshingProjectSizes}
                        title="Recalculate source sizes"
                      >
                        {refreshingProjectSizes ? "Refreshing..." : "Refresh sizes"}
                      </button>
                      <button
                        className="btn btn-outline-dark btn-sm"
                        type="button"
                        onClick={handleRegenerateEmbeddingsAll}
                        title="Queue a full semantic embeddings rebuild for all assets."
                      >
                        Rebuild semantic (all)
                      </button>
                      <button
                        className="btn btn-outline-dark btn-sm"
                        type="button"
                        onClick={() => {
                          setProjectFullCopy(Boolean(settings.default_full_project_copy));
                          setShowCreateProject(true);
                        }}
                        title="Create a new project entry."
                      >
                        New project
                      </button>
                      <button
                        className="btn btn-outline-dark btn-sm"
                        type="button"
                        onClick={handleExportProjects}
                        title="Export project list as CSV."
                      >
                        Export CSV
                      </button>
                      <label className="btn btn-outline-dark btn-sm project-import-btn" title="Select a CSV file to import projects from.">
                        Import CSV
                        <input
                          type="file"
                          accept=".csv"
                          onChange={(e) => setProjectImportFile(e.target.files?.[0] || null)}
                        />
                      </label>
                      <button
                        className="btn btn-outline-dark btn-sm"
                        type="button"
                        onClick={handleImportProjects}
                        disabled={!projectImportFile}
                        title="Run CSV project import."
                      >
                        Run Import
                      </button>
                    </div>
                    {projectStatsAggregate.typeList.length > 0 && (
                      <div className="project-stats-tags">
                        {projectStatsAggregate.typeList.slice(0, 12).map(([type, count]) => (
                          <TagPill key={`project-stats-${type}`} label={`${type} ${count}`} />
                        ))}
                      </div>
                    )}
                  </div>
                  <div className="project-stats-box project-tag-stats-box">
                    <div className="project-stats-row">
                      <div className="project-stats-item">
                        <span className="project-stats-label">Projects</span>
                        <span className="project-stats-value">{projects.length}</span>
                      </div>
                      <div className="project-stats-item">
                        <span className="project-stats-label">Source gesamt</span>
                        <span className="project-stats-value">
                          {formatBytes(projectStatsAggregate.totalSourceBytes)}
                        </span>
                      </div>
                      <div className="project-stats-item">
                        <span className="project-stats-label">Assets mit Tags</span>
                        <span className="project-stats-value">{tagStatsAggregate.taggedAssets}</span>
                      </div>
                      <div className="project-stats-item">
                        <span className="project-stats-label">Assets gesamt</span>
                        <span className="project-stats-value">{tagStatsAggregate.totalAssets}</span>
                      </div>
                      <div className="project-stats-item">
                        <span className="project-stats-label">Tags gesamt</span>
                        <span className="project-stats-value">{tagStatsAggregate.tagAssignmentsTotal}</span>
                      </div>
                      <div className="project-stats-item">
                        <span className="project-stats-label">Tags einzigartig</span>
                        <span className="project-stats-value">{tagStatsAggregate.uniqueTagsTotal}</span>
                      </div>
                      <div className="project-stats-item">
                        <span className="project-stats-label">Avg tags/asset</span>
                        <span className="project-stats-value">{tagStatsAggregate.avgTagsPerTaggedAsset.toFixed(2)}</span>
                      </div>
                      <div className="project-stats-item">
                        <span className="project-stats-label">Assets ohne Tags</span>
                        <span className="project-stats-value">{tagStatsAggregate.assetsWithoutTags}</span>
                      </div>
                    </div>
                    <div className="project-stats-tags project-action-tags">
                        <div className="project-action-row-group project-action-row-group-local">
                        <div className="project-action-row-label">
                          <FontAwesomeIcon icon={faHardDrive} /> Local
                        </div>
                        <div className="project-action-row project-action-row-local">
                          <button
                            className="btn btn-outline-dark btn-sm"
                            type="button"
                            onClick={handleTagMissingAllProjects}
                            title="Tags only; embeddings are not generated."
                          >
                            Tag missing (all)
                          </button>
                          <button
                            className="btn btn-outline-dark btn-sm"
                            type="button"
                            onClick={handleNameTagsAllSimple}
                            title="Generate tags from asset names for all assets (local processing)."
                          >
                            Name to tags (all)
                          </button>
                          <button
                            className="btn btn-outline-dark btn-sm"
                            type="button"
                            onClick={handleNameTagsAllSimpleMissing}
                            title="Generate tags from asset names only for assets below the missing-tag threshold (local processing)."
                          >
                            Name to tags missing (all)
                          </button>
                        </div>
                      </div>
                        <div className="project-action-row-group">
                        <div className="project-action-row-label">
                          <FontAwesomeIcon icon={faLightbulb} /> {llmActionGroupLabel}
                        </div>
                        <div className="project-action-row project-action-row-llm">
                          <button
                            className="btn btn-outline-dark btn-sm"
                            type="button"
                            onClick={handleTranslateNameTagsAll}
                            disabled={!llmReady}
                            title={llmReadyTitle}
                          >
                          Asset title (all)
                          </button>
                          <button
                            className="btn btn-outline-dark btn-sm"
                            type="button"
                            onClick={handleTranslateNameTagsAllMissing}
                            disabled={!llmReady}
                            title={llmReadyTitle}
                          >
                          Asset title missing (all)
                          </button>
                          <button
                            className="btn btn-outline-dark btn-sm"
                            type="button"
                            onClick={handleTranslateAllTags}
                            disabled={!llmReady}
                            title={llmReadyTitle}
                          >
                          Translate tags (all)
                          </button>
                          <button
                            className="btn btn-outline-dark btn-sm"
                            type="button"
                            onClick={handleTranslateAllTagsMissing}
                            disabled={!llmReady}
                            title={llmReadyTitle}
                          >
                          Translate tags missing (all)
                          </button>
                        </div>
                      </div>
                    </div>
                    <div className="project-restart-note">
                      After a server restart, actions may be slower at first while archived batch outputs are imported.
                    </div>
                  </div>
                </div>
                </div>
                <div className="project-list-actions">
                  <div className="project-filters-row">
                    <select
                      className="form-select form-select-sm project-filter-select"
                      value={projectAiFilter}
                      onChange={(e) => setProjectAiFilter(e.target.value)}
                    >
                      <option value="all">AI: All</option>
                      <option value="only">AI: Only</option>
                      <option value="exclude">AI: Exclude</option>
                    </select>
                    <select
                      className="form-select form-select-sm project-filter-select"
                      value={projectArtStyleFilter}
                      onChange={(e) => setProjectArtStyleFilter(e.target.value)}
                    >
                      <option value="__all__">Art style: All</option>
                      {hasEmptyProjectArtStyle && <option value="__none__">Art style: None</option>}
                      {projectArtStyles.map((style) => (
                        <option key={style} value={style}>
                          {style}
                        </option>
                      ))}
                    </select>
                    <select
                      className="form-select form-select-sm project-filter-select"
                      value={projectEraFilter}
                      onChange={(e) => setProjectEraFilter(e.target.value)}
                    >
                      <option value="__all__">Era: All</option>
                      {hasEmptyProjectEra && <option value="__none__">Era: None</option>}
                      {projectEraOptions.map((era) => (
                        <option key={era} value={era}>
                          {era}
                        </option>
                      ))}
                    </select>
                    <label className="form-check form-check-inline project-empty-toggle">
                      <input
                        className="form-check-input"
                        type="checkbox"
                        checked={showEmptyProjects}
                        onChange={(e) => setShowEmptyProjects(e.target.checked)}
                      />
                      <span className="form-check-label">Only empty</span>
                    </label>
                    <select
                      className="form-select form-select-sm project-filter-select"
                      value={projectSortKey}
                      onChange={(e) => setProjectSortKey(e.target.value)}
                    >
                      <option value="name">Sort: Name</option>
                      <option value="id">Sort: ID</option>
                      <option value="source_folder">Sort: Source pack folder</option>
                    </select>
                    <button
                      className="btn btn-outline-dark btn-sm"
                      type="button"
                      onClick={() =>
                        setProjectSortDir((prev) => (prev === "asc" ? "desc" : "asc"))
                      }
                    >
                      {projectSortDir === "asc" ? "Asc" : "Desc"}
                    </button>
                    <input
                      className="form-control form-control-sm project-search"
                      placeholder="Search name, tag, category, description"
                      value={projectSearch}
                      onChange={(e) => setProjectSearch(e.target.value)}
                    />
                  </div>
                </div>
              </div>
              <div className="project-list-wrap row g-3">
                {sortedProjects.map((project) => {
                  const folderLabel = project.source_path
                    ? project.source_path.split(/[/\\]+/).filter(Boolean).pop()
                    : project.folder_path
                      ? project.folder_path.split(/[/\\]+/).filter(Boolean).pop()
                      : "-";
                  return (
                    <div key={project.id} className="col-12 col-md-6 col-lg-3">
                      <div
                        className="card project-item h-100"
                        onContextMenu={(e) => {
                          e.preventDefault();
                        }}
                      >
                        {editingProjectId === project.id ? (
                          <div className="card-body project-edit">
                            <label className="form-label">Name</label>
                            <input
                              className="form-control"
                              value={editName}
                              onChange={(e) => setEditName(e.target.value)}
                            />
                            <label className="form-label">Link</label>
                            <input
                              className="form-control"
                              value={editLink}
                              onChange={(e) => setEditLink(e.target.value)}
                            />
                            <label className="form-label">Screenshot URL (optional)</label>
                            <input
                              className="form-control"
                              placeholder="Screenshot URL"
                              value={editScreenshotUrl}
                              onChange={(e) => setEditScreenshotUrl(e.target.value)}
                            />
                            <div className="drop-zone">
                              <input
                                type="file"
                                accept="image/*"
                                onChange={(e) => setEditScreenshotFile(e.target.files?.[0] || null)}
                              />
                              <div>
                                {editScreenshotFile
                                  ? editScreenshotFile.name
                                  : "Project screenshot via drag & drop or file"}
                              </div>
                            </div>
                            <label className="form-label">Tags</label>
                            <input
                              className="form-control"
                              value={editTags}
                              onChange={(e) => setEditTags(e.target.value)}
                            />
                            <label className="form-label">Source content path</label>
                            <input
                              className="form-control"
                              placeholder="Source content path"
                              value={editSourcePath}
                              onChange={(e) => setEditSourcePath(e.target.value)}
                            />
                            <label className="form-label">Source pack folder (optional)</label>
                            <input
                              className="form-control"
                              placeholder="Source pack folder"
                              value={editSourceFolder}
                              onChange={(e) => setEditSourceFolder(e.target.value)}
                            />
                            <label className="form-check project-check">
                              <input
                                type="checkbox"
                                checked={editFullCopy}
                                onChange={(e) => setEditFullCopy(e.target.checked)}
                              />
                              <span>Copy full project (default: component only)</span>
                            </label>
                            <label className="form-check project-check">
                              <input
                                type="checkbox"
                                checked={editIsAi}
                                onChange={(e) => setEditIsAi(e.target.checked)}
                              />
                              <span>AI-generated</span>
                            </label>
                            <label className="form-check project-check">
                              <input
                                type="checkbox"
                                checked={(editArtStyle || "regular") === "stylized"}
                                onChange={(e) => setEditArtStyle(e.target.checked ? "stylized" : "regular")}
                              />
                              <span>Stylized</span>
                            </label>
                            <div className="project-actions">
                              <button
                                className="btn btn-primary btn-sm"
                                type="button"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleUpdateProject(project.id);
                                }}
                              >
                                Save
                              </button>
                              <button
                                className="btn btn-outline-dark btn-sm"
                                type="button"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setEditingProjectId(null);
                                }}
                              >
                                Cancel
                              </button>
                            </div>
                          </div>
                        ) : (
                          <>
                            <div className="project-thumb">
                              {getProjectCoverUrl(project) ? (
                                <img
                                  src={getProjectCoverUrl(project)}
                                  alt={project.name}
                                  onClick={() => setProjectPreview(getProjectCoverUrl(project))}
                                />
                              ) : (
                                <div className="thumb-placeholder">No image</div>
                              )}
                            </div>
                            <div className="card-body project-body">
                              <h3 className="project-title">{project.name || "-"}</h3>
                              <div className="project-meta">
                                <span className="project-meta-label">Link</span>
                                {project.link ? (
                                  <button
                                    className="project-link-fab"
                                    type="button"
                                    onClick={() => window.open(project.link, "_blank", "noopener")}
                                    aria-label="Open link"
                                    title="Open link"
                                  >
                                    🔗
                                  </button>
                                ) : (
                                  <span className="project-meta-value">-</span>
                                )}
                              </div>
                              <div className="project-meta">
                                <span className="project-meta-label">Art style</span>
                                <span className="project-meta-value">{project.art_style || "-"}</span>
                              </div>
                              <div className="project-meta">
                                <span className="project-meta-label">Era</span>
                                <span className="project-meta-value">{project.project_era || "-"}</span>
                              </div>
                              <div className="project-meta">
                                <span className="project-meta-label">Project size</span>
                                <span className="project-meta-value">
                                  {formatProjectBytes(project.folder_size_bytes ?? project.size_bytes)}
                                </span>
                              </div>
                              <div className="project-meta">
                                <span className="project-meta-label">Source size</span>
                                <span className="project-meta-value">
                                  {project.source_size_bytes !== undefined
                                    ? formatProjectBytes(project.source_size_bytes)
                                    : "-"}
                                </span>
                              </div>
                              <div className="project-meta">
                                <span className="project-meta-label">Path</span>
                                <span className="project-meta-value">{folderLabel}</span>
                              </div>
                              <div className="project-meta">
                                <span className="project-meta-label">Tagged assets</span>
                                <span className="project-meta-value">
                                  {(projectStats[project.id]?.tagged ?? 0)}/{(projectStats[project.id]?.total ?? 0)}
                                </span>
                              </div>
                              <div className="project-meta">
                                <span className="project-meta-label">Storage</span>
                                <span className="project-meta-value">
                                  {project.reimported_once ? (
                                    <span title="Local: sync has been executed at least once." aria-label="Local">
                                      <FontAwesomeIcon icon={faHardDrive} /> Local
                                    </span>
                                  ) : (
                                    <span title="External: uses source directory until the first sync." aria-label="External">
                                      <FontAwesomeIcon icon={faFolderOpen} /> External
                                    </span>
                                  )}
                                </span>
                              </div>
                              {projectStats[project.id]?.types &&
                                Object.keys(projectStats[project.id].types || {}).length > 0 && (
                                  <div className="project-tags">
                                    {Object.entries(projectStats[project.id].types).map(([type, count]) => (
                                      <TagPill key={`${project.id}-${type}`} label={`${type} ${count}`} />
                                    ))}
                                  </div>
                                )}
                              {(project.tags || []).length > 0 && (
                                <div className="project-tags">
                                  {(project.tags || []).map((tag) => (
                                    <TagPill key={tag} label={tag} onClick={() => handleTagClick(tag)} />
                                  ))}
                                </div>
                              )}
                                <div className="project-actions">
                                <div className="project-action-row">
                                  <button
                                    className="btn btn-dark btn-sm project-view-assets"
                                    type="button"
                                    onClick={() => handleViewProjectAssets(project.id)}
                                    title="Open the asset list for this project."
                                  >
                                    <FontAwesomeIcon icon={faEye} /> View Assets
                                  </button>
                                  <button className="btn btn-outline-dark btn-sm" onClick={(e) => {
                                    e.stopPropagation();
                                    startEditProject(project);
                                  }} title="Edit project name, source path, and metadata.">
                                    Edit
                                  </button>
                                  <button className="btn btn-outline-dark btn-sm" onClick={() => handleReimportProject(project)} title="Sync files from source and reimport in Asset Tool.">
                                    <FontAwesomeIcon icon={faCopy} /> Sync
                                  </button>
                                  <button
                                    className="btn btn-outline-dark btn-sm"
                                    onClick={() => handleOpenProject(project)}
                                    title="Open the local project folder."
                                  >
                                    <FontAwesomeIcon icon={faFolderOpen} /> Open Project Folder
                                  </button>
                                  <button
                                    className="btn btn-outline-dark btn-sm"
                                    onClick={() => handleOpenProjectSource(project)}
                                    title="Open the source project folder."
                                  >
                                    <FontAwesomeIcon icon={faFolderOpen} /> Open Source Folder
                                  </button>
                                  <button
                                    className="btn btn-outline-dark btn-sm"
                                    onClick={() => handleGenerateSetcard(project.id)}
                                    title="Generate or refresh the setcard preview image for this project."
                                  >
                                    <FontAwesomeIcon icon={faImages} /> Setcard
                                  </button>
                                  <button className="btn btn-outline-dark btn-sm" onClick={() => handleRunExportCmd(project)} title="Launch UnrealEditor-Cmd to re-export project files, then reimport/sync in Asset Tool.">
                                    Re-export via UE Cmd
                                  </button>
                                </div>
                                <div className="project-action-row-group project-action-row-group-local">
                                  <div className="project-action-row-label">
                                    <FontAwesomeIcon icon={faHardDrive} /> Local
                                  </div>
                                  <div className="project-action-row">
                                    <button className="btn btn-outline-dark btn-sm" onClick={() => handleTagMissing(project.id)} title="Tag assets in this project that are below the minimum tag threshold.">
                                      Tag missing
                                    </button>
                                    <button className="btn btn-outline-dark btn-sm" onClick={() => handleRetagAll(project.id)} title="Regenerate and replace tags for all assets in this project (local processing).">
                                      Tag all
                                    </button>
                                    <button
                                      className="btn btn-outline-dark btn-sm"
                                      onClick={() => handleRegenerateProjectEmbeddings(project.id)}
                                      title="Rebuild semantic embeddings for assets in this project."
                                    >
                                      Rebuild semantic
                                    </button>
                                    <button
                                      className="btn btn-outline-dark btn-sm"
                                      type="button"
                                      onClick={() => handleNameTagsProjectSimple(project.id)}
                                      title="Generate tags from asset names for this project (local processing)."
                                    >
                                      Name to tags
                                    </button>
                                    <button
                                      className="btn btn-outline-dark btn-sm"
                                      type="button"
                                      onClick={() => handleNameTagsProjectSimpleMissing(project.id)}
                                      title="Generate tags from asset names only for missing-tag assets in this project (local processing)."
                                    >
                                      Name to tags missing
                                    </button>
                                  </div>
                                </div>
                                <div className="project-action-row-group">
                                  <div className="project-action-row-label">
                                    <FontAwesomeIcon icon={faLightbulb} /> {llmActionGroupLabel}
                                  </div>
                                  <div className="project-action-row">
                                    <button
                                      className="btn btn-outline-dark btn-sm"
                                      type="button"
                                      onClick={() => handleTranslateNameTagsProject(project.id)}
                                      disabled={!llmReady}
                                      title={llmReadyTitle}
                                    >
                                      Asset title
                                    </button>
                                    <button
                                      className="btn btn-outline-dark btn-sm"
                                      type="button"
                                      onClick={() => handleTranslateNameTagsProjectMissing(project.id)}
                                      disabled={!llmReady}
                                      title={llmReadyTitle}
                                    >
                                      Asset title missing
                                    </button>
                                    <button
                                      className="btn btn-outline-dark btn-sm"
                                      type="button"
                                      onClick={() => handleTranslateTagsProject(project.id)}
                                      disabled={!llmReady}
                                      title={llmReadyTitle}
                                    >
                                      Translate tags
                                    </button>
                                    <button
                                      className="btn btn-outline-dark btn-sm"
                                      type="button"
                                      onClick={() => handleTranslateTagsProjectMissing(project.id)}
                                      disabled={!llmReady}
                                      title={llmReadyTitle}
                                    >
                                      Translate tags missing
                                    </button>
                                  </div>
                                </div>
                                <div className="project-action-row project-action-row-danger">
                                  <button
                                    className="btn btn-outline-danger btn-sm"
                                    onClick={() => handleDeleteProjectAssets(project.id)}
                                  >
                                    Delete assets
                                  </button>
                                  <button className="btn btn-outline-danger btn-sm" onClick={() => handleDeleteProject(project.id)}>
                                    Delete project
                                  </button>
                                </div>
                              </div>
                              {projectTagStatus[project.id] && projectTagStatus[project.id].status !== "idle" && (
                                <div className="project-meta">
                                  Tags: {projectTagStatus[project.id].done} / {projectTagStatus[project.id].total} (
                                  {projectTagStatus[project.id].status})
                                </div>
                              )}
                            </div>
                          </>
                        )}
                      </div>
                    </div>
                  );
                })}
                {filteredProjects.length === 0 && (
                  <div className="project-empty">No projects match that search.</div>
                )}
              </div>
            </div>
            <div className="settings-note">
              Import creates folders under {`open/data/projects`} if missing.
            </div>
            {showCreateProject && (
              <div className="modal-overlay" onClick={() => setShowCreateProject(false)}>
                <div className="project-modal" onClick={(e) => e.stopPropagation()}>
                  <form className="project-form" onSubmit={handleCreateProject}>
                    <h3>Create project</h3>
                    <input
                      className="form-control"
                      placeholder="Name"
                      value={projectName}
                      onChange={(e) => setProjectName(e.target.value)}
                      required
                    />
                    <input
                      className="form-control"
                      placeholder="Link"
                      value={projectLink}
                      onChange={(e) => setProjectLink(e.target.value)}
                    />
                    <input
                      className="form-control"
                      placeholder="Tags (comma separated)"
                      value={projectTags}
                      onChange={(e) => setProjectTags(e.target.value)}
                    />
                    <label className="form-check project-check">
                      <input
                        type="checkbox"
                        checked={(projectArtStyle || "regular") === "stylized"}
                        onChange={(e) => setProjectArtStyle(e.target.checked ? "stylized" : "regular")}
                      />
                      <span>Stylized</span>
                    </label>
                    <input
                      className="form-control"
                      placeholder="Source content path"
                      value={projectSourcePath}
                      onChange={(e) => setProjectSourcePath(e.target.value)}
                      required
                    />
                    <input
                      className="form-control"
                      placeholder="Source pack folder"
                      value={projectSourceFolder}
                      onChange={(e) => setProjectSourceFolder(e.target.value)}
                      required
                    />
                      <label className="form-check project-check">
                        <input
                          type="checkbox"
                          checked={projectFullCopy}
                          onChange={(e) => setProjectFullCopy(e.target.checked)}
                        />
                        <span>Copy full project (default: component only)</span>
                      </label>
                      <label className="form-check project-check">
                        <input
                          type="checkbox"
                          checked={createIsAi}
                          onChange={(e) => setCreateIsAi(e.target.checked)}
                        />
                        <span>AI-generated</span>
                      </label>
                      {projectFullCopy ? (
                        <>
                          <label className="form-label">Pick source folder</label>
                          <input
                            className="form-control"
                            type="file"
                            webkitdirectory="true"
                            directory="true"
                            onChange={(e) => {
                              const files = Array.from(e.target.files || []);
                              if (!files.length) return;
                              const first = files[0];
                              const path = first.path || "";
                              if (path) {
                                setProjectSourcePath(path);
                                return;
                              }
                              const rel = first.webkitRelativePath || "";
                              const root = rel.split("/")[0];
                              if (root) setProjectSourcePath(root);
                            }}
                          />
                        </>
                      ) : null}
                    <input
                      className="form-control"
                      placeholder="Screenshot URL"
                      value={projectScreenshotUrl}
                      onChange={(e) => setProjectScreenshotUrl(e.target.value)}
                    />
                    <div className="drop-zone">
                      <input
                        type="file"
                        accept="image/*"
                        onChange={(e) => setProjectScreenshotFile(e.target.files?.[0] || null)}
                      />
                      <div>
                        {projectScreenshotFile
                          ? projectScreenshotFile.name
                          : "Screenshot via drag & drop or file"}
                      </div>
                    </div>
                    <div className="project-actions">
                      <button className="btn btn-primary btn-sm" type="submit">
                        Create project
                      </button>
                      <button
                        className="btn btn-outline-dark btn-sm"
                        type="button"
                        onClick={() => setShowCreateProject(false)}
                      >
                        Cancel
                      </button>
                    </div>
                  </form>
                </div>
              </div>
            )}
          </div>
        </section>
      )}
      {view === "tasks" && (
        <section className="panel">
          <div className="section-title">Tasks</div>
          <div className="task-toolbar">
            <button className="btn btn-outline-danger btn-sm" onClick={() => cancelAllTasks()}>
              Cancel all
            </button>
            <button className="btn btn-outline-success btn-sm" onClick={handleCleanupTasks}>
              Remove finished
            </button>
            <button className="btn btn-outline-dark btn-sm" onClick={handleRepairOpenAiQueue}>
              Repair OpenAI queue
            </button>
          </div>
          {queueStatus && (
            <div className="settings-note">
              {`Worker ${queueStatus.worker_busy ? "busy" : "idle"}${
                queueStatus.worker_active_task_id ? ` (task ${queueStatus.worker_active_task_id})` : ""
              } | tasks queued ${queueStatus.tasks?.queued || 0}, running ${queueStatus.tasks?.running || 0} | OpenAI pending ${
                queueStatus.openai_batches?.pending || 0
              }, ready ${queueStatus.openai_batches?.ready || 0}, in_progress ${
                queueStatus.openai_batches?.in_progress || 0
              }, finalizing ${queueStatus.openai_batches?.finalizing || 0}${
                queueStatusUpdatedAt ? ` | updated ${Math.floor(queueAgeMs / 1000)}s ago` : ""
              }${queueIsStale ? " | stale" : ""}`}
            </div>
          )}
          {queueStatusError && <div className="settings-note text-warning">{`Queue offline: ${queueStatusError}`}</div>}
          <div className="tasks-list">
            {tasksLoading && <div className="load-more">Loading tasks...</div>}
            {!tasksLoading && tasks.length === 0 && <div className="empty-state">No tasks</div>}
            {tasks.map((task) => {
              const progress = task.progress || {};
              const total = progress.total ?? 0;
              const done = progress.done ?? 0;
              const percent = total > 0 ? ((done / total) * 100).toFixed(1) : "0.0";
              const errors = progress.errors ?? 0;
              const message = task.message ?? progress.message ?? "";
              const projectName = task.target_id
                ? projectMap.get(task.target_id)?.name
                : null;
              const title = projectName
                ? `${formatTaskLabel(task.kind)} - ${projectName}`
                : formatTaskLabel(task.kind);
              const startedAt = task.started_at || task.created_at;
              const finishedAt = task.finished_at || (task.status === "running" ? new Date().toISOString() : "");
              const durationMs = startedAt ? (new Date(finishedAt || Date.now()).getTime() - new Date(startedAt).getTime()) : 0;
              const durationLabel = formatDuration(durationMs);
              const statusLabel = String(task.status || "queued");
              return (
                <div key={task.id} className={`task-row task-${task.status}`}>
                  <div className="task-main">
                    <div className="task-title">{title}</div>
                    <div className="task-meta">
                      <span className={`task-badge task-badge-${statusLabel}`}>{statusLabel}</span>
                      {total ? `${done}/${total} (${percent}%)` : ""}
                      {errors ? ` • errors ${errors}` : ""}
                      {message ? ` • ${message}` : ""}
                      {durationLabel ? ` • ${durationLabel}` : ""}
                    </div>
                    {total > 0 && (
                      <div className="task-progress">
                        <div className="task-progress-bar" style={{ width: `${percent}%` }} />
                      </div>
                    )}
                  </div>
                  <div className="task-actions">
                    {(task.status === "queued" || task.status === "running") && (
                      <button
                        className="btn btn-outline-danger btn-sm"
                        onClick={() => handleCancelTask(task.id)}
                      >
                        Cancel
                      </button>
                    )}
                    {(task.status === "done" || task.status === "error" || task.status === "canceled") && (
                      <button
                        className="btn btn-outline-dark btn-sm"
                        onClick={() => handleDeleteTask(task.id)}
                      >
                        Remove
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}

{view === "settings" && (
          <section className="panel">
            <form className="settings-form" onSubmit={handleSaveSettings}>
              <div className="settings-save-floating">
                <button className="btn btn-primary" type="submit">
                  Save settings
                </button>
              </div>
              <h3>Server settings</h3>
              <div className="settings-compact-group">
                <div className="settings-compact-title">Core workflow + tags</div>
                <div className="settings-compact-grid">
                  <div className="settings-compact-item">
                    <div className="form-check">
                      <input
                        className="form-check-input"
                        type="checkbox"
                        checked={isTrue(settings.skip_export_if_on_server)}
                        onChange={(e) =>
                          setSettings((prev) => ({ ...prev, skip_export_if_on_server: e.target.checked }))
                        }
                        id="skipExportIfOnServer"
                      />
                      <label className="form-check-label" htmlFor="skipExportIfOnServer">
                        Skip export if asset exists on server
                      </label>
                    </div>
                  </div>
                  <div className="settings-compact-item">
                    <div className="form-check">
                      <input
                        className="form-check-input"
                        type="checkbox"
                        checked={isTrue(settings.export_overwrite_zips)}
                        onChange={(e) =>
                          setSettings((prev) => ({ ...prev, export_overwrite_zips: e.target.checked }))
                        }
                        id="exportOverwriteZips"
                      />
                      <label className="form-check-label" htmlFor="exportOverwriteZips">
                        Overwrite export zips
                      </label>
                    </div>
                  </div>
                  <div className="settings-compact-item">
                    <div className="form-check">
                      <input
                        className="form-check-input"
                        type="checkbox"
                        checked={isTrue(settings.export_upload_after_export)}
                        onChange={(e) =>
                          setSettings((prev) => ({ ...prev, export_upload_after_export: e.target.checked }))
                        }
                        id="exportUploadAfter"
                      />
                      <label className="form-check-label" htmlFor="exportUploadAfter">
                        Upload after export
                      </label>
                    </div>
                  </div>
                  <div className="settings-compact-item">
                    <div className="form-check">
                      <input
                        className="form-check-input"
                        type="checkbox"
                        checked={Boolean(settings.generate_embeddings_on_import)}
                        onChange={(e) =>
                          setSettings((prev) => ({ ...prev, generate_embeddings_on_import: e.target.checked }))
                        }
                        id="generateEmbeddingsOnImport"
                      />
                      <label className="form-check-label" htmlFor="generateEmbeddingsOnImport">
                        Generate semantic embeddings on import
                      </label>
                    </div>
                  </div>
                  <div className="settings-compact-item">
                    <div className="form-check">
                      <input
                        className="form-check-input"
                        type="checkbox"
                        checked={Boolean(settings.default_full_project_copy)}
                        onChange={(e) =>
                          setSettings((prev) => ({ ...prev, default_full_project_copy: e.target.checked }))
                        }
                        id="defaultFullProjectCopy"
                      />
                      <label className="form-check-label" htmlFor="defaultFullProjectCopy">
                        Default new projects to full project copy
                      </label>
                    </div>
                  </div>
                  <div className="settings-compact-item">
                    <div className="form-check">
                      <input
                        className="form-check-input"
                        type="checkbox"
                        checked={Number(settings.tag_display_limit) === 5}
                        onChange={(e) =>
                          setSettings((prev) => ({ ...prev, tag_display_limit: e.target.checked ? 5 : 0 }))
                        }
                        id="tagDisplayLimit"
                      />
                      <label className="form-check-label" htmlFor="tagDisplayLimit">
                        Show only top 5 tags (display only)
                      </label>
                    </div>
                  </div>
                  <div className="settings-compact-item settings-row-break">
                    <div className="form-row">
                      <label className="form-label">Sidebar width (px)</label>
                      <input
                        className="form-control"
                        type="number"
                        min="200"
                        max="520"
                        step="10"
                        value={settings.sidebar_width ?? 280}
                        onChange={(e) =>
                          setSettings((prev) => ({ ...prev, sidebar_width: Number(e.target.value) }))
                        }
                      />
                    </div>
                  </div>
                  <div className="settings-compact-item">
                    <div className="form-row">
                      <label className="form-label">Sidebar project wrap (chars)</label>
                      <input
                        className="form-control"
                        type="number"
                        min="10"
                        max="200"
                        step="5"
                        value={settings.project_name_wrap_chars ?? 90}
                        onChange={(e) =>
                          setSettings((prev) => ({
                            ...prev,
                            project_name_wrap_chars: Number(e.target.value || 0),
                          }))
                        }
                      />
                    </div>
                  </div>
                  <div className="settings-compact-item">
                    <div className="form-row">
                      <label className="form-label">Unreal command executable</label>
                      <input
                        className="form-control"
                        placeholder="UnrealEditor-Cmd path"
                        value={settings.ue_cmd_path || "I:/epic/UE_5.7/Engine/Binaries/Win64/UnrealEditor-Cmd.exe"}
                        onChange={(e) => setSettings((prev) => ({ ...prev, ue_cmd_path: e.target.value }))}
                      />
                    </div>
                  </div>
                  <div className="settings-compact-item">
                    <div className="form-row">
                      <label className="form-label">Listener base URL</label>
                      <input
                        className="form-control"
                        placeholder="http://127.0.0.1:9090"
                        value={settings.import_base_url || ""}
                        onChange={(e) => setSettings((prev) => ({ ...prev, import_base_url: e.target.value }))}
                      />
                    </div>
                  </div>
                </div>
              </div>
              {/* Hidden advanced paths: kept in settings payload, not shown in UI */}
              <div className="settings-compact-group">
                <div className="settings-compact-title">Capture + tag tuning</div>
                <div className="export-image-grid">
                  <div>
                    <label className="form-label">Discard frames</label>
                    <input
                      className="form-control"
                      type="number"
                      min="0"
                      max="20"
                      value={settings.export_capture360_discard_frames ?? 2}
                      onChange={(e) =>
                        setSettings((prev) => ({ ...prev, export_capture360_discard_frames: Number(e.target.value) }))
                      }
                    />
                  </div>
                  <div>
                    <label className="form-label">Default images</label>
                    <input
                      className="form-control"
                      type="number"
                      min="1"
                      max="24"
                      value={settings.export_default_image_count ?? 1}
                      onChange={(e) =>
                        setSettings((prev) => ({ ...prev, export_default_image_count: Number(e.target.value) }))
                      }
                    />
                  </div>
                  <div>
                    <label className="form-label">StaticMesh</label>
                    <input
                      className="form-control"
                      type="number"
                      min="1"
                      max="24"
                      value={settings.export_static_mesh_image_count ?? ""}
                      onChange={(e) =>
                        setSettings((prev) => ({ ...prev, export_static_mesh_image_count: e.target.value }))
                      }
                    />
                  </div>
                  <div>
                    <label className="form-label">SkeletalMesh</label>
                    <input
                      className="form-control"
                      type="number"
                      min="1"
                      max="24"
                      value={settings.export_skeletal_mesh_image_count ?? ""}
                      onChange={(e) =>
                        setSettings((prev) => ({ ...prev, export_skeletal_mesh_image_count: e.target.value }))
                      }
                    />
                  </div>
                  <div>
                    <label className="form-label">Material</label>
                    <input
                      className="form-control"
                      type="number"
                      min="1"
                      max="24"
                      value={settings.export_material_image_count ?? ""}
                      onChange={(e) =>
                        setSettings((prev) => ({ ...prev, export_material_image_count: e.target.value }))
                      }
                    />
                  </div>
                  <div>
                    <label className="form-label">Blueprint</label>
                    <input
                      className="form-control"
                      type="number"
                      min="1"
                      max="24"
                      value={settings.export_blueprint_image_count ?? ""}
                      onChange={(e) =>
                        setSettings((prev) => ({ ...prev, export_blueprint_image_count: e.target.value }))
                      }
                    />
                  </div>
                  <div>
                    <label className="form-label">Niagara</label>
                    <input
                      className="form-control"
                      type="number"
                      min="1"
                      max="24"
                      value={settings.export_niagara_image_count ?? ""}
                      onChange={(e) =>
                        setSettings((prev) => ({ ...prev, export_niagara_image_count: e.target.value }))
                      }
                    />
                  </div>
                  <div>
                    <label className="form-label">AnimSequence</label>
                    <input
                      className="form-control"
                      type="number"
                      min="1"
                      max="24"
                      value={settings.export_anim_sequence_image_count ?? ""}
                      onChange={(e) =>
                        setSettings((prev) => ({ ...prev, export_anim_sequence_image_count: e.target.value }))
                      }
                    />
                  </div>
                  <div>
                    <label className="form-label">Tag image quality</label>
                    <input
                      className="form-control"
                      type="number"
                      min="30"
                      max="95"
                      step="1"
                      value={settings.tag_image_quality ?? 80}
                      onChange={(e) =>
                        setSettings((prev) => ({ ...prev, tag_image_quality: Number(e.target.value) }))
                      }
                    />
                  </div>
                  <div>
                    <label className="form-label">Min tags for missing</label>
                    <input
                      className="form-control"
                      type="number"
                      min="1"
                      step="1"
                      value={settings.tag_missing_min_tags ?? 1}
                      onChange={(e) =>
                        setSettings((prev) => ({ ...prev, tag_missing_min_tags: e.target.value }))
                      }
                    />
                  </div>
                </div>
              </div>
              <h3>LLM</h3>
              <div className="provider-grid">
                <div
                  className={`provider-card ${settings.provider === "openai" ? "active" : ""}`}
                  onClick={() => settings.provider !== "openai" && handleProviderChange("openai")}
                >
                  <div className="provider-header">
                    <div>OpenAI</div>
                    <span className={`key-dot ${hasOpenAiKey ? "on" : "off"}`} title={ hasOpenAiKey ? "API key saved" : "No API key" } />
                    {settings.provider === "openai" ? (
                      <span className="provider-active">Active</span>
                    ) : (
                      <button
                        className="btn btn-outline-dark btn-xs provider-activate"
                        type="button"
                        onClick={() => handleProviderChange("openai")}
                      >
                        Set active
                      </button>
                    )}

                  </div>
                  {settings.provider === "openai" && (
                    <>
                      <input
                        className="form-control"
                        placeholder="Base URL"
                        value={settings.openai_base_url || ""}
                        onChange={(e) =>
                          setSettings((prev) => ({ ...prev, openai_base_url: e.target.value }))
                        }
                      />
                      <input
                        className="form-control"
                        placeholder="Model"
                        value={settings.openai_model || ""}
                        onChange={(e) =>
                          setSettings((prev) => ({ ...prev, openai_model: e.target.value }))
                        }
                      />
                      <input
                        className="form-control"
                        placeholder="Translation model"
                        value={settings.openai_translate_model || ""}
                        onChange={(e) =>
                          setSettings((prev) => ({ ...prev, openai_translate_model: e.target.value }))
                        }
                      />
                      <input
                        className="form-control"
                        placeholder="API key"
                        value={openaiKeyInput}
                        onChange={(e) => setOpenaiKeyInput(e.target.value)}
                      />
                      <div className="form-row">
                        <label className="form-label">Tag prompt</label>
                        <textarea
                          className="form-control"
                          rows="4"
                          value={settings.tag_prompt_template_openai ?? DEFAULT_TAG_PROMPT}
                          onChange={(e) =>
                            setSettings((prev) => ({ ...prev, tag_prompt_template_openai: e.target.value }))
                          }
                        />
                        <button
                          className="btn btn-outline-dark btn-sm"
                          type="button"
                          onClick={() =>
                            setSettings((prev) => ({
                              ...prev,
                              tag_prompt_template_openai: DEFAULT_TAG_PROMPT,
                            }))
                          }
                        >
                          Reset prompt
                        </button>
                      </div>

                      <div className="llm-test-row">
                        <input
                          className="form-control llm-test-file"
                          type="file"
                          accept="image/*"
                          onChange={(e) => setLlmTestImage(e.target.files?.[0] || null)}
                        />
                        <button
                          className="btn btn-outline-dark btn-sm"
                          type="button"
                          onClick={handleTestLlmTags}
                        >
                          Test image
                        </button>
                      </div>
                      {llmTestResult && (
                        <div className="llm-test-result">
                          <div className="form-row">
                            <label className="form-label">Tags</label>
                            <input
                              className="form-control"
                              readOnly
                              value={(llmTestResult.tags || []).join(", ")}
                            />
                          </div>
                          {llmTestResult.output && typeof llmTestResult.output === "object" && (
                            <div className="llm-test-kv">
                              {Object.entries(llmTestResult.output).map(([key, value]) => (
                                <div key={key} className="form-row">
                                  <label className="form-label">{key}</label>
                                  <input
                                    className="form-control"
                                    readOnly
                                    value={Array.isArray(value) ? value.join(", ") : String(value)}
                                  />
                                </div>
                              ))}
                            </div>
                          )}
                          {llmTestResult.output && typeof llmTestResult.output !== "object" && (
                            <div className="form-row">
                              <label className="form-label">Output</label>
                              <input className="form-control" readOnly value={String(llmTestResult.output)} />
                            </div>
                          )}
                        </div>
                      )}
                      {hasOpenAiKey && !openaiKeyInput && (
                        <div className="settings-note">Key saved.</div>
                      )}
                    </>
                  )}
                </div>

                <div
                  className={`provider-card ${settings.provider === "openrouter" ? "active" : ""}`}
                  onClick={() => settings.provider !== "openrouter" && handleProviderChange("openrouter")}
                >
                  <div className="provider-header">
                    <div>OpenRouter</div>
                    <span className={`key-dot ${hasOpenRouterKey ? "on" : "off"}`} title={ hasOpenRouterKey ? "API key saved" : "No API key" } />
                    {settings.provider === "openrouter" ? (
                      <span className="provider-active">Active</span>
                    ) : (
                      <button
                        className="btn btn-outline-dark btn-xs provider-activate"
                        type="button"
                        onClick={() => handleProviderChange("openrouter")}
                      >
                        Set active
                      </button>
                    )}

                  </div>
                  {settings.provider === "openrouter" && (
                    <>
                      <input
                        className="form-control"
                        placeholder="Base URL"
                        value={settings.openrouter_base_url || ""}
                        onChange={(e) =>
                          setSettings((prev) => ({ ...prev, openrouter_base_url: e.target.value }))
                        }
                      />
                      <input
                        className="form-control"
                        placeholder="Model"
                        value={settings.openrouter_model || ""}
                        onChange={(e) =>
                          setSettings((prev) => ({ ...prev, openrouter_model: e.target.value }))
                        }
                      />
                      <input
                        className="form-control"
                        placeholder="Translation model"
                        value={settings.openrouter_translate_model || ""}
                        onChange={(e) =>
                          setSettings((prev) => ({ ...prev, openrouter_translate_model: e.target.value }))
                        }
                      />
                      <input
                        className="form-control"
                        placeholder="API key"
                        value={openrouterKeyInput}
                        onChange={(e) => setOpenrouterKeyInput(e.target.value)}
                      />
                      <div className="form-row">
                        <label className="form-label">Tag prompt</label>
                        <textarea
                          className="form-control"
                          rows="4"
                          value={settings.tag_prompt_template_openrouter ?? DEFAULT_TAG_PROMPT}
                          onChange={(e) =>
                            setSettings((prev) => ({ ...prev, tag_prompt_template_openrouter: e.target.value }))
                          }
                        />
                        <button
                          className="btn btn-outline-dark btn-sm"
                          type="button"
                          onClick={() =>
                            setSettings((prev) => ({
                              ...prev,
                              tag_prompt_template_openrouter: DEFAULT_TAG_PROMPT,
                            }))
                          }
                        >
                          Reset prompt
                        </button>

                      <div className="llm-test-row">
                        <input
                          className="form-control llm-test-file"
                          type="file"
                          accept="image/*"
                          onChange={(e) => setLlmTestImage(e.target.files?.[0] || null)}
                        />
                        <button
                          className="btn btn-outline-dark btn-sm"
                          type="button"
                          onClick={handleTestLlmTags}
                        >
                          Test image
                        </button>
                      </div>
                      {llmTestResult && (
                        <div className="llm-test-result">
                          <div className="form-row">
                            <label className="form-label">Tags</label>
                            <input
                              className="form-control"
                              readOnly
                              value={(llmTestResult.tags || []).join(", ")}
                            />
                          </div>
                          {llmTestResult.output && typeof llmTestResult.output === "object" && (
                            <div className="llm-test-kv">
                              {Object.entries(llmTestResult.output).map(([key, value]) => (
                                <div key={key} className="form-row">
                                  <label className="form-label">{key}</label>
                                  <input
                                    className="form-control"
                                    readOnly
                                    value={Array.isArray(value) ? value.join(", ") : String(value)}
                                  />
                                </div>
                              ))}
                            </div>
                          )}
                          {llmTestResult.output && typeof llmTestResult.output !== "object" && (
                            <div className="form-row">
                              <label className="form-label">Output</label>
                              <input className="form-control" readOnly value={String(llmTestResult.output)} />
                            </div>
                          )}
                        </div>
                      )}
                      </div>
                      {hasOpenRouterKey && !openrouterKeyInput && (
                        <div className="settings-note">Key saved.</div>
                      )}
                    </>
                  )}
                </div>

                <div
                  className={`provider-card ${settings.provider === "groq" ? "active" : ""}`}
                  onClick={() => settings.provider !== "groq" && handleProviderChange("groq")}
                >
                  <div className="provider-header">
                    <div>Groq</div>
                    <span className={`key-dot ${hasGroqKey ? "on" : "off"}`} title={ hasGroqKey ? "API key saved" : "No API key" } />
                    {settings.provider === "groq" ? (
                      <span className="provider-active">Active</span>
                    ) : (
                      <button
                        className="btn btn-outline-dark btn-xs provider-activate"
                        type="button"
                        onClick={() => handleProviderChange("groq")}
                      >
                        Set active
                      </button>
                    )}

                  </div>
                  {settings.provider === "groq" && (
                    <>
                      <input
                        className="form-control"
                        placeholder="Base URL"
                        value={settings.groq_base_url || ""}
                        onChange={(e) =>
                          setSettings((prev) => ({ ...prev, groq_base_url: e.target.value }))
                        }
                      />
                      <input
                        className="form-control"
                        placeholder="Model"
                        value={settings.groq_model || ""}
                        onChange={(e) => setSettings((prev) => ({ ...prev, groq_model: e.target.value }))}
                      />
                      <input
                        className="form-control"
                        placeholder="API key"
                        value={groqKeyInput}
                        onChange={(e) => setGroqKeyInput(e.target.value)}
                      />
                      <div className="form-row">
                        <label className="form-label">Tag prompt</label>
                        <textarea
                          className="form-control"
                          rows="4"
                          value={settings.tag_prompt_template_groq ?? DEFAULT_TAG_PROMPT}
                          onChange={(e) =>
                            setSettings((prev) => ({ ...prev, tag_prompt_template_groq: e.target.value }))
                          }
                        />
                        <button
                          className="btn btn-outline-dark btn-sm"
                          type="button"
                          onClick={() =>
                            setSettings((prev) => ({
                              ...prev,
                              tag_prompt_template_groq: DEFAULT_TAG_PROMPT,
                            }))
                          }
                        >
                          Reset prompt
                        </button>

                      <div className="llm-test-row">
                        <input
                          className="form-control llm-test-file"
                          type="file"
                          accept="image/*"
                          onChange={(e) => setLlmTestImage(e.target.files?.[0] || null)}
                        />
                        <button
                          className="btn btn-outline-dark btn-sm"
                          type="button"
                          onClick={handleTestLlmTags}
                        >
                          Test image
                        </button>
                      </div>
                      {llmTestResult && (
                        <div className="llm-test-result">
                          <div className="form-row">
                            <label className="form-label">Tags</label>
                            <input
                              className="form-control"
                              readOnly
                              value={(llmTestResult.tags || []).join(", ")}
                            />
                          </div>
                          {llmTestResult.output && typeof llmTestResult.output === "object" && (
                            <div className="llm-test-kv">
                              {Object.entries(llmTestResult.output).map(([key, value]) => (
                                <div key={key} className="form-row">
                                  <label className="form-label">{key}</label>
                                  <input
                                    className="form-control"
                                    readOnly
                                    value={Array.isArray(value) ? value.join(", ") : String(value)}
                                  />
                                </div>
                              ))}
                            </div>
                          )}
                          {llmTestResult.output && typeof llmTestResult.output !== "object" && (
                            <div className="form-row">
                              <label className="form-label">Output</label>
                              <input className="form-control" readOnly value={String(llmTestResult.output)} />
                            </div>
                          )}
                        </div>
                      )}
                      </div>
                      {hasGroqKey && !groqKeyInput && (
                        <div className="settings-note">Key saved.</div>
                      )}
                    </>
                  )}
                </div>

                <div
                  className={`provider-card ${settings.provider === "ollama" ? "active" : ""}`}
                  onClick={() => settings.provider !== "ollama" && handleProviderChange("ollama")}
                >
                  <div className="provider-header">
                    <div>Ollama</div>
                    {settings.provider === "ollama" ? (
                      <span className="provider-active">Active</span>
                    ) : (
                      <button
                        className="btn btn-outline-dark btn-xs provider-activate"
                        type="button"
                        onClick={() => handleProviderChange("ollama")}
                      >
                        Set active
                      </button>
                    )}
                  </div>
                  {settings.provider === "ollama" && (
                    <>
                      <input
                        className="form-control"
                        placeholder="Base URL"
                        value={settings.ollama_base_url || ""}
                        onChange={(e) =>
                          setSettings((prev) => ({ ...prev, ollama_base_url: e.target.value }))
                        }
                      />
                      <input
                        className="form-control"
                        placeholder="Model"
                        value={settings.ollama_model || ""}
                        onChange={(e) =>
                          setSettings((prev) => ({ ...prev, ollama_model: e.target.value }))
                        }
                      />
                      <input
                        className="form-control"
                        placeholder="Translation model"
                        value={settings.ollama_translate_model || ""}
                        onChange={(e) =>
                          setSettings((prev) => ({ ...prev, ollama_translate_model: e.target.value }))
                        }
                      />
                      <div className="form-row">
                        <label className="form-label">Tag prompt</label>
                        <textarea
                          className="form-control"
                          rows="4"
                          value={settings.tag_prompt_template_ollama ?? DEFAULT_TAG_PROMPT}
                          onChange={(e) =>
                            setSettings((prev) => ({ ...prev, tag_prompt_template_ollama: e.target.value }))
                          }
                        />
                        <button
                          className="btn btn-outline-dark btn-sm"
                          type="button"
                          onClick={() =>
                            setSettings((prev) => ({
                              ...prev,
                              tag_prompt_template_ollama: DEFAULT_TAG_PROMPT,
                            }))
                          }
                        >
                          Reset prompt
                        </button>

                      <div className="llm-test-row">
                        <input
                          className="form-control llm-test-file"
                          type="file"
                          accept="image/*"
                          onChange={(e) => setLlmTestImage(e.target.files?.[0] || null)}
                        />
                        <button
                          className="btn btn-outline-dark btn-sm"
                          type="button"
                          onClick={handleTestLlmTags}
                        >
                          Test image
                        </button>
                      </div>
                      {llmTestResult && (
                        <div className="llm-test-result">
                          <div className="form-row">
                            <label className="form-label">Tags</label>
                            <input
                              className="form-control"
                              readOnly
                              value={(llmTestResult.tags || []).join(", ")}
                            />
                          </div>
                          {llmTestResult.output && typeof llmTestResult.output === "object" && (
                            <div className="llm-test-kv">
                              {Object.entries(llmTestResult.output).map(([key, value]) => (
                                <div key={key} className="form-row">
                                  <label className="form-label">{key}</label>
                                  <input
                                    className="form-control"
                                    readOnly
                                    value={Array.isArray(value) ? value.join(", ") : String(value)}
                                  />
                                </div>
                              ))}
                            </div>
                          )}
                          {llmTestResult.output && typeof llmTestResult.output !== "object" && (
                            <div className="form-row">
                              <label className="form-label">Output</label>
                              <input className="form-control" readOnly value={String(llmTestResult.output)} />
                            </div>
                          )}
                        </div>
                      )}
                      </div>
                      <div className="settings-note">No API key required.</div>
                    </>
                  )}
                </div>
              </div>
              <div className="settings-row">
                <button className="btn btn-outline-dark btn-sm btn-white-text" type="button" onClick={handleRegenerateEmbeddingsAll}>
                  Rebuild embeddings (all)
                </button>
              </div>
              <div className="settings-compact-group">
                <div className="settings-compact-title">LLM runtime</div>
                <div className="settings-compact-grid">
                  <div className="settings-compact-item">
                    <div className="form-check">
                      <input
                        className="form-check-input"
                        type="checkbox"
                        checked={Boolean(settings.tag_use_batch_mode)}
                        onChange={(e) =>
                          setSettings((prev) => ({ ...prev, tag_use_batch_mode: e.target.checked }))
                        }
                        id="tagBatchMode"
                      />
                      <label className="form-check-label batch-mode-label" htmlFor="tagBatchMode">
                        Tag batch mode (OpenAI/Groq)
                      </label>
                    </div>
                  </div>
                  <div className={`settings-compact-item ${!settings.tag_use_batch_mode ? "settings-hidden" : ""}`}>
                    <div className="form-row">
                      <label className="form-label">Batch size (assets per batch)</label>
                      <input
                        className="form-control"
                        type="number"
                        min="1"
                        max="50000"
                        step="1"
                        value={settings.tag_batch_max_assets ?? 500}
                        onChange={(e) =>
                          setSettings((prev) => ({
                            ...prev,
                            tag_batch_max_assets: Number(e.target.value || 0),
                          }))
                        }
                        disabled={!settings.tag_use_batch_mode}
                      />
                    </div>
                  </div>
                  <div className={`settings-compact-item ${!settings.tag_use_batch_mode ? "settings-hidden" : ""}`}>
                    <div className="form-row">
                      <label className="form-label">Project batch concurrency</label>
                      <input
                        className="form-control"
                        type="number"
                        min="1"
                        max="10"
                        step="1"
                        value={settings.tag_batch_project_concurrency ?? 3}
                        onChange={(e) =>
                          setSettings((prev) => ({
                            ...prev,
                            tag_batch_project_concurrency: Number(e.target.value || 0),
                          }))
                        }
                        disabled={!settings.tag_use_batch_mode}
                      />
                    </div>
                  </div>
                </div>
                <div className="llm-runtime-grid4">
                  <div className="settings-compact-item">
                    <div className="form-row">
                      <label className="form-label">Tag language</label>
                      <select
                        className="form-select"
                        value={settings.tag_language || "english"}
                        onChange={(e) => setSettings((prev) => ({ ...prev, tag_language: e.target.value }))}
                      >
                        <option value="english">English</option>
                        <option value="german">German</option>
                        <option value="spanish">Spanish</option>
                        <option value="french">French</option>
                      </select>
                    </div>
                  </div>
                  <div className="settings-compact-item">
                    <div className="form-row">
                      <label className="form-label">Tag image size (px)</label>
                      <input
                        className="form-control"
                        type="number"
                        min="64"
                        max="2048"
                        step="32"
                        placeholder="Tag image size (px)"
                        value={settings.tag_image_size ?? 512}
                        onChange={(e) => setSettings((prev) => ({ ...prev, tag_image_size: Number(e.target.value) }))}
                      />
                    </div>
                  </div>
                  <div className="settings-compact-item">
                    <div className="form-row">
                      <label className="form-label">Cooldown (s)</label>
                      <input
                        className="form-control"
                        type="number"
                        min="0"
                        step="0.1"
                        value={settings.llm_min_interval_seconds ?? 0}
                        onChange={(e) =>
                          setSettings((prev) => ({
                            ...prev,
                            llm_min_interval_seconds: Number(e.target.value || 0),
                          }))
                        }
                      />
                    </div>
                  </div>
                  <div className="settings-compact-item">
                    <div className="settings-inline-row llm-temperature-row">
                      <label className="form-check">
                        <input
                          className="form-check-input"
                          type="checkbox"
                          checked={Boolean(settings.use_temperature)}
                          onChange={(e) => setSettings((prev) => ({ ...prev, use_temperature: e.target.checked }))}
                        />
                        <span className="form-check-label">Use temperature</span>
                      </label>
                      <input
                        className="form-control"
                        type="number"
                        min="0"
                        max="2"
                        step="0.1"
                        placeholder="Temperature (0-2)"
                        value={settings.temperature ?? 1}
                        onChange={(e) => setSettings((prev) => ({ ...prev, temperature: Number(e.target.value) }))}
                        disabled={!settings.use_temperature}
                      />
                    </div>
                  </div>
                </div>
                <div className="settings-compact-grid">
                  <div className="settings-note llm-runtime-note">
                    OpenAI/Groq only. Sends multiple tag requests per batch file.
                  </div>
                  <div className="settings-note llm-runtime-note">
                    Image size detection: larger images can yield more tags, but cost more.
                  </div>
                </div>
              </div>
              <div className="settings-two-col">
                <div className="settings-box">
                  <label className="form-label">Tag include types (optional)</label>
                  <div className="filter-list tag-filter-list filter-grid-4">
                    {assetTypes.map((assetType) => (
                      <label key={`settings-include-${assetType}`} className="filter-item">
                        <input
                          type="checkbox"
                          checked={settingsTagIncludeTypes.includes(assetType)}
                          onChange={() =>
                            toggleTypeList(
                              assetType,
                              settingsTagIncludeTypes,
                              setSettingsTagIncludeTypes,
                              settingsTagExcludeTypes,
                              setSettingsTagExcludeTypes
                            )
                          }
                        />
                        {assetType}
                      </label>
                    ))}
                  </div>
                </div>
                <div className="settings-box settings-split">
                  <label className="form-label">Tag exclude types</label>
                  <div className="filter-list tag-filter-list filter-grid-4">
                    {assetTypes.map((assetType) => (
                      <label key={`settings-exclude-${assetType}`} className="filter-item">
                        <input
                          type="checkbox"
                          checked={settingsTagExcludeTypes.includes(assetType)}
                          onChange={() =>
                            toggleTypeList(
                              assetType,
                              settingsTagExcludeTypes,
                              setSettingsTagExcludeTypes,
                              settingsTagIncludeTypes,
                              setSettingsTagIncludeTypes
                            )
                          }
                        />
                        {assetType}
                      </label>
                    ))}
                  
                  </div>
                </div>
              </div>
              <div className="settings-two-col">
                <div className="settings-box">
                  <label className="form-label">Export include types (optional)</label>
                  <div className="filter-list filter-list-full filter-grid-4">
                    {assetTypes.map((assetType) => (
                      <label key={`settings-export-include-${assetType}`} className="filter-item">
                        <input
                          type="checkbox"
                          checked={settingsExportIncludeTypes.includes(assetType)}
                          onChange={() =>
                            toggleTypeList(
                              assetType,
                              settingsExportIncludeTypes,
                              setSettingsExportIncludeTypes,
                              settingsExportExcludeTypes,
                              setSettingsExportExcludeTypes
                            )
                          }
                        />
                        {assetType}
                      </label>
                    ))}
                  </div>
                </div>
                <div className="settings-box settings-split">
                  <label className="form-label">Export exclude types</label>
                  <div className="filter-list filter-list-full filter-grid-4">
                    {assetTypes.map((assetType) => (
                      <label key={`settings-export-exclude-${assetType}`} className="filter-item">
                        <input
                          type="checkbox"
                          checked={settingsExportExcludeTypes.includes(assetType)}
                          onChange={() =>
                            toggleTypeList(
                              assetType,
                              settingsExportExcludeTypes,
                              setSettingsExportExcludeTypes,
                              settingsExportIncludeTypes,
                              setSettingsExportIncludeTypes
                            )
                          }
                        />
                        {assetType}
                      </label>
                    ))}
                  
                  </div>
                </div>
              </div>
            <button className="btn btn-primary" type="submit">
              Save settings
            </button>
            <p className="settings-note">
              Use the active provider settings above.
            </p>
            </form>
            <div className="settings-form">
              <h3>Tag import/export (CSV)</h3>
              <div className="tag-io-grid">
                <div className="tag-io-card">
                  <h4>Export</h4>
                  <div className="form-row">
                    <label className="form-label">Export hash</label>
                    <select
                      className="form-select"
                      value={tagExportHash}
                      onChange={(e) => setTagExportHash(e.target.value)}
                    >
                      <option value="blake3">blake3 (hash_main_blake3)</option>
                      <option value="sha256">sha256 (hash_main_sha256)</option>
                    </select>
                  </div>
                  <div className="form-row">
                    <label className="form-label">Export project (optional)</label>
                    <select
                      className="form-select"
                      value={tagExportProjectId}
                      onChange={(e) => setTagExportProjectId(e.target.value)}
                    >
                      <option value="">All projects</option>
                      {projects.map((project) => (
                        <option key={project.id} value={project.id}>
                          {project.name}
                        </option>
                      ))}
                    </select>
                  </div>
                  <button className="btn btn-outline-dark btn-sm" type="button" onClick={handleExportTags}>
                    Export tags
                  </button>
                </div>
                <div className="tag-io-card">
                  <h4>Import</h4>
                  <div className="form-row">
                    <label className="form-label">Import hash</label>
                    <select
                      className="form-select"
                      value={tagImportHash}
                      onChange={(e) => setTagImportHash(e.target.value)}
                    >
                      <option value="blake3">blake3 (hash_main_blake3)</option>
                      <option value="sha256">sha256 (hash_main_sha256)</option>
                    </select>
                  </div>
                  <div className="form-row">
                    <label className="form-label">Import project (optional)</label>
                    <select
                      className="form-select"
                      value={tagImportProjectId}
                      onChange={(e) => setTagImportProjectId(e.target.value)}
                    >
                      <option value="">All projects</option>
                      {projects.map((project) => (
                        <option key={project.id} value={project.id}>
                          {project.name}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="form-row">
                    <label className="form-label">Import mode</label>
                    <select
                      className="form-select"
                      value={tagImportMode}
                      onChange={(e) => setTagImportMode(e.target.value)}
                    >
                      <option value="replace">Replace tags</option>
                      <option value="merge">Merge tags</option>
                    </select>
                  </div>
                  <div className="form-row">
                    <label className="filter-item">
                      <input
                        type="checkbox"
                        checked={Boolean(settings.tag_translate_enabled)}
                        onChange={(e) =>
                          setSettings((prev) => ({ ...prev, tag_translate_enabled: e.target.checked }))
                        }
                      />
                      Translate tags via LLM (also on import)
                    </label>
                  </div>
                  <input
                    className="form-control"
                    type="file"
                    accept=".csv"
                    onChange={(e) => setTagImportFile(e.target.files?.[0] || null)}
                  />
                  <button className="btn btn-outline-dark btn-sm" type="button" onClick={handleImportTags}>
                    Import tags
                  </button>
                </div>
              </div>
            </div>
            <div className="tag-io-danger">
              <div className="tag-io-danger-title">Danger zone</div>
              <div className="tag-io-danger-text">
                Deletes all tags for all assets. This cannot be undone.
              </div>
              <button className="btn btn-outline-danger btn-sm" type="button" onClick={handleClearAllTags}>
                Delete all tags
              </button>
            </div>
          <div className="settings-danger" style={{ marginTop: "300px" }}>
            <button
              className="btn btn-outline-danger"
              type="button"
              onClick={() => setShowResetModal(true)}
            >
              Reset database (keep settings)
            </button>
          </div>
        </section>
        )}
        {projectPreview && (
          <div className="modal-overlay" onClick={() => setProjectPreview(null)}>
            <div className="image-modal" onClick={(e) => e.stopPropagation()}>
              <img src={projectPreview} alt="Project preview" />
              <button
                className="modal-close"
                type="button"
                onClick={() => setProjectPreview(null)}
                aria-label="Close"
              >
                x
              </button>
            </div>
          </div>
        )}
        {detailImagePreview && (
          <div className="modal-overlay" onClick={() => setDetailImagePreview(null)}>
            <div className="image-modal image-modal-large" onClick={(e) => e.stopPropagation()}>
              <img src={detailImagePreview} alt="Asset preview" />
              <button
                className="modal-close"
                type="button"
                onClick={() => setDetailImagePreview(null)}
                aria-label="Close"
              >
                x
              </button>
            </div>
          </div>
        )}
        {showResetModal && (
          <div className="modal-overlay" onClick={() => setShowResetModal(false)}>
            <div className="project-modal" onClick={(e) => e.stopPropagation()}>
              <h3>Reset database</h3>
              <p className="settings-note">
                Type OK to confirm. This will delete all projects and assets (settings are kept).
                A backup will be created first.
              </p>
              <input
                className="form-control"
                placeholder="Type OK"
                value={resetConfirmInput}
                onChange={(e) => setResetConfirmInput(e.target.value)}
              />
              <div className="project-actions">
                <button
                  className="btn btn-danger btn-sm"
                  type="button"
                  disabled={resetConfirmInput.trim().toLowerCase() !== "ok"}
                  onClick={async () => {
                    setShowResetModal(false);
                    setResetConfirmInput("");
                    await handleResetDatabase();
                  }}
                >
                  Reset database
                </button>
                <button
                  className="btn btn-outline-dark btn-sm"
                  type="button"
                  onClick={() => {
                    setShowResetModal(false);
                    setResetConfirmInput("");
                  }}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}
        <ToastContainer position="bottom-right" autoClose={3500} hideProgressBar={false} newestOnTop />
        {view !== "settings" && (
          <button
            className="scroll-top-fab"
            type="button"
            onClick={() => {
              lastAssetScrollYRef.current = null;
              window.scrollTo({ top: 0, behavior: "smooth" });
            }}
            aria-label="Back to top"
            title="Back to top"
          >
            ^
          </button>
        )}
    </div>
  );
}
