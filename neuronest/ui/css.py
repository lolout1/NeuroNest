MAIN_CSS = """
/* Global */
.container { max-width: 1400px; margin: auto; padding: 0 16px; }

/* Header */
.hero {
    text-align: center; padding: 28px 16px 12px;
    background: linear-gradient(135deg, #eef2ff 0%, #faf5ff 50%, #ecfdf5 100%);
    border-radius: 16px; margin-bottom: 20px; border: 1px solid #e0e7ff;
}
.hero h1 { font-size: 2.2em; margin: 0 0 4px; font-weight: 800; color: #1e1b4b; letter-spacing: -0.5px; }
.hero-sub { color: #4b5563; font-size: 1em; margin: 0 0 14px; line-height: 1.6; max-width: 800px; display: inline-block; }
.metrics-row { display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; margin: 12px 0 8px; }
.metric {
    display: inline-flex; flex-direction: column; align-items: center;
    padding: 10px 18px; border-radius: 12px; min-width: 100px;
    background: white; border: 1px solid #e5e7eb; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.metric-val { font-size: 1.35em; font-weight: 800; color: #4f46e5; line-height: 1.2; }
.metric-label { font-size: 0.7em; color: #6b7280; font-weight: 500; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.5px; }
.badge-row { text-align: center; margin: 8px 0 4px; }
.badge {
    display: inline-block; padding: 3px 10px; margin: 2px 3px;
    border-radius: 16px; font-size: 0.72em; font-weight: 600;
    background: #f5f3ff; color: #5b21b6; border: 1px solid #ddd6fe;
}

/* Try Now buttons */
.try-now-row { display: flex; justify-content: center; gap: 12px; margin: 18px 0 6px; flex-wrap: wrap; }
.try-now-btn {
    padding: 12px 32px; border-radius: 10px; font-size: 1em; font-weight: 700;
    cursor: pointer; border: none; letter-spacing: 0.3px;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: white; box-shadow: 0 4px 14px rgba(79,70,229,0.35);
    transition: all 0.2s ease;
}
.try-now-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(79,70,229,0.45); }
.try-now-btn:active { transform: translateY(0); }
.try-now-secondary {
    background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%) !important;
    box-shadow: 0 4px 14px rgba(124,58,237,0.3) !important;
}
.try-now-secondary:hover { box-shadow: 0 6px 20px rgba(124,58,237,0.4) !important; }

/* Nav Tabs */
.nav-tabs > .tab-nav {
    display: flex; gap: 4px; padding: 6px;
    background: #f9fafb; border-radius: 12px; border: 1px solid #e5e7eb;
    justify-content: center; flex-wrap: wrap;
}
.nav-tabs > .tab-nav button {
    padding: 8px 20px !important; border-radius: 8px !important;
    font-weight: 500 !important; font-size: 0.9em !important;
    border: none !important; transition: all 0.15s ease !important;
}
.nav-tabs > .tab-nav button:hover { background: #eef2ff !important; }
.nav-tabs > .tab-nav button.selected {
    background: #4f46e5 !important; color: white !important;
    box-shadow: 0 2px 8px rgba(79,70,229,0.25) !important;
}

/* Responsive */
@media (max-width: 768px) {
    .container { padding: 0 8px; }
    .hero { padding: 16px 10px 8px; }
    .hero h1 { font-size: 1.6em; }
    .hero-sub { font-size: 0.85em; }
    .nav-tabs > .tab-nav { overflow-x: auto; flex-wrap: nowrap; justify-content: flex-start; -webkit-overflow-scrolling: touch; }
    .nav-tabs > .tab-nav button { white-space: nowrap; font-size: 0.82em !important; padding: 6px 14px !important; }
    .metrics-row { gap: 6px; }
    .metric { padding: 6px 10px; min-width: 70px; }
    .metric-val { font-size: 1.1em; }
    .badge { font-size: 0.65em; padding: 2px 8px; }
    .try-now-row { gap: 8px; margin: 12px 0 4px; }
    .try-now-btn { padding: 10px 22px; font-size: 0.9em; }
    .sample-section { padding: 10px; }
    .xai-controls { padding: 10px; }
    .main-button { height: 46px !important; font-size: 0.95em !important; }
    .report-box { padding: 12px; }
}
@media (max-width: 480px) {
    .hero h1 { font-size: 1.3em; }
    .hero-sub { font-size: 0.78em; }
    .metrics-row { gap: 4px; }
    .metric { padding: 4px 8px; min-width: 60px; }
    .metric-val { font-size: 0.95em; }
    .metric-label { font-size: 0.6em; }
}

/* Buttons */
.main-button {
    height: 52px !important; font-size: 1.05em !important; font-weight: 600 !important;
    width: 100% !important; border-radius: 10px !important;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    border: none !important; color: white !important;
    box-shadow: 0 2px 8px rgba(79,70,229,0.3) !important;
    transition: all 0.2s ease !important;
}
.main-button:hover { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(79,70,229,0.4) !important; }

/* Sections */
.sample-section { padding: 16px; margin-bottom: 16px; background: #fafbfc; border-radius: 12px; border: 1px solid #e5e7eb; }
.controls-row { padding: 16px; margin-bottom: 16px; background: #f9fafb; border-radius: 12px; border: 1px solid #e5e7eb; }
.report-box { max-width: 100%; margin: 16px 0; padding: 20px; background: #fff; border-radius: 12px; border: 1px solid #e5e7eb; line-height: 1.7; }
.info-card { padding: 20px; background: #fafbfc; border-radius: 12px; border: 1px solid #e5e7eb; line-height: 1.7; margin: 8px 0; }

/* Images */
.image-output { margin: 8px 0; }
.image-output img { width: 100%; max-width: 100%; border-radius: 10px; border: 1px solid #e5e7eb; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }

/* XAI */
.xai-controls { padding: 16px; background: #f9fafb; border-radius: 12px; border: 1px solid #e5e7eb; margin-bottom: 16px; }
.xai-report { max-width: 100%; margin: 16px 0; border-radius: 12px; border: 1px solid #e5e7eb; overflow-y: auto; }

/* Examples */
.examples-holder img { border-radius: 8px; cursor: pointer; transition: all 0.2s; border: 2px solid transparent; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
.examples-holder img:hover { border-color: #4f46e5; box-shadow: 0 4px 12px rgba(79,70,229,0.2); }

/* Dark mode */
@media (prefers-color-scheme: dark) {
    .hero { background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #064e3b 100%); border-color: #4338ca; }
    .hero h1 { color: #e0e7ff; }
    .hero-sub { color: #c7d2fe; }
    .metric { background: #1f2937; border-color: #374151; }
    .metric-val { color: #a5b4fc; }
    .metric-label { color: #9ca3af; }
    .badge { background: #312e81; color: #c7d2fe; border-color: #4338ca; }
    .sidebar-tabs > .tab-nav { background: #1f2937; border-color: #374151; }
    .sidebar-tabs > .tab-nav button:hover { background: #312e81 !important; }
    .sample-section, .controls-row, .xai-controls, .info-card { background: #1f2937; border-color: #374151; }
    .report-box, .xai-report { background: #1f2937; border-color: #374151; }
    .xai-report table th { background: #374151; }
}

/* Info Tabs — lighter, secondary navigation */
.info-tabs > .tab-nav {
    display: flex; gap: 4px; padding: 4px 6px;
    background: #f3f4f6; border-radius: 10px; border: 1px solid #e5e7eb;
    justify-content: center; flex-wrap: wrap; margin-bottom: 8px;
}
.info-tabs > .tab-nav button {
    padding: 6px 16px !important; border-radius: 6px !important;
    font-weight: 500 !important; font-size: 0.85em !important;
    border: none !important; color: #4b5563 !important;
    transition: all 0.15s ease !important;
}
.info-tabs > .tab-nav button:hover { background: #e5e7eb !important; }
.info-tabs > .tab-nav button.selected {
    background: #6366f1 !important; color: white !important;
    box-shadow: 0 1px 4px rgba(99,102,241,0.25) !important;
}

/* Workspace Tabs — prominent, primary */
.workspace-tabs > .tab-nav {
    display: flex; gap: 6px; padding: 6px 8px;
    background: #eef2ff; border-radius: 12px; border: 1px solid #c7d2fe;
    justify-content: center; flex-wrap: wrap; margin-top: 12px;
}
.workspace-tabs > .tab-nav button {
    padding: 10px 28px !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 1em !important;
    border: none !important; transition: all 0.15s ease !important;
}
.workspace-tabs > .tab-nav button:hover { background: #ddd6fe !important; }
.workspace-tabs > .tab-nav button.selected {
    background: #4f46e5 !important; color: white !important;
    box-shadow: 0 2px 10px rgba(79,70,229,0.3) !important;
}

/* Agent Panel — floating accordion */
.agent-panel {
    margin-top: 20px !important; border: 2px solid #c7d2fe !important;
    border-radius: 12px !important; background: #faf5ff !important;
    box-shadow: 0 4px 16px rgba(79,70,229,0.12) !important;
}
.agent-panel > .label-wrap {
    background: #eef2ff !important; border-radius: 10px 10px 0 0 !important;
    padding: 12px 16px !important;
}
.agent-button {
    background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%) !important;
    color: white !important; font-weight: 600 !important;
    border-radius: 8px !important; border: none !important;
    height: 48px !important; width: 100% !important;
    box-shadow: 0 2px 8px rgba(124,58,237,0.3) !important;
    transition: all 0.2s ease !important;
}
.agent-button:hover { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(124,58,237,0.4) !important; }
.agent-output {
    max-height: 600px; overflow-y: auto; padding: 16px;
    background: white; border-radius: 0 0 10px 10px; border-top: 1px solid #e5e7eb;
    line-height: 1.7;
}

/* Responsive — info/workspace tabs */
@media (max-width: 768px) {
    .info-tabs > .tab-nav { overflow-x: auto; flex-wrap: nowrap; justify-content: flex-start; -webkit-overflow-scrolling: touch; }
    .info-tabs > .tab-nav button { white-space: nowrap; font-size: 0.78em !important; padding: 5px 12px !important; }
    .workspace-tabs > .tab-nav { overflow-x: auto; flex-wrap: nowrap; justify-content: flex-start; -webkit-overflow-scrolling: touch; }
    .workspace-tabs > .tab-nav button { white-space: nowrap; font-size: 0.9em !important; padding: 8px 18px !important; }
    .agent-panel { margin-top: 12px !important; }
    .agent-button { height: 42px !important; font-size: 0.9em !important; }
}
@media (max-width: 480px) {
    .workspace-tabs > .tab-nav button { font-size: 0.82em !important; padding: 6px 14px !important; }
}

/* Accessibility */
@media (prefers-contrast: high) { .sample-section, .controls-row, .xai-controls, .info-card, .agent-panel { border: 2px solid currentColor; } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }

/* Dark mode — info/workspace/agent */
@media (prefers-color-scheme: dark) {
    .info-tabs > .tab-nav { background: #1f2937; border-color: #374151; }
    .info-tabs > .tab-nav button { color: #d1d5db !important; }
    .info-tabs > .tab-nav button:hover { background: #374151 !important; }
    .info-tabs > .tab-nav button.selected { background: #6366f1 !important; color: white !important; }
    .workspace-tabs > .tab-nav { background: #1e1b4b; border-color: #4338ca; }
    .workspace-tabs > .tab-nav button:hover { background: #312e81 !important; }
    .agent-panel { background: #1f2937 !important; border-color: #4338ca !important; }
    .agent-panel > .label-wrap { background: #312e81 !important; }
    .agent-output { background: #1f2937; border-color: #374151; }
}
"""
