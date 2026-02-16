import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..config import OUTDOOR_CLASS_IDS

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

SAFETY_SURFACE_IDS = {
    3: "floor", 13: "earth/ground", 28: "rug", 52: "path",
    53: "stairs", 59: "stairway/staircase", 94: "land/ground",
    96: "escalator", 110: "stool", 121: "step/stair", 37: "bathtub",
    145: "shower", 6: "road",
}

PRIMARY_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
FALLBACK_MODEL = "mistralai/Mistral-Small-24B-Instruct-2501"


@dataclass
class AnalysisContext:
    segmentation: Optional[Dict[str, Any]] = None
    blackspot: Optional[Dict[str, Any]] = None
    contrast: Optional[Dict[str, Any]] = None
    statistics: Dict[str, Any] = field(default_factory=dict)
    xai_report: Optional[str] = None

    @classmethod
    def from_results(cls, results: Dict[str, Any]) -> "AnalysisContext":
        if not results:
            return cls()
        return cls(
            segmentation=results.get("segmentation"),
            blackspot=results.get("blackspot"),
            contrast=results.get("contrast"),
            statistics=results.get("statistics", {}),
            xai_report=results.get("xai_report"),
        )

    @property
    def has_analysis(self) -> bool:
        return any([self.segmentation, self.blackspot, self.contrast, self.xai_report])

    def serialize(self) -> str:
        if not self.has_analysis:
            return ""
        sections = []
        if self.segmentation:
            sections.append(self._serialize_segmentation())
        if self.blackspot:
            sections.append(self._serialize_blackspot())
        if self.contrast:
            sections.append(self._serialize_contrast())
        safety_flags = self._serialize_safety_surfaces()
        if safety_flags:
            sections.append(safety_flags)
        if self.xai_report:
            sections.append(self.xai_report)
        return "\n\n".join(sections)

    def _serialize_segmentation(self) -> str:
        seg_stats = self.statistics.get("segmentation", {})
        image_size = seg_stats.get("image_size", "unknown")
        mask = self.segmentation.get("mask")
        if mask is not None:
            indoor_count = len(set(np.unique(mask).tolist()) - OUTDOOR_CLASS_IDS)
            lines = [
                f"[SEGMENTATION] {indoor_count} indoor classes detected | Resolution: {image_size}",
            ]
            lines.extend(self._class_distribution(mask))
        else:
            num_classes = seg_stats.get("num_classes", 0)
            lines = [
                f"[SEGMENTATION] {num_classes} classes detected | Resolution: {image_size}",
            ]
        return "\n".join(lines)

    def _class_distribution(self, mask: np.ndarray) -> List[str]:
        try:
            from ade20k_classes import ADE20K_NAMES
        except ImportError:
            ADE20K_NAMES = None

        unique, counts = np.unique(mask, return_counts=True)
        total = mask.size
        sorted_idx = np.argsort(-counts)
        lines = ["Top indoor classes by pixel coverage:"]
        rank = 0
        for idx in sorted_idx:
            cls_id = int(unique[idx])
            if cls_id in OUTDOOR_CLASS_IDS:
                continue
            rank += 1
            if rank > 15:
                break
            pix = int(counts[idx])
            pct = pix / total * 100
            name = ADE20K_NAMES[cls_id] if ADE20K_NAMES and cls_id < len(ADE20K_NAMES) else f"class_{cls_id}"
            name = name.split(",")[0].strip()
            lines.append(f"  {rank}. {name} (id={cls_id}): {pix:,} px ({pct:.1f}%)")
        return lines

    def _serialize_blackspot(self) -> str:
        bs = self.blackspot
        lines = [
            "[BLACKSPOT DETECTION]",
            f"  Detections: {bs.get('num_detections', 0)}",
            f"  Floor area: {bs.get('floor_area', 0):,} px",
            f"  Blackspot area: {bs.get('blackspot_area', 0):,} px",
            f"  Coverage: {bs.get('coverage_percentage', 0):.2f}%",
            f"  Avg confidence: {bs.get('avg_confidence', 0):.2f}",
        ]
        detections = bs.get("detections", [])
        if detections:
            lines.append("  Individual detections:")
            for i, det in enumerate(detections[:10], 1):
                score = det.get("score", det.get("confidence", 0))
                area = det.get("area", 0)
                lines.append(f"    {i}. confidence={score:.2f}, area={area:,} px")
        return "\n".join(lines)

    def _serialize_contrast(self) -> str:
        cs = self.statistics.get("contrast", {})
        lines = [
            "[CONTRAST ANALYSIS] WCAG 2.1 boundary contrast evaluation",
            f"  Total segments analyzed: {cs.get('total_segments', 0)}",
            f"  Pairs evaluated: {cs.get('analyzed_pairs', 0)}",
            f"  Failing pairs: {cs.get('low_contrast_pairs', 0)}",
            f"  Critical issues: {cs.get('critical_issues', 0)}",
            f"  High priority: {cs.get('high_priority_issues', 0)}",
            f"  Medium priority: {cs.get('medium_priority_issues', 0)}",
            f"  Floor-object issues: {cs.get('floor_object_issues', 0)}",
        ]
        issues = self.contrast.get("issues", []) if self.contrast else []
        if issues:
            lines.append("  Specific failing boundaries:")
            for i, issue in enumerate(issues[:8], 1):
                cat1, cat2 = issue.get("categories", ("?", "?"))
                ratio = issue.get("wcag_ratio", 0)
                severity = issue.get("severity", "unknown")
                is_floor = issue.get("is_floor_object", False)
                floor_tag = " [FLOOR-OBJECT]" if is_floor else ""
                lines.append(
                    f"    {i}. {cat1} <> {cat2}: {ratio:.2f}:1 "
                    f"(severity={severity}){floor_tag}"
                )
        return "\n".join(lines)

    def _serialize_safety_surfaces(self) -> str:
        mask = self.segmentation.get("mask") if self.segmentation else None
        if mask is None:
            return ""
        present = []
        unique_ids = set(np.unique(mask).tolist())
        for cls_id, name in SAFETY_SURFACE_IDS.items():
            if cls_id in unique_ids:
                present.append(name)
        if not present:
            return ""
        return "[SAFETY-RELEVANT SURFACES] " + ", ".join(present)


def build_system_prompt(context: AnalysisContext) -> str:
    if not context.has_analysis:
        return (
            "You are the NeuroNest AI assistant, part of a production ML system for "
            "Alzheimer's care environment safety analysis built at Texas State University.\n\n"
            "The system uses EoMT-DINOv3-Large for 150-class semantic segmentation, "
            "a fine-tuned Mask R-CNN for blackspot detection, and vectorized WCAG 2.1 "
            "contrast analysis to identify visual hazards in living spaces.\n\n"
            "No analysis has been performed yet. Ask the user to go to the Live Demo tab, "
            "upload a room image, and click 'Analyze Environment'. Once analysis results "
            "are available, you can explain the findings in detail."
        )

    analysis_data = context.serialize()
    return (
        "You are the NeuroNest AI assistant, a domain expert in Alzheimer's care "
        "environment safety. You are part of a production ML pipeline that has just "
        "analyzed a room image. Below are the analysis results.\n\n"
        f"--- ANALYSIS DATA ---\n{analysis_data}\n--- END DATA ---\n\n"
        "INSTRUCTIONS:\n"
        "- Explain findings in plain language suitable for caregivers and family members. "
        "A caregiver with no technical background should understand your recommendations.\n"
        "- Prioritize safety: fall risks from blackspots, poor contrast between surfaces, "
        "and navigation obstacles are the most critical findings.\n"
        "- Blackspots are dark floor areas that dementia patients perceive as holes, voids, "
        "or drop-offs. This visual-perceptual deficit causes freezing, avoidance, and falls. "
        "Even small blackspot coverage percentages can be dangerous if located in walkways.\n"
        "- For contrast issues, reference WCAG 2.1 standards: Level AA requires 4.5:1 "
        "contrast ratio for text, but for environmental boundaries (floor-wall, furniture-floor), "
        "apply a 3:1 minimum ratio. Critical severity means the boundary is nearly invisible "
        "to someone with visual-perceptual impairment.\n"
        "- Provide actionable remediation: specific changes like adding contrast tape, "
        "improving lighting, replacing dark flooring, using contrasting furniture.\n"
        "- For XAI (Explainable AI) findings: explain what the model is focusing on and how "
        "confident its predictions are. High entropy at boundaries means the model struggles "
        "to distinguish adjacent surfaces — this correlates with low-contrast visibility for "
        "patients. GradCAM shows where the model activates for floor detection, which reveals "
        "how it identifies surfaces for blackspot analysis. Integrated Gradients and Chefer "
        "Relevancy show pixel-level attribution — which pixels most influence predictions.\n"
        "- When discussing technical details (segmentation classes, model architecture), "
        "be precise and reference the actual data provided.\n"
        "- NEVER invent findings not present in the analysis data above. If a type of "
        "analysis was not performed, say so.\n"
        "- Be concise but thorough. Use bullet points for recommendations."
    )


def create_chat_handler(
    state_accessor: Callable[[], Dict[str, Any]],
) -> Callable:
    if not HF_AVAILABLE:
        def _unavailable(message, history):
            return (
                "The AI assistant requires the `huggingface_hub` package. "
                "Install it with: `pip install huggingface_hub`"
            )
        return _unavailable

    def _build_messages(
        system_prompt: str,
        history: List[Dict[str, str]],
        user_message: str,
    ) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": system_prompt}]
        for entry in history:
            messages.append({"role": entry["role"], "content": entry["content"]})
        messages.append({"role": "user", "content": user_message})
        return messages

    def _query_model(
        client: InferenceClient,
        messages: List[Dict[str, str]],
        model: str,
    ) -> str:
        response = client.chat_completion(
            messages=messages,
            model=model,
            max_tokens=1024,
            temperature=0.7,
        )
        return response.choices[0].message.content

    def chat_fn(
        message: str,
        history: List[Dict[str, str]],
    ) -> str:
        if not message or not message.strip():
            return ""

        state = state_accessor()
        context = AnalysisContext.from_results(state)
        system_prompt = build_system_prompt(context)
        messages = _build_messages(system_prompt, history, message)
        client = InferenceClient()

        for model in [PRIMARY_MODEL, FALLBACK_MODEL]:
            try:
                return _query_model(client, messages, model)
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")
                continue

        return (
            "I'm unable to reach the AI service right now. This can happen due to "
            "API rate limits on the free HuggingFace Inference tier. Please try again "
            "in a few moments, or set the `HF_TOKEN` environment variable with your "
            "HuggingFace API token for higher rate limits."
        )

    return chat_fn
