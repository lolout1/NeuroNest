"""Helper functions for the NeuroNest application."""

from typing import Dict


def generate_analysis_report(results: Dict) -> str:
    """Generate enhanced analysis report text"""
    report = ["# NeuroNest Analysis Report\n"]

    # Segmentation results
    if results['segmentation']:
        stats = results['statistics'].get('segmentation', {})
        report.append(f"## 🎯 Semantic Segmentation")
        report.append(f"- **Objects detected:** {stats.get('num_classes', 'N/A')}")
        report.append(f"- **Image size:** {stats.get('image_size', 'N/A')}")
        report.append("")

    # Enhanced blackspot results
    if results['blackspot']:
        bs_stats = results['statistics'].get('blackspot', {})
        report.append(f"## ⚫ Blackspot Detection")
        report.append(f"- **Floor area:** {bs_stats.get('floor_area_pixels', 0):,} pixels")
        report.append(f"- **Blackspot area:** {bs_stats.get('blackspot_area_pixels', 0):,} pixels")
        report.append(f"- **Coverage:** {bs_stats.get('coverage_percentage', 0):.2f}% of floor")
        report.append(f"- **Individual blackspots:** {bs_stats.get('num_detections', 0)}")
        report.append(f"- **Average confidence:** {bs_stats.get('avg_confidence', 0):.2f}")

        # Risk assessment
        coverage = bs_stats.get('coverage_percentage', 0)
        if coverage > 5:
            report.append(f"- **⚠️ Risk Level:** HIGH - Significant blackspot coverage detected")
        elif coverage > 1:
            report.append(f"- **⚠️ Risk Level:** MEDIUM - Moderate blackspot coverage")
        elif coverage > 0:
            report.append(f"- **✓ Risk Level:** LOW - Minimal blackspot coverage")
        else:
            report.append(f"- **✓ Risk Level:** NONE - No blackspots detected")
        report.append("")

    # Contrast analysis results (existing code)
    if results['contrast']:
        contrast_stats = results['statistics'].get('contrast', {})
        report.append(f"## 🎨 Contrast Analysis")
        report.append(f"- **Total issues:** {contrast_stats.get('total_issues', 0)}")
        report.append(f"- **🔴 Critical:** {contrast_stats.get('critical_issues', 0)}")
        report.append(f"- **🟠 High priority:** {contrast_stats.get('high_priority_issues', 0)}")
        report.append(f"- **🟡 Medium priority:** {contrast_stats.get('medium_priority_issues', 0)}")
        report.append("")

        # Add detailed issues
        if results['contrast']['critical_issues']:
            report.append("### Critical Issues (Immediate Attention Required)")
            for issue in results['contrast']['critical_issues']:
                cats = f"{issue['categories'][0]}-{issue['categories'][1]}"
                ratio = issue['contrast_ratio']
                report.append(f"- **{cats}**: {ratio:.1f}:1 contrast ratio")
                report.append(f"  _{issue['description']}_")
            report.append("")

    # Enhanced recommendations
    report.append("## 📋 Recommendations")

    # Blackspot-specific recommendations
    if results['blackspot']:
        coverage = results['statistics'].get('blackspot', {}).get('coverage_percentage', 0)
        if coverage > 0:
            report.append("### Blackspot Mitigation")
            report.append("- Remove or replace dark-colored floor materials in detected areas")
            report.append("- Improve lighting in blackspot areas")
            report.append("- Consider using light-colored rugs or mats to cover blackspots")
            report.append("- Add visual cues like contrasting tape around problem areas")
            report.append("")

    # Contrast-specific recommendations
    contrast_issues = results['statistics'].get('contrast', {}).get('total_issues', 0)
    if contrast_issues > 0:
        report.append("### Contrast Improvements")
        report.append("- Increase lighting in low-contrast areas")
        report.append("- Use contrasting colors for furniture and floors")
        report.append("- Add visual markers for important boundaries")
        report.append("- Consider color therapy guidelines for dementia")
        report.append("")

    if coverage == 0 and contrast_issues == 0:
        report.append("✅ **Environment Assessment: EXCELLENT**")
        report.append("No significant safety issues detected. This environment appears well-suited for individuals with Alzheimer's.")

    return "\n".join(report)
