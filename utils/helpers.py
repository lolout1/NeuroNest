"""Enhanced helper functions for comprehensive NeuroNest analysis reporting."""

from typing import Dict


def generate_analysis_report(results: Dict) -> str:
    """Generate ultra-comprehensive analysis report with robust error handling"""
    report = ["# 🧠 NeuroNest Comprehensive Analysis Report\n"]

    # Executive Summary
    report.append("## 📊 Executive Summary")
    
    # Safely get statistics with defaults
    contrast_stats = results.get('statistics', {}).get('contrast', {})
    blackspot_stats = results.get('statistics', {}).get('blackspot', {})
    
    contrast_issues = contrast_stats.get('total_issues', 0)
    critical_issues = contrast_stats.get('critical_count', 0)
    blackspot_coverage = blackspot_stats.get('coverage_percentage', 0)
    
    total_risk_score = critical_issues * 3 + contrast_issues + (1 if blackspot_coverage > 1 else 0)
    
    if total_risk_score == 0:
        risk_level = "🟢 **EXCELLENT** - Environment exceeds Alzheimer's safety standards"
    elif critical_issues > 0 or total_risk_score > 15:
        risk_level = "🔴 **HIGH RISK** - Immediate intervention required"
    elif total_risk_score > 8:
        risk_level = "🟠 **MODERATE RISK** - Significant improvements needed"
    elif total_risk_score > 3:
        risk_level = "🟡 **LOW RISK** - Minor improvements recommended"
    else:
        risk_level = "🟢 **GOOD** - Environment is generally safe"
    
    report.append(f"**Overall Risk Assessment:** {risk_level}")
    report.append(f"**Risk Score:** {total_risk_score}/30")
    report.append("")

    # Segmentation results
    if results.get('segmentation'):
        stats = results['statistics'].get('segmentation', {})
        report.append(f"## 🎯 Object Detection & Segmentation")
        report.append(f"- **Objects identified:** {stats.get('num_classes', 'N/A')}")
        report.append(f"- **Image resolution:** {stats.get('image_size', 'N/A')}")
        report.append("")

    # Enhanced blackspot analysis
    if results.get('blackspot'):
        report.append(f"## ⚫ Blackspot Hazard Analysis (Floors Only)")
        report.append(f"- **Total floor area:** {blackspot_stats.get('floor_area_pixels', 0):,} pixels")
        report.append(f"- **Blackspot area:** {blackspot_stats.get('blackspot_area_pixels', 0):,} pixels")
        report.append(f"- **Hazard coverage:** {blackspot_stats.get('coverage_percentage', 0):.2f}% of floor area")
        report.append(f"- **Individual blackspots:** {blackspot_stats.get('num_detections', 0)}")
        report.append(f"- **Detection method:** Color-based analysis (black/dark areas only)")

        # Enhanced risk assessment
        coverage = blackspot_stats.get('coverage_percentage', 0)
        if coverage > 10:
            report.append(f"- **🚨 EXTREME HAZARD:** Severe blackspot coverage - immediate action required")
        elif coverage > 5:
            report.append(f"- **⚠️ HIGH HAZARD:** Significant blackspot coverage detected")
        elif coverage > 1:
            report.append(f"- **⚠️ MODERATE HAZARD:** Notable blackspot coverage")
        elif coverage > 0:
            report.append(f"- **✓ LOW HAZARD:** Minimal blackspot coverage")
        else:
            report.append(f"- **✅ NO HAZARD:** No blackspots detected on floors")
        report.append("")

    # Ultra-comprehensive contrast analysis
    if results.get('contrast'):
        report.append(f"## 🎨 Ultra-Comprehensive Contrast Analysis")
        report.append(f"- **Analysis method:** Ultra-high sensitivity color detection")
        report.append(f"- **Objects analyzed:** {contrast_stats.get('total_segments', 0)}")
        report.append(f"- **Object pairs checked:** {contrast_stats.get('total_pairs_checked', 0)}")
        report.append(f"- **Adjacent pairs:** {contrast_stats.get('adjacent_pairs', 0)}")
        report.append(f"- **Total contrast violations:** {contrast_stats.get('total_issues', 0)}")
        report.append("")
        
        report.append(f"### 📈 Issue Breakdown")
        report.append(f"- **🔴 Critical issues:** {contrast_stats.get('critical_count', 0)} (safety-critical)")
        report.append(f"- **🟠 High priority:** {contrast_stats.get('high_count', 0)} (navigation impact)")
        report.append(f"- **🟡 Medium priority:** {contrast_stats.get('medium_count', 0)} (comfort impact)")
        report.append(f"- **🔵 Low priority:** {contrast_stats.get('low_count', 0)} (minor issues)")
        report.append(f"- **✅ Good contrasts:** {contrast_stats.get('good_contrast_count', 0)}")
        report.append("")
        
        report.append(f"### 🔬 Detection Thresholds")
        report.append(f"- **WCAG standard:** {contrast_stats.get('wcag_threshold', 4.5)}:1")
        report.append(f"- **Alzheimer's optimized:** {contrast_stats.get('alzheimer_threshold', 7.0)}:1")
        report.append("")

        # Critical issues detailed breakdown
        critical_issues_list = results.get('contrast', {}).get('critical_issues', [])
        if critical_issues_list:
            report.append("### 🚨 CRITICAL ISSUES (Immediate Action Required)")
            for i, issue in enumerate(critical_issues_list[:10], 1):
                cats = f"{issue['categories'][0]} ↔ {issue['categories'][1]}"
                ratio = issue.get('contrast_ratio', 0)
                
                report.append(f"**{i}. {cats}**")
                report.append(f"   • WCAG contrast: {ratio:.1f}:1 (need ≥7:1)")
                
                if 'issues' in issue and issue['issues']:
                    for sub_issue in issue['issues'][:3]:
                        report.append(f"   • {sub_issue}")
                report.append("")
            
            if len(critical_issues_list) > 10:
                remaining = len(critical_issues_list) - 10
                report.append(f"*...and {remaining} more critical issues*")
                report.append("")

        # High priority issues summary
        high_issues_list = results.get('contrast', {}).get('high_issues', [])
        if high_issues_list:
            report.append("### ⚠️ HIGH PRIORITY ISSUES")
            for i, issue in enumerate(high_issues_list[:5], 1):
                cats = f"{issue['categories'][0]} ↔ {issue['categories'][1]}"
                ratio = issue.get('contrast_ratio', 0)
                report.append(f"{i}. **{cats}**: {ratio:.1f}:1 contrast")
            
            if len(high_issues_list) > 5:
                remaining = len(high_issues_list) - 5
                report.append(f"*...and {remaining} more high priority issues*")
            report.append("")

        # Good contrasts (safely handle missing key)
        good_contrasts_list = results.get('contrast', {}).get('good_contrasts', [])
        if good_contrasts_list:
            report.append("### ✅ Excellent Contrasts (Meeting Alzheimer's Standards)")
            for good in good_contrasts_list[:5]:
                cats = f"{good['categories'][0]} ↔ {good['categories'][1]}"
                ratio = good.get('contrast_ratio', 0)
                report.append(f"• **{cats}**: {ratio:.1f}:1 contrast ratio")
            
            if len(good_contrasts_list) > 5:
                remaining = len(good_contrasts_list) - 5
                report.append(f"*...and {remaining} more excellent contrasts*")
            report.append("")

    # Evidence-based recommendations
    report.append("## 📋 Evidence-Based Action Plan")

    # Immediate actions for critical issues
    if critical_issues > 0:
        report.append("### 🚨 IMMEDIATE ACTIONS (Within 24 Hours)")
        report.append("- **PRIORITY 1:** Address all critical contrast violations immediately")
        report.append("- **PRIORITY 2:** Implement temporary high-contrast markers on safety-critical boundaries")
        report.append("- **PRIORITY 3:** Increase lighting in all critical areas to minimum 1000 lux")
        report.append("- **PRIORITY 4:** Remove or replace any identical/similar colored adjacent objects")
        report.append("")

    # Blackspot mitigation
    if blackspot_coverage > 0:
        report.append("### 🖤 Blackspot Elimination Strategy")
        if blackspot_coverage > 5:
            report.append("- **URGENT:** Complete removal of all dark flooring materials")
            report.append("- **Install:** High-contrast transition strips at all blackspot boundaries")
        report.append("- **Lighting:** Install LED strips with minimum 750 lux at floor level")
        report.append("- **Contrast aids:** Place light-colored, high-contrast rugs over dark areas")
        report.append("- **Navigation:** Add tactile markers and bright edge treatments")
        report.append("- **Monitoring:** Weekly assessment of blackspot areas for safety")
        report.append("")

    # Color contrast optimization
    if contrast_issues > 0:
        report.append("### 🎨 Color Contrast Optimization")
        report.append("- **Target:** Achieve minimum 7:1 contrast ratio for all object boundaries")
        report.append("- **Similar colors:** Replace any objects with similar colors immediately")
        report.append("- **Saturation:** Use highly saturated colors, eliminate muted/pastel tones")
        report.append("- **Texture:** Add contrasting textures where color changes aren't possible")
        report.append("- **Patterns:** Use bold patterns to differentiate similar-colored items")
        report.append("")

    # Overall environment assessment
    report.append("## 🎯 Environmental Safety Conclusion")
    
    if total_risk_score == 0:
        report.append("🏆 **OUTSTANDING ENVIRONMENT**")
        report.append("This space exceeds all Alzheimer's accessibility standards.")
    elif critical_issues > 0:
        report.append("🚨 **IMMEDIATE INTERVENTION REQUIRED**")
        report.append(f"Critical safety issues detected require immediate attention.")
    elif total_risk_score > 10:
        report.append("⚠️ **SUBSTANTIAL IMPROVEMENTS NEEDED**")
        report.append("Multiple issues may cause confusion and navigation difficulties.")
    elif total_risk_score > 5:
        report.append("📋 **TARGETED IMPROVEMENTS RECOMMENDED**")
        report.append("Focused interventions will significantly enhance safety.")
    else:
        report.append("✅ **GOOD ENVIRONMENT WITH ENHANCEMENT OPPORTUNITIES**")
        report.append("Environment is generally suitable with areas for optimization.")

    # Technical summary
    report.append(f"\n---")
    report.append(f"*Analysis: Ultra-sensitive detection • Floors-only blackspots • Evidence-based recommendations*")

    return "\n".join(report)
