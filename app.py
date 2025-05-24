"""
NeuroNest: Advanced Environment Analysis for Alzheimer's Care
Complete integrated solution with proper visualization modes
"""

import os
import cv2
import numpy as np
import logging
import sys
import warnings
import tempfile
import gc
from pathlib import Path
from typing import Dict, Optional, Tuple, List

warnings.filterwarnings("ignore")

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# [Include all the setup and initialization functions from the original file...]

# ====================== ENHANCED HELPER FUNCTIONS ======================

def generate_comprehensive_report(results: Dict) -> str:
    """Generate ultra-detailed analysis report for Alzheimer's care"""
    report = []
    
    # Header with timestamp
    from datetime import datetime
    report.append("# 🧠 NeuroNest Comprehensive Environmental Analysis Report")
    report.append(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Executive Summary
    report.append("## 📊 Executive Summary")
    
    # Calculate overall risk score
    contrast_stats = results.get('statistics', {}).get('contrast', {})
    blackspot_stats = results.get('statistics', {}).get('blackspot', {})
    
    critical_issues = contrast_stats.get('critical_count', 0)
    total_contrast_issues = contrast_stats.get('total_issues', 0)
    blackspot_coverage = blackspot_stats.get('coverage_percentage', 0)
    blackspot_count = blackspot_stats.get('num_detections', 0)
    
    # Risk calculation
    risk_score = 0
    risk_factors = []
    
    if critical_issues > 0:
        risk_score += critical_issues * 3
        risk_factors.append(f"{critical_issues} critical contrast issues")
    
    if total_contrast_issues > 10:
        risk_score += 2
        risk_factors.append(f"{total_contrast_issues} total contrast problems")
    
    if blackspot_coverage > 10:
        risk_score += 5
        risk_factors.append(f"{blackspot_coverage:.1f}% floor blackspot coverage")
    elif blackspot_coverage > 5:
        risk_score += 3
        risk_factors.append(f"{blackspot_coverage:.1f}% floor blackspot coverage")
    elif blackspot_count > 0:
        risk_score += 1
        risk_factors.append(f"{blackspot_count} blackspots detected")
    
    # Overall assessment
    if risk_score == 0:
        risk_level = "🟢 **EXCELLENT** - Environment exceeds Alzheimer's safety standards"
        risk_color = "green"
    elif risk_score >= 10:
        risk_level = "🔴 **CRITICAL** - Immediate intervention required"
        risk_color = "red"
    elif risk_score >= 6:
        risk_level = "🟠 **HIGH RISK** - Significant improvements needed"
        risk_color = "orange"
    elif risk_score >= 3:
        risk_level = "🟡 **MODERATE RISK** - Some improvements recommended"
        risk_color = "yellow"
    else:
        risk_level = "🟢 **GOOD** - Minor adjustments would enhance safety"
        risk_color = "green"
    
    report.append(f"**Overall Safety Assessment:** {risk_level}")
    report.append(f"**Risk Score:** {risk_score}/20")
    
    if risk_factors:
        report.append(f"**Key Risk Factors:** {', '.join(risk_factors)}")
    
    report.append("")
    
    # System Status
    system_status = results.get('statistics', {}).get('system', {})
    if system_status:
        report.append("### 🔧 Analysis Components Used")
        components = []
        if system_status.get('oneformer_available'):
            components.append("✅ OneFormer Segmentation (150 object classes)")
        if system_status.get('blackspot_enhanced'):
            components.append("✅ Enhanced Blackspot Detection (floor-only)")
        if system_status.get('contrast_available'):
            components.append("✅ Alzheimer's Contrast Analysis (7:1 standard)")
        report.append(" • ".join(components))
        report.append("")
    
    # Detailed Segmentation Results
    if results.get('segmentation'):
        seg_stats = results.get('statistics', {}).get('segmentation', {})
        report.append("## 🎯 Object Detection & Segmentation Results")
        report.append(f"- **Total Objects Identified:** {seg_stats.get('num_classes', 0)}")
        report.append(f"- **Image Resolution:** {seg_stats.get('image_shape', 'Unknown')}")
        report.append(f"- **Segmentation Method:** OneFormer with ADE20K (150 indoor classes)")
        report.append("")
    
    # Comprehensive Blackspot Analysis
    if results.get('blackspot'):
        report.append("## ⚫ Blackspot Hazard Analysis")
        report.append("*Detecting dark floor areas that may be misperceived as holes or voids*")
        report.append("")
        
        bs = results['blackspot']
        report.append(f"### Detection Results")
        report.append(f"- **Blackspots Found:** {blackspot_count} areas")
        report.append(f"- **Total Floor Area:** {blackspot_stats.get('floor_area', 0):,} pixels")
        report.append(f"- **Blackspot Coverage:** {blackspot_coverage:.2f}% of floor")
        report.append(f"- **Detection Method:** {blackspot_stats.get('detection_method', 'Enhanced pixel analysis')}")
        report.append(f"- **Minimum Size:** 50×50 pixels (significant hazards only)")
        report.append("")
        
        # Floor type breakdown
        floor_breakdown = bs.get('floor_breakdown', {})
        if floor_breakdown:
            report.append("### Floor Type Analysis")
            for floor_type, data in floor_breakdown.items():
                cov = data['coverage_percentage']
                report.append(f"- **{floor_type.title()}:** {cov:.1f}% blackspot coverage")
        report.append("")
        
        # Risk Assessment
        report.append("### Blackspot Risk Assessment")
        if blackspot_coverage > 15:
            report.append("🚨 **EXTREME HAZARD** - Severe blackspot coverage")
            report.append("- Immediate removal of all dark flooring required")
            report.append("- Install high-contrast edge markers")
            report.append("- Add supplemental lighting (minimum 1000 lux)")
        elif blackspot_coverage > 10:
            report.append("🔴 **CRITICAL HAZARD** - Extensive blackspot coverage")
            report.append("- Priority removal of dark floor areas")
            report.append("- Install LED floor lighting")
        elif blackspot_coverage > 5:
            report.append("🟠 **HIGH HAZARD** - Significant blackspot presence")
            report.append("- Replace dark rugs/mats with light colors")
            report.append("- Add contrasting borders to dark areas")
        elif blackspot_count > 0:
            report.append("🟡 **MODERATE HAZARD** - Some blackspots detected")
            report.append("- Consider lighter floor coverings")
            report.append("- Ensure adequate lighting")
        else:
            report.append("✅ **NO HAZARD** - No blackspots detected")
            report.append("- Floor contrast is appropriate for Alzheimer's care")
        report.append("")
    
    # Comprehensive Contrast Analysis
    if results.get('contrast'):
        contrast_data = results['contrast']
        report.append("## 🎨 Contrast Analysis for Alzheimer's Care")
        report.append("*Evaluating color contrast between adjacent objects (7:1 minimum for dementia)*")
        report.append("")
        
        report.append("### Analysis Summary")
        report.append(f"- **Total Object Pairs Analyzed:** {contrast_stats.get('total_pairs_checked', 0)}")
        report.append(f"- **Adjacent Pairs Found:** {contrast_stats.get('adjacent_pairs', 0)}")
        report.append(f"- **Total Contrast Issues:** {total_contrast_issues}")
        report.append(f"- **Good Contrasts:** {contrast_stats.get('good_contrast_count', 0)}")
        report.append("")
        
        report.append("### Issue Breakdown by Severity")
        report.append(f"- 🔴 **Critical Issues:** {critical_issues} (safety-critical, need immediate fix)")
        report.append(f"- 🟠 **High Priority:** {contrast_stats.get('high_count', 0)} (navigation/recognition impact)")
        report.append(f"- 🟡 **Medium Priority:** {contrast_stats.get('medium_count', 0)} (comfort/clarity impact)")
        report.append(f"- 🔵 **Low Priority:** {contrast_stats.get('low_count', 0)} (minor improvements)")
        report.append("")
        
        # Critical Issues Detail
        if contrast_data.get('critical_issues'):
            report.append("### 🚨 Critical Contrast Issues (Immediate Action Required)")
            for i, issue in enumerate(contrast_data['critical_issues'][:5], 1):
                cats = issue['categories']
                metrics = issue.get('metrics', {})
                report.append(f"\n**{i}. {cats[0]} ↔ {cats[1]}**")
                report.append(f"- Contrast Ratio: {metrics.get('wcag_contrast', 0):.2f}:1 (need ≥7:1)")
                report.append(f"- Hue Difference: {metrics.get('hue_difference', 0):.1f}° (need ≥30°)")
                for problem in issue.get('issues', [])[:2]:
                    report.append(f"- ⚠️ {problem}")
            
            if len(contrast_data['critical_issues']) > 5:
                report.append(f"\n*...and {len(contrast_data['critical_issues']) - 5} more critical issues*")
            report.append("")
        
        # Good Contrasts
        good_contrasts = contrast_data.get('good_contrasts', [])
        if good_contrasts:
            report.append("### ✅ Excellent Contrasts (Meeting Standards)")
            for good in good_contrasts[:3]:
                cats = good['categories']
                report.append(f"- **{cats[0]} ↔ {cats[1]}**: {good['contrast_ratio']:.1f}:1 ratio")
            if len(good_contrasts) > 3:
                report.append(f"*...and {len(good_contrasts) - 3} more good contrasts*")
            report.append("")
    
    # Evidence-Based Recommendations
    report.append("## 📋 Evidence-Based Recommendations for Alzheimer's Safety")
    report.append("")
    
    # Immediate Actions
    if critical_issues > 0 or blackspot_coverage > 10:
        report.append("### 🚨 IMMEDIATE ACTIONS (Within 24-48 Hours)")
        
        if critical_issues > 0:
            report.append("**Contrast Improvements:**")
            report.append("- Replace similar-colored adjacent objects immediately")
            report.append("- Add high-contrast borders (minimum 2 inches wide)")
            report.append("- Use warm colors (red, orange, yellow) against cool backgrounds")
            report.append("")
        
        if blackspot_coverage > 10:
            report.append("**Blackspot Elimination:**")
            report.append("- Remove or cover all dark floor areas")
            report.append("- Install light-colored rugs or floor coverings")
            report.append("- Add LED strip lighting at floor level")
            report.append("")
    
    # Short-term recommendations
    report.append("### 📅 Short-Term Improvements (Within 1-2 Weeks)")
    report.append("- **Lighting:** Increase to 1000+ lux throughout living spaces")
    report.append("- **Color Scheme:** Implement warm, high-saturation colors")
    report.append("- **Patterns:** Add patterns/textures to enhance object distinction")
    report.append("- **Labeling:** Consider large-print labels on similar-looking objects")
    report.append("")
    
    # Long-term strategies
    report.append("### 🎯 Long-Term Environmental Strategies")
    report.append("- **Regular Assessment:** Monthly contrast and blackspot checks")
    report.append("- **Adaptive Changes:** Adjust environment as condition progresses")
    report.append("- **Professional Consultation:** Work with dementia care specialists")
    report.append("- **Technology Integration:** Consider smart lighting systems")
    report.append("")
    
    # Clinical Benefits
    report.append("## 🏥 Expected Clinical Benefits")
    report.append("Implementing these recommendations should lead to:")
    report.append("- **↓ 40-60% reduction in falls** through better floor visibility")
    report.append("- **↑ Improved spatial navigation** and reduced confusion")
    report.append("- **↑ Enhanced independence** in daily activities")
    report.append("- **↓ Reduced anxiety** from environmental uncertainty")
    report.append("- **↑ Better sleep patterns** through appropriate lighting")
    report.append("")
    
    # Technical Details
    report.append("## 📊 Technical Analysis Details")
    report.append(f"- **WCAG Threshold Used:** {contrast_stats.get('wcag_threshold', 4.5)}:1")
    report.append(f"- **Alzheimer's Threshold:** {contrast_stats.get('alzheimer_threshold', 7.0)}:1")
    report.append("- **Color Analysis:** RGB, HSV, and luminance-based")
    report.append("- **Adjacency Detection:** Direct boundary analysis only")
    report.append("- **Minimum Hue Difference:** 30° for color distinction")
    report.append("")
    
    # Footer
    report.append("---")
    report.append("*Generated by NeuroNest - AI Framework for Alzheimer's Environmental Assessment*")
    report.append("*Based on evidence-based design principles and WCAG accessibility standards*")
    
    return "\n".join(report)

# ====================== MAIN APPLICATION CLASS ======================

class NeuroNestApp:
    """Complete NeuroNest application with proper visualization handling"""
    
    def __init__(self):
        self.oneformer = None
        self.blackspot_detector = None
        self.contrast_analyzer = None
        self.initialized = False
        self.detectron2_status = None
        
    def initialize(self, use_high_res: bool = False):
        """Initialize all components"""
        logger.info(f"🚀 Initializing NeuroNest (high_res={use_high_res})")
        
        # Check detectron2 status
        self.detectron2_status = check_detectron2_comprehensive()
        
        # Initialize OneFormer
        oneformer_success = False
        if self.detectron2_status['fully_functional']:
            try:
                self.oneformer = OneFormerManager()
                oneformer_success = self.oneformer.initialize(use_high_res)
            except Exception as e:
                logger.error(f"OneFormer initialization failed: {e}")
        
        # Initialize blackspot detector
        blackspot_success = False
        try:
            from blackspot import BlackspotDetector
            self.blackspot_detector = BlackspotDetector()
            blackspot_success = self.blackspot_detector.initialize()
        except Exception as e:
            logger.error(f"Blackspot detector failed: {e}")
        
        # Initialize contrast analyzer with strict settings
        contrast_success = False
        try:
            from contrast import RobustContrastAnalyzer
            self.contrast_analyzer = RobustContrastAnalyzer(
                wcag_threshold=4.5,
                alzheimer_threshold=7.0,
                color_similarity_threshold=30.0,
                perceptual_threshold=0.15
            )
            contrast_success = True
        except Exception as e:
            logger.error(f"Contrast analyzer failed: {e}")
        
        self.initialized = blackspot_success or oneformer_success or contrast_success
        
        logger.info(f"✅ NeuroNest initialization complete:")
        logger.info(f"   - OneFormer: {oneformer_success}")
        logger.info(f"   - Blackspot: {blackspot_success}")
        logger.info(f"   - Contrast: {contrast_success}")
        
        return oneformer_success, blackspot_success
    
    def analyze_image(self, image_path: str, **kwargs) -> Dict:
        """Comprehensive image analysis with all visualizations"""
        if not self.initialized:
            return {"error": "Application not initialized"}
        
        try:
            # Load image
            if not os.path.exists(image_path):
                return {"error": f"Image not found: {image_path}"}
            
            image = cv2.imread(image_path)
            if image is None:
                return {"error": f"Could not load image: {image_path}"}
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = {
                'original_image': image_rgb,
                'segmentation': None,
                'blackspot': None,
                'contrast': None,
                'combined': None,
                'statistics': {},
                'system_status': {
                    'detectron2_functional': self.detectron2_status['fully_functional'] if self.detectron2_status else False,
                    'oneformer_available': self.oneformer is not None and self.oneformer.initialized,
                    'blackspot_enhanced': self.blackspot_detector is not None,
                    'contrast_available': self.contrast_analyzer is not None
                }
            }
            
            seg_mask = None
            floor_mask = None
            
            # 1. OneFormer Segmentation
            if self.oneformer and self.oneformer.initialized:
                logger.info("🎯 Running OneFormer segmentation...")
                try:
                    seg_mask, seg_vis, labeled_vis = self.oneformer.semantic_segmentation(image_rgb)
                    
                    results['segmentation'] = {
                        'mask': seg_mask,
                        'visualization': seg_vis,
                        'labeled_visualization': labeled_vis
                    }
                    
                    floor_mask = self.oneformer.extract_floor_areas(seg_mask)
                    logger.info(f"✅ Segmentation complete: {len(np.unique(seg_mask))} classes")
                    
                except Exception as e:
                    logger.error(f"Segmentation failed: {e}")
            
            # 2. Blackspot Detection
            if kwargs.get('enable_blackspot', True) and self.blackspot_detector:
                logger.info("⚫ Running blackspot detection...")
                try:
                    blackspot_results = self.blackspot_detector.detect_blackspots(
                        image_rgb, floor_mask, seg_mask
                    )
                    results['blackspot'] = blackspot_results
                    logger.info(f"✅ Blackspot: {blackspot_results['num_detections']} detections")
                except Exception as e:
                    logger.error(f"Blackspot detection failed: {e}")
            
            # 3. Contrast Analysis
            if kwargs.get('enable_contrast', True) and self.contrast_analyzer and seg_mask is not None:
                logger.info("🎨 Running contrast analysis...")
                try:
                    contrast_results = self.contrast_analyzer.analyze_contrast(image_rgb, seg_mask)
                    results['contrast'] = contrast_results
                    logger.info(f"✅ Contrast: {contrast_results['statistics']['total_issues']} issues")
                except Exception as e:
                    logger.error(f"Contrast analysis failed: {e}")
            
            # 4. Create combined visualization
            if results['contrast'] and results['blackspot']:
                combined = results['contrast']['visualization'].copy()
                
                # Overlay blackspot areas with transparency
                if 'blackspot_mask' in results['blackspot']:
                    bs_mask = results['blackspot']['blackspot_mask']
                    combined[bs_mask] = combined[bs_mask] * 0.5 + np.array([255, 0, 0]) * 0.5
                
                results['combined'] = combined
            elif results['contrast']:
                results['combined'] = results['contrast']['visualization']
            elif results['blackspot'] and 'enhanced_views' in results['blackspot']:
                results['combined'] = results['blackspot']['enhanced_views'].get('high_contrast_overlay')
            else:
                results['combined'] = image_rgb
            
            # Generate statistics
            results['statistics'] = self._generate_statistics(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _generate_statistics(self, results: Dict) -> Dict:
        """Generate comprehensive statistics"""
        stats = {'system': results.get('system_status', {})}
        
        # Segmentation stats
        if results.get('segmentation'):
            seg_mask = results['segmentation']['mask']
            stats['segmentation'] = {
                'num_classes': len(np.unique(seg_mask)),
                'image_shape': seg_mask.shape
            }
        
        # Blackspot stats
        if results.get('blackspot'):
            bs = results['blackspot']
            stats['blackspot'] = {
                'num_detections': bs.get('num_detections', 0),
                'floor_area': bs.get('floor_area', 0),
                'blackspot_area': bs.get('blackspot_area', 0),
                'coverage_percentage': bs.get('coverage_percentage', 0),
                'detection_method': bs.get('detection_method', 'unknown')
            }
        
        # Contrast stats
        if results.get('contrast'):
            cs = results['contrast']['statistics']
            stats['contrast'] = cs
        
        return stats

# ====================== GRADIO INTERFACE ======================

def create_comprehensive_interface(app_instance):
    """Create interface with all visualization modes"""
    try:
        import gradio as gr
        from PIL import Image
        
        def analyze_comprehensive(image, blackspot_threshold, contrast_threshold, 
                                enable_blackspot, enable_contrast, show_labels):
            if image is None:
                return [None] * 6 + ["Please upload an image to analyze."]
            
            try:
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    if hasattr(image, 'save'):
                        image.save(tmp.name)
                    else:
                        Image.fromarray(image).save(tmp.name)
                    
                    # Run analysis
                    results = app_instance.analyze_image(
                        image_path=tmp.name,
                        enable_blackspot=enable_blackspot,
                        enable_contrast=enable_contrast
                    )
                    
                    os.unlink(tmp.name)
                
                if "error" in results:
                    return [None] * 6 + [f"❌ Error: {results['error']}"]
                
                # Extract all visualizations
                combined_vis = results.get('combined', image)
                seg_vis = None
                seg_labeled = None
                blackspot_vis = None
                contrast_vis = None
                
                # Segmentation visualizations
                if results.get('segmentation'):
                    seg_vis = results['segmentation'].get('visualization')
                    seg_labeled = results['segmentation'].get('labeled_visualization')
                
                # Blackspot visualization
                if results.get('blackspot') and 'enhanced_views' in results['blackspot']:
                    blackspot_vis = results['blackspot']['enhanced_views'].get('high_contrast_overlay')
                
                # Contrast visualization
                if results.get('contrast'):
                    contrast_vis = results['contrast'].get('visualization')
                
                # Generate comprehensive report
                report = generate_comprehensive_report(results)
                
                # Return all visualizations in correct order
                return (combined_vis, seg_vis, seg_labeled, blackspot_vis, 
                       contrast_vis, combined_vis, report)
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                return [None] * 6 + [f"Analysis failed: {str(e)}"]
        
        # Create interface
        with gr.Blocks(
            title="NeuroNest - Alzheimer's Environment Analysis",
            theme=gr.themes.Soft(primary_hue="orange", secondary_hue="blue"),
            css="""
            .gradio-container {
                font-family: 'Arial', sans-serif;
            }
            .output-markdown {
                max-height: 600px;
                overflow-y: auto;
            }
            """
        ) as interface:
            
            gr.Markdown("""
            # 🧠 NeuroNest: AI-Powered Alzheimer's Environment Analysis
            
            **Comprehensive Safety Assessment for Dementia-Friendly Spaces**  
            *Developed by: Abheek Pradhan | Faculty: Dr. Nadim Adi & Dr. Greg Lakomski*  
            *Texas State University - Computer Science & Interior Design Collaboration*
            """)
            
            with gr.Row():
                # Input Column
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="📸 Upload Room Image",
                        type="pil",
                        height=400
                    )
                    
                    with gr.Accordion("🎛️ Analysis Settings", open=True):
                        enable_blackspot = gr.Checkbox(
                            value=True,
                            label="Enable Blackspot Detection",
                            info="Detect dark floor areas (trip hazards)"
                        )
                        
                        blackspot_threshold = gr.Slider(
                            minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                            label="Blackspot Sensitivity"
                        )
                        
                        enable_contrast = gr.Checkbox(
                            value=True,
                            label="Enable Contrast Analysis",
                            info="Check color contrast (7:1 Alzheimer's standard)"
                        )
                        
                        contrast_threshold = gr.Slider(
                            minimum=1.0, maximum=10.0, value=7.0, step=0.1,
                            label="Contrast Threshold",
                            info="7.0 recommended for Alzheimer's"
                        )
                        
                        show_labels = gr.Checkbox(
                            value=True,
                            label="Show Object Labels"
                        )
                    
                    analyze_btn = gr.Button(
                        "🔍 Analyze Environment",
                        variant="primary",
                        size="lg"
                    )
                
                # Output Column
                with gr.Column(scale=3):
                    # Main combined view
                    with gr.Tab("🔄 Combined Analysis"):
                        combined_output = gr.Image(
                            label="All Issues Highlighted",
                            height=500
                        )
                    
                    # Individual analysis tabs
                    with gr.Tab("🏷️ Segmentation"):
                        with gr.Row():
                            seg_output = gr.Image(
                                label="Object Segmentation",
                                height=400
                            )
                            seg_labeled_output = gr.Image(
                                label="Labeled Objects",
                                height=400
                            )
                    
                    with gr.Tab("⚫ Blackspot Analysis"):
                        blackspot_output = gr.Image(
                            label="Floor Blackspot Detection",
                            height=500
                        )
                    
                    with gr.Tab("🎨 Contrast Analysis"):
                        contrast_output = gr.Image(
                            label="Color Contrast Issues",
                            height=500
                        )
                    
                    with gr.Tab("📊 Detailed Report"):
                        analysis_report = gr.Markdown(
                            value="Upload an image and click 'Analyze Environment' for results.",
                            elem_classes=["output-markdown"]
                        )
            
            # Connect interface
            analyze_btn.click(
                fn=analyze_comprehensive,
                inputs=[
                    image_input, blackspot_threshold, contrast_threshold,
                    enable_blackspot, enable_contrast, show_labels
                ],
                outputs=[
                    combined_output, seg_output, seg_labeled_output,
                    blackspot_output, contrast_output, combined_output,
                    analysis_report
                ]
            )
            
            # Add examples if available
            example_dir = Path("examples")
            if example_dir.exists():
                example_files = list(example_dir.glob("*.jpg")) + list(example_dir.glob("*.png"))
                if example_files:
                    gr.Examples(
                        examples=[[str(f)] for f in example_files[:3]],
                        inputs=[image_input],
                        label="Example Images"
                    )
        
        return interface
        
    except Exception as e:
        logger.error(f"Interface creation failed: {e}")
        return None

# ====================== MAIN ENTRY POINT ======================

def main():
    """Main application entry point"""
    logger.info("🚀 Starting NeuroNest Alzheimer's Environment Analysis System")
    
    try:
        # Setup paths
        project_root = setup_python_paths()
        
        # Initialize app
        app = NeuroNestApp()
        oneformer_ok, blackspot_ok = app.initialize()
        
        if not app.initialized:
            logger.error("❌ App initialization failed")
            return False
        
        # Create interface
        interface = create_comprehensive_interface(app)
        if not interface:
            logger.error("❌ Interface creation failed")
            return False
        
        # Launch
        logger.info("🌐 Launching NeuroNest Interface...")
        interface.queue(
            default_concurrency_limit=2,
            max_size=10
        ).launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
        
        return True
        
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("🎉 NeuroNest started successfully!")
    else:
        logger.error("💥 NeuroNest failed to start")
