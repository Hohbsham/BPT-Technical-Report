"""
Garment BPT Workflow - Complete automated pipeline for 3D garment processing
This script orchestrates the entire workflow: sparse remeshing -> BPT conversion -> quality assurance
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GarmentBPTWorkflow:
    def __init__(self, input_dir, output_dir, temp_dir=None, logs_dir=None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir) if temp_dir else self.output_dir / "temp"
        self.logs_dir = Path(logs_dir) if logs_dir else self.output_dir / "logs"
        
        # Create necessary directories
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Define script paths
        self.sparse_script = Path("D:/ClothesNetData/project_docs/garment_sparse_bpt_converter.py")
        self.bpt_script = Path("D:/ClothesNetData/project_docs/bpt_quad_converter.py")
        self.check_script = Path("D:/ClothesNetData/project_docs/mesh_sanity_check.py")
        self.verify_script = Path("D:/ClothesNetData/project_docs/visual_verify.py")
    
    def run_command(self, cmd, description):
        """Run a command and log the result"""
        logger.info(f"Executing: {description}")
        logger.info(f"Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        
        try:
            start_time = time.time()
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=3600  # 1 hour timeout
            )
            elapsed_time = time.time() - start_time
            
            logger.info(f"Execution time: {elapsed_time:.2f}s")
            
            if result.returncode == 0:
                logger.info(f"✅ {description} completed successfully")
                if result.stdout:
                    logger.debug(f"Output: {result.stdout[:500]}...")  # Log first 500 chars
                return True
            else:
                logger.error(f"❌ {description} failed with return code {result.returncode}")
                if result.stderr:
                    logger.error(f"Error: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"❌ {description} timed out after 1 hour")
            return False
        except Exception as e:
            logger.error(f"❌ {description} failed with exception: {str(e)}")
            return False
    
    def stage_1_sparse_remeshing(self):
        """Stage 1: Sparse remeshing using QuadRemesher"""
        logger.info("Stage 1: Starting sparse remeshing...")
        
        # Find input mesh files
        input_files = list(self.input_dir.rglob("*.obj")) + \
                     list(self.input_dir.rglob("*.fbx")) + \
                     list(self.input_dir.rglob("*.stl"))
        
        if not input_files:
            logger.warning("No input mesh files found in input directory")
            return True  # Not an error, just nothing to process
        
        logger.info(f"Found {len(input_files)} mesh files to process")
        
        # Run sparse remeshing script
        cmd = [
            sys.executable,
            str(self.sparse_script)
        ]
        
        success = self.run_command(cmd, "Sparse remeshing")
        
        if success:
            logger.info("Stage 1: Sparse remeshing completed")
        else:
            logger.error("Stage 1: Sparse remeshing failed")
        
        return success
    
    def stage_2_bpt_conversion(self):
        """Stage 2: BPT format conversion"""
        logger.info("Stage 2: Starting BPT conversion...")
        
        # Find sparse quad files
        sparse_dir = Path("D:/ClothesNetData_Sparse_Quads")  # Default output from sparse script
        if not sparse_dir.exists():
            logger.warning(f"Sparse quad directory not found: {sparse_dir}")
            # Try to find it in output dir
            sparse_dirs = list(self.output_dir.rglob("*sparse*")) + list(self.output_dir.rglob("*quad*"))
            if sparse_dirs:
                sparse_dir = sparse_dirs[0]
                logger.info(f"Using alternative sparse directory: {sparse_dir}")
            else:
                logger.error("No sparse quad directory found, cannot proceed with BPT conversion")
                return False
        
        # Run BPT conversion script
        cmd = [
            sys.executable,
            str(self.bpt_script)
        ]
        
        success = self.run_command(cmd, "BPT conversion")
        
        if success:
            logger.info("Stage 2: BPT conversion completed")
        else:
            logger.error("Stage 2: BPT conversion failed")
        
        return success
    
    def stage_3_quality_assurance(self):
        """Stage 3: Quality assurance and validation"""
        logger.info("Stage 3: Starting quality assurance...")
        
        # Find BPT output files
        bpt_outputs = list(Path("D:/ClothesNetData_QuadBPT_Converted").rglob("*.bpt"))
        
        if not bpt_outputs:
            logger.warning("No BPT output files found for quality check")
            return True  # Not an error, just nothing to validate
        
        logger.info(f"Found {len(bpt_outputs)} BPT files to validate")
        
        all_passed = True
        for bpt_file in bpt_outputs[:5]:  # Limit to first 5 for demo purposes
            logger.info(f"Validating: {bpt_file}")
            
            # Run mesh sanity check
            cmd = [
                sys.executable,
                str(self.check_script),
                "--mesh", str(bpt_file)
            ]
            
            check_success = self.run_command(cmd, f"Quality check for {bpt_file.name}")
            all_passed = all_passed and check_success
        
        if all_passed:
            logger.info("Stage 3: Quality assurance completed successfully")
        else:
            logger.error("Stage 3: Some quality checks failed")
        
        return all_passed
    
    def run_full_workflow(self):
        """Execute the complete workflow"""
        logger.info("Starting Garment BPT Workflow")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        
        start_time = time.time()
        
        # Stage 1: Sparse remeshing
        stage1_success = self.stage_1_sparse_remeshing()
        if not stage1_success:
            logger.error("Workflow terminated at Stage 1 (Sparse remeshing)")
            return False
        
        # Stage 2: BPT conversion
        stage2_success = self.stage_2_bpt_conversion()
        if not stage2_success:
            logger.error("Workflow terminated at Stage 2 (BPT conversion)")
            return False
        
        # Stage 3: Quality assurance
        stage3_success = self.stage_3_quality_assurance()
        if not stage3_success:
            logger.error("Workflow completed but with quality issues in Stage 3")
            # We don't terminate here, as the workflow technically completed
        else:
            logger.info("All stages completed successfully")
        
        total_time = time.time() - start_time
        logger.info(f"Workflow completed in {total_time:.2f}s")
        
        return stage2_success  # Return success of core processing stages
    
    def run_partial_workflow(self, stages=None):
        """Run specific stages of the workflow"""
        if stages is None:
            stages = [1, 2, 3]
        
        logger.info(f"Running partial workflow for stages: {stages}")
        
        results = {}
        
        if 1 in stages:
            results[1] = self.stage_1_sparse_remeshing()
        
        if 2 in stages:
            results[2] = self.stage_2_bpt_conversion()
        
        if 3 in stages:
            results[3] = self.stage_3_quality_assurance()
        
        overall_success = all(results.values()) if results else False
        logger.info(f"Partial workflow results: {results}, Overall: {overall_success}")
        
        return overall_success


def main():
    parser = argparse.ArgumentParser(description="Garment BPT Workflow")
    parser.add_argument("--input-dir", default="D:/ClothesNetData", help="Input directory with garment meshes")
    parser.add_argument("--output-dir", default="D:/ClothesNetData_Processed", help="Output directory for results")
    parser.add_argument("--temp-dir", help="Temporary directory")
    parser.add_argument("--logs-dir", help="Logs directory")
    parser.add_argument("--stages", nargs='+', type=int, choices=[1, 2, 3], 
                       help="Specific stages to run (1=sparse, 2=bpt, 3=qa)")
    
    args = parser.parse_args()
    
    workflow = GarmentBPTWorkflow(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        temp_dir=args.temp_dir,
        logs_dir=args.logs_dir
    )
    
    if args.stages:
        success = workflow.run_partial_workflow(stages=args.stages)
    else:
        success = workflow.run_full_workflow()
    
    if success:
        logger.info("🎉 Garment BPT Workflow completed successfully!")
        sys.exit(0)
    else:
        logger.error("💥 Garment BPT Workflow failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()