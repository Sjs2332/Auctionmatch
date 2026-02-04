from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import shutil
import os
import uuid
import json
from pathlib import Path
import logging

from analysis_service import execute_analysis
from config import AnalyzerConfig
from utils import setup_logging

# Setup logging
setup_logging("INFO")
logger = logging.getLogger(__name__)

app = FastAPI(title="AuctionMatch Analytical Engine")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the Next.js origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

class AnalysisStatus:
    tasks = {}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"file_id": file_id, "filename": file.filename, "path": str(file_path.absolute())}

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def resolve_file_path(path_str: str) -> str:
    """Resolve a path string to an absolute path, checking project root for defaults."""
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return str(p)
    
    # Try relative to backend (CWD)
    if p.exists():
        return str(p.absolute())
        
    # Try relative to project root (parent of backend)
    root_p = PROJECT_ROOT / p
    if root_p.exists():
        return str(root_p.absolute())
        
    return path_str

@app.post("/analyze")
async def run_analysis(
    background_tasks: BackgroundTasks,
    dealer_files: str = Form(...),  # JSON list of paths
    inventory_file: str = Form(...),
    min_score: float = Form(7.0),
    min_odometer: int = Form(0),
    max_odometer: int = Form(200000),
    top_n: int = Form(50),
    config_overrides: Optional[str] = Form(None)
):
    task_id = str(uuid.uuid4())
    AnalysisStatus.tasks[task_id] = {"status": "processing", "result": None, "error": None}
    
    # Resolve paths to handle repository defaults sent as simple filenames
    dealer_files_list = [resolve_file_path(f) for f in json.loads(dealer_files)]
    resolved_inventory = resolve_file_path(inventory_file)
    
    overrides = json.loads(config_overrides) if config_overrides else {}
    
    config = AnalyzerConfig()
    config.apply_overrides(overrides)
    
    def stringify_keys(obj):
        if isinstance(obj, dict):
            return {str(k): stringify_keys(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [stringify_keys(i) for i in obj]
        if isinstance(obj, float):
            import math
            if math.isnan(obj) or math.isinf(obj):
                return None  # Or 0.0, or "NaN"
        return obj

    def perform_analysis():
        try:
            import numpy as np  # Ensure numpy is accessible
            result = execute_analysis(
                dealer_files=dealer_files_list,
                inventory_file=resolved_inventory,
                min_score=min_score,
                min_odometer=min_odometer,
                max_odometer=max_odometer,
                top_n=top_n,
                log_level="INFO",
                config=config
            )
            
            # Pre-process result to fix tuple keys and numpy types
            serializable_result = stringify_keys(result)
            
            # Save results to disk
            result_path = RESULTS_DIR / f"{task_id}.json"
            with open(result_path, "w") as f:
                class NewEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if typeof_obj := type(obj):
                            if "numpy" in str(typeof_obj):
                                if hasattr(obj, "tolist"):
                                    return obj.tolist()
                                return float(obj) if "float" in str(typeof_obj).lower() else int(obj)
                        return super(NewEncoder, self).default(obj)
                
                json.dump(serializable_result, f, cls=NewEncoder)
            
            AnalysisStatus.tasks[task_id] = {"status": "completed", "result_file": str(result_path)}
        except Exception as e:
            logger.exception("Analysis failed")
            AnalysisStatus.tasks[task_id] = {"status": "failed", "error": str(e)}

    background_tasks.add_task(perform_analysis)
    
    return {"task_id": task_id}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in AnalysisStatus.tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = AnalysisStatus.tasks[task_id]
    if task["status"] == "completed":
        with open(task["result_file"], "r") as f:
            result = json.load(f)
        return {"status": "completed", "result": result}
    
    return task

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
