import os
import tempfile
import base64
import time
import docker
from docker.errors import ImageNotFound
from pathlib import Path
from typing import Dict, Any

DOCKER_IMAGE_NAME = "data-analysis-sandbox:latest"

def _ensure_docker_image(client: docker.DockerClient):
    """
    Ensure the base Docker image with pandas, matplotlib, and seaborn exists.
    If not, builds it.
    """
    try:
        client.images.get(DOCKER_IMAGE_NAME)
    except ImageNotFound:
        print(f"Image {DOCKER_IMAGE_NAME} not found. Building it now. This may take a few minutes...")
        dockerfile_content = """
FROM python:3.10-slim
RUN pip install --no-cache-dir pandas matplotlib seaborn numpy scipy
WORKDIR /workspace
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            df_path = temp_dir_path / "Dockerfile"
            df_path.write_text(dockerfile_content)
            client.images.build(path=str(temp_dir_path), tag=DOCKER_IMAGE_NAME)

def run_python_code_in_sandbox(code: str, data_context: str | None = None) -> Dict[str, Any]:
    """
    Run Python code in a secure, isolated Docker sandbox.
    
    Args:
        code: The Python code to execute.
        data_context: Optional JSON string or raw text data to be written to 'data.json' in the sandbox.
    
    Returns:
        A dictionary containing:
            - stdout: Standard output from the code.
            - stderr: Standard error from the code.
            - images: A list of dicts with 'filename' and 'base64' encoded strings of any images saved in the output directory.
    """
    client = docker.from_env()
    _ensure_docker_image(client)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        
        # Write the python script
        script_path = temp_dir_path / "script.py"
        script_path.write_text(code, encoding="utf-8")
        
        # Write data context if provided
        if data_context:
            data_path = temp_dir_path / "data.json"
            data_path.write_text(data_context, encoding="utf-8")
            
        # Create output directory for images
        output_dir = temp_dir_path / "output"
        output_dir.mkdir()
        
        # We need absolute path for docker volume bind, handling Windows paths appropriately
        # Docker Desktop on Windows handles C:\ paths well when converted to string
        host_dir = str(temp_dir_path.absolute())
        
        volumes = {
            host_dir: {'bind': '/workspace', 'mode': 'rw'}
        }
        
        stdout = ""
        stderr = ""
        
        try:
            # Run the container detached so we can monitor timeout
            container = client.containers.run(
                image=DOCKER_IMAGE_NAME,
                command="python script.py",
                volumes=volumes,
                network_disabled=True,
                mem_limit="512m",
                nano_cpus=1000000000, # 1 CPU core
                working_dir="/workspace",
                detach=True
            )
            
            # Wait for container to finish or timeout (30 seconds)
            start_time = time.time()
            timeout = 30
            timed_out = False
            
            while True:
                container.reload()
                if container.status in ('exited', 'dead', 'removing'):
                    break
                if time.time() - start_time > timeout:
                    container.kill()
                    timed_out = True
                    stderr += "\nSandbox execution error: Execution timed out (30s)."
                    break
                time.sleep(0.5)
            
            # Fetch logs
            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            if not timed_out:
                stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
            else:
                stderr += "\n" + container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
            
            # Cleanup
            container.remove(force=True)
            
        except Exception as e:
            stderr += f"\nSandbox execution error: {str(e)}"
            
        # Collect base64 images from output_dir
        base64_images = []
        for file_path in output_dir.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                with open(file_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                    base64_images.append({
                        "filename": file_path.name,
                        "base64": encoded_string
                    })
                    
        return {
            "stdout": stdout.strip(),
            "stderr": stderr.strip(),
            "images": base64_images
        }
