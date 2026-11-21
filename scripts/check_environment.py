"""Report dependency versions, CUDA availability, and installed-package modifications."""
import importlib.metadata as metadata
import hashlib,base64,json
import torch
report={'cuda_available':torch.cuda.is_available(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'packages':{}}
for name in ['mjlab','rsl-rl-lib','mujoco','mujoco-warp','warp-lang','tensordict']:
    dist=metadata.distribution(name)
    changed=[]
    for entry in dist.files or []:
        if entry.suffix=='.py' and entry.hash and entry.hash.mode=='sha256':
            path=dist.locate_file(entry)
            if path.is_file():
                digest=base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).decode().rstrip('=')
                if digest!=entry.hash.value:changed.append(str(entry))
    report['packages'][name]={'version':dist.version,'modified_python_files':changed}
print(json.dumps(report,indent=2))
