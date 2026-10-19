import io
import json
import sys
from pathlib import Path

nb_path = Path(r"c:\Users\Байсултан\Desktop\HW3AI\HW3.ipynb")
nb = json.loads(nb_path.read_text(encoding='utf-8'))
ns = {}

for cell in nb['cells']:
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if not src.strip():
        continue
    stdout = io.StringIO()
    stderr = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = stdout, stderr
    try:
        exec(compile(src, '<notebook-cell>', 'exec'), ns, ns)
        text = stdout.getvalue()
        err_text = stderr.getvalue()
        outputs = []
        if text:
            outputs.append({"output_type": "stream", "name": "stdout", "text": text})
        if err_text:
            outputs.append({"output_type": "stream", "name": "stderr", "text": err_text})
        cell['outputs'] = outputs
        cell['execution_count'] = 1
    except Exception as e:
        cell['execution_count'] = 1
        cell['outputs'] = [{
            "output_type": "error",
            "ename": type(e).__name__,
            "evalue": str(e),
            "traceback": [f"{type(e).__name__}: {e}"],
        }]
    finally:
        sys.stdout, sys.stderr = old_out, old_err

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('Executed notebook and wrote outputs to', nb_path)
